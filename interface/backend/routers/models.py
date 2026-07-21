"""
Dynamic model installation and registry.
Runs pip install in a subprocess and registers the model in a local JSON registry.
"""
import os
import json
import subprocess
import sys

from fastapi import APIRouter
from pydantic import BaseModel
from core.job_manager import create_job, run_job
from fastapi import BackgroundTasks

router = APIRouter()

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "model_registry.json")


def load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {"detection": [], "ocr": [], "vlm": []}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(data: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


class ModelInstallRequest(BaseModel):
    name:         str
    install_cmd:  str
    module_path:  str
    module_class: str
    task:         str  # "detection" | "ocr" | "vlm"


@router.get("/registry")
def get_registry():
    return load_registry()


@router.post("/install")
async def install_model(req: ModelInstallRequest, background: BackgroundTasks):
    job = create_job(f"Install model · {req.name}")

    def _run(job, req):
        job.log(f"Running: {req.install_cmd}")
        result = subprocess.run(
            req.install_cmd.split(),
            capture_output=True, text=True,
            executable=sys.executable.replace("python", "pip") if "pip" in req.install_cmd else None
        )
        if result.returncode != 0:
            raise RuntimeError(f"Install failed:\n{result.stderr}")

        job.log("Install successful. Registering model...")
        registry = load_registry()
        task_list = registry.setdefault(req.task, [])

        # Remove existing entry with same name
        registry[req.task] = [m for m in task_list if m["name"] != req.name]
        registry[req.task].append({
            "name":         req.name,
            "module_path":  req.module_path,
            "module_class": req.module_class,
        })
        save_registry(registry)
        job.log(f"Model '{req.name}' registered under '{req.task}'.")
        return {"name": req.name, "task": req.task}

    background.add_task(run_job, job, _run, req=req)
    return {"job_id": job.job_id}


@router.get("/available/{task}")
def get_available_models(task: str):
    """Returns built-in + dynamically installed models for a task."""
    builtin = {
        "detection": ["yolov11", "rtdetr"],
        "ocr":       ["easyocr", "tesseract"],
        "vlm":       ["claude", "gpt4o", "qwen_vl"],
    }
    registry = load_registry()
    dynamic  = [m["name"] for m in registry.get(task, [])]
    return {"models": builtin.get(task, []) + dynamic}