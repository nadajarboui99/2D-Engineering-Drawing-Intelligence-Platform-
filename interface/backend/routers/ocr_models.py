"""
OCR model management — list, install, register custom models.
"""
import subprocess
import sys

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from core.ocr_registry import list_models, register_custom, remove_custom
from core.job_manager import create_job, run_job

router = APIRouter()


class CustomModelPayload(BaseModel):
    id:           str
    label:        str
    description:  str
    install_cmd:  str
    check_import: str


@router.get("/list")
def get_models():
    return list_models()


@router.post("/install/{model_id}")
async def install_model(model_id: str, background: BackgroundTasks):
    """Install a built-in model by running its install_cmd."""
    models = list_models()
    model  = next((m for m in models if m["id"] == model_id), None)
    if not model:
        return {"error": f"Model '{model_id}' not found"}

    job = create_job(f"Install OCR · {model['label']}")

    def _run(job, model):
        cmds = model["install_cmd"].split("&&")
        for cmd in cmds:
            cmd = cmd.strip()
            job.log(f"Running: {cmd}")
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.stdout:
                job.log(result.stdout[-300:])
            if result.returncode != 0:
                raise RuntimeError(f"Failed: {result.stderr[-300:]}")
        job.log(f"{model['label']} installed successfully.")
        return {"model_id": model_id}

    background.add_task(run_job, job, _run, model=model)
    return {"job_id": job.job_id}


@router.post("/register")
def register_model(payload: CustomModelPayload):
    """Register a custom OCR model (with its own wrapper already implemented)."""
    entry = register_custom(
        id=payload.id, label=payload.label,
        description=payload.description,
        install_cmd=payload.install_cmd,
        check_import=payload.check_import,
    )
    return {"ok": True, "entry": entry}


@router.delete("/{model_id}")
def delete_model(model_id: str):
    remove_custom(model_id)
    return {"ok": True}