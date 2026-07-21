"""
Weights management — upload, list, delete detection model weights.
"""
import os
import shutil

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.weights_registry import register, list_weights, delete_weights, get_weights_path

router   = APIRouter()
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


@router.post("/upload")
async def upload_weights(
    file: UploadFile = File(...),
    name: str = Form(...),
    task: str = Form(...),   # "tables" | "dimensions"
):
    if not file.filename.endswith(".pt"):
        raise HTTPException(400, "Only .pt files are supported")

    dest = os.path.join(WEIGHTS_DIR, f"{task}_{name}.pt")
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    entry = register(name=name, task=task, path=dest, source="upload")
    return {"ok": True, "entry": entry}


@router.post("/register-path")
async def register_existing(name: str, task: str, path: str):
    """Register an existing .pt file already on disk (e.g. from a training run)."""
    if not os.path.exists(path):
        raise HTTPException(404, f"File not found: {path}")
    entry = register(name=name, task=task, path=path, source="local")
    return {"ok": True, "entry": entry}


@router.get("/list")
def get_list(task: str = None):
    return list_weights(task)


@router.delete("/{entry_id}")
def delete(entry_id: str):
    delete_weights(entry_id)
    return {"ok": True}