"""
Detection model registry for CUSTOM ARCHITECTURES (beyond the YOLO/RT-DETR
checkpoints auto-discovered under runs/). Each entry points at a wrapper class in
Table_dimensions_detection/models/ implementing `predict(images, conf_threshold,
imgsz) -> [{"boxes", "scores", "labels"}]`, imported dynamically — so adding a
new architecture is just a wrapper file + a registry entry, per task.
"""
import os
import json
import importlib

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "detection_registry.json")

# seed architectures, deps are not installed until a model is actually picked
BUILTIN_ARCHS = [
    {
        "id":            "table-transformer",
        "label":         "Table Transformer (DETR)",
        "task":          "tables",
        "arch":          "table_transformer",
        "wrapper_module": "table_transformer",
        "wrapper_class":  "TableTransformerDetector",
        "weights":        None,   # downloads from HuggingFace on first use
        "install_cmd":    "pip install transformers torch timm",
        "check_import":   "transformers",
        "description":    "Microsoft Table Transformer — table detection trained on PubTables-1M (generic, off-the-shelf).",
        "source":         "builtin",
    },
]


def _load_custom() -> list:
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _save_custom(data: list):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def is_installed(check_import: str) -> bool:
    try:
        importlib.import_module((check_import or "").split(".")[0])
        return True
    except Exception:
        return False


def list_models(task: str = None) -> list:
    models = BUILTIN_ARCHS + _load_custom()
    if task and task != "both":
        models = [m for m in models if m.get("task") == task]
    return [{**m, "installed": is_installed(m.get("check_import", ""))} for m in models]


def get_model(model_id: str) -> dict | None:
    return next((m for m in (BUILTIN_ARCHS + _load_custom()) if m["id"] == model_id), None)


def register_custom(id, label, task, arch, wrapper_module, wrapper_class,
                    weights="", install_cmd="", check_import="", description="") -> dict:
    custom = [m for m in _load_custom() if m["id"] != id]
    entry = {"id": id, "label": label, "task": task, "arch": arch,
             "wrapper_module": wrapper_module, "wrapper_class": wrapper_class,
             "weights": weights or None, "install_cmd": install_cmd,
             "check_import": check_import, "description": description, "source": "custom"}
    custom.append(entry)
    _save_custom(custom)
    return entry


def remove_custom(model_id: str):
    _save_custom([m for m in _load_custom() if m["id"] != model_id])
