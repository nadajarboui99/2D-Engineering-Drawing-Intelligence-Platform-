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

# Seed zero-shot architectures (scaffold — HF weights download on first use).
# check_import lists EVERY module the wrapper needs (comma-separated); all must
# import for the model to report as available.
def _arch(id, label, task, module, cls, install, check, desc):
    return {"id": id, "label": label, "task": task, "arch": module,
            "wrapper_module": module, "wrapper_class": cls, "weights": None,
            "install_cmd": install, "check_import": check,
            "description": desc, "source": "builtin"}

BUILTIN_ARCHS = [
    _arch("table-transformer", "Table Transformer (DETR)", "tables",
          "table_transformer", "TableTransformerDetector",
          "pip install transformers timm torch", "transformers,timm,torch",
          "Microsoft Table Transformer — table detection trained on PubTables-1M (zero-shot, generic)."),
    _arch("grounding-dino", "Grounding DINO (open-vocab)", "dimensions",
          "grounding_dino", "GroundingDINODetector",
          "pip install transformers torch", "transformers,torch",
          "Open-vocabulary, text-prompted detector. Zero-shot dimensions via prompt 'dimension . measurement . numeric value . tolerance .'."),
    # Florence-2 is a unified detector — offered for both tasks (same wrapper, "<OD>").
    _arch("florence2-tables", "Florence-2 (unified OD)", "tables",
          "florence2", "Florence2Detector",
          "pip install transformers timm einops torch", "transformers,timm,einops,torch",
          "Microsoft Florence-2 — unified zero-shot detection via the <OD> task."),
    _arch("florence2-dimensions", "Florence-2 (unified OD)", "dimensions",
          "florence2", "Florence2Detector",
          "pip install transformers timm einops torch", "transformers,timm,einops,torch",
          "Microsoft Florence-2 — unified zero-shot detection via the <OD> task."),
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
    """True only if EVERY listed module (comma-separated) imports."""
    for mod in (check_import or "").split(","):
        mod = mod.strip().split(".")[0]
        if not mod:
            continue
        try:
            importlib.import_module(mod)
        except Exception:
            return False
    return True


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
