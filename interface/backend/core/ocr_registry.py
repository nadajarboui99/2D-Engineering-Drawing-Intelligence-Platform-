"""
OCR model registry.
Tracks which OCR models are installed and available.
Each model needs: a pip package, an import check, and a wrapper class.
"""
import os
import json
import importlib

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "ocr_registry.json")

# wrapper_module/wrapper_class point at a BaseOCR subclass in ocr/models/
BUILTIN_MODELS = [
    {
        "id":          "easyocr",
        "label":       "EasyOCR",
        "description": "Deep learning OCR — handles rotated and stylized text well",
        "install_cmd": "pip install easyocr",
        "check_import": "easyocr",
        "wrapper_module": "easyocr_model", "wrapper_class": "EasyOCRModel",
        "source":      "builtin",
    },
    {
        "id":          "tesseract",
        "label":       "Tesseract",
        "description": "Classic OCR engine — fast, good for clean printed text",
        "install_cmd": "pip install pytesseract && brew install tesseract",
        "check_import": "pytesseract",
        "wrapper_module": "tesseract_model", "wrapper_class": "TesseractModel",
        "source":      "builtin",
    },
    {
        "id":          "trocr",
        "label":       "TrOCR (Microsoft)",
        "description": "Transformer OCR (HuggingFace). Best on single-line/cropped text; downloads weights automatically.",
        "install_cmd": "pip install transformers torch pillow",
        "check_import": "transformers",
        "wrapper_module": "trocr_model", "wrapper_class": "TrOCRModel",
        "source":      "builtin",
    },
    {
        "id":          "paddleocr",
        "label":       "PaddleOCR",
        "description": "Full OCR pipeline (detection + recognition), strong on dense documents.",
        "install_cmd": "pip install paddlepaddle paddleocr",
        "check_import": "paddleocr",
        "wrapper_module": "paddleocr_model", "wrapper_class": "PaddleOCRModel",
        "source":      "builtin",
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
        importlib.import_module(check_import.split(".")[0])
        return True
    except ImportError:
        return False


def list_models() -> list:
    custom = _load_custom()
    all_models = BUILTIN_MODELS + custom
    return [
        {**m, "installed": is_installed(m["check_import"])}
        for m in all_models
    ]


def get_model(model_id: str) -> dict | None:
    return next((m for m in (BUILTIN_MODELS + _load_custom()) if m["id"] == model_id), None)


def register_custom(id: str, label: str, description: str,
                    install_cmd: str, check_import: str,
                    wrapper_module: str = "", wrapper_class: str = "") -> dict:
    custom = _load_custom()
    custom = [m for m in custom if m["id"] != id]
    entry = {
        "id": id, "label": label, "description": description,
        "install_cmd": install_cmd, "check_import": check_import,
        "wrapper_module": wrapper_module, "wrapper_class": wrapper_class,
        "source": "custom",
    }
    custom.append(entry)
    _save_custom(custom)
    return entry


def remove_custom(model_id: str):
    custom = _load_custom()
    custom = [m for m in custom if m["id"] != model_id]
    _save_custom(custom)