"""
OCR model registry.
Tracks which OCR models are installed and available.
Each model needs: a pip package, an import check, and a wrapper class.
"""
import os
import json
import importlib

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "ocr_registry.json")

# Built-in supported models
BUILTIN_MODELS = [
    {
        "id":          "easyocr",
        "label":       "EasyOCR",
        "description": "Deep learning OCR — handles rotated and stylized text well",
        "install_cmd": "pip install easyocr",
        "check_import": "easyocr",
        "source":      "builtin",
    },
    {
        "id":          "tesseract",
        "label":       "Tesseract",
        "description": "Classic OCR engine — fast, good for clean printed text",
        "install_cmd": "pip install pytesseract && brew install tesseract",
        "check_import": "pytesseract",
        "source":      "builtin",
    },
    {
        "id":          "trocr",
        "label":       "TrOCR (Microsoft)",
        "description": "Transformer-based OCR — state of the art for printed text, downloads weights from HuggingFace automatically",
        "install_cmd": "pip install transformers torch pillow",
        "check_import": "transformers",
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


def register_custom(id: str, label: str, description: str,
                    install_cmd: str, check_import: str) -> dict:
    custom = _load_custom()
    custom = [m for m in custom if m["id"] != id]
    entry = {
        "id": id, "label": label, "description": description,
        "install_cmd": install_cmd, "check_import": check_import,
        "source": "custom",
    }
    custom.append(entry)
    _save_custom(custom)
    return entry


def remove_custom(model_id: str):
    custom = _load_custom()
    custom = [m for m in custom if m["id"] != model_id]
    _save_custom(custom)