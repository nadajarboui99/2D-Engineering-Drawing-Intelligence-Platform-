import os
import yaml

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

DETECTION_DIR = os.path.join(BASE, "Table_dimensions_detection")
OCR_DIR       = os.path.join(BASE, "ocr")
VLM_DIR       = os.path.join(BASE, "vlm")

PATHS = {
    "detection_tables":     os.path.join(DETECTION_DIR, "configs", "tables.yaml"),
    "detection_dimensions": os.path.join(DETECTION_DIR, "configs", "dimensions.yaml"),
    "ocr":                  os.path.join(OCR_DIR,       "configs", "ocr.yaml"),
    "vlm":                  os.path.join(VLM_DIR,       "configs", "vlm.yaml"),
    "feature_schema":       os.path.join(VLM_DIR,       "configs", "feature_schema.yaml"),
    "prompts_whole_image":     os.path.join(VLM_DIR, "configs", "prompts", "whole_image.txt"),
    "prompts_whole_image_ocr": os.path.join(VLM_DIR, "configs", "prompts", "whole_image_ocr.txt"),
    "prompts_cropped_ocr":     os.path.join(VLM_DIR, "configs", "prompts", "cropped_ocr.txt"),
}


def load(key: str) -> dict:
    path = PATHS.get(key, "")
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save(key: str, data: dict):
    path = PATHS[key]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def read_text(key: str) -> str:
    path = PATHS.get(key, "")
    if not path or not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


def write_text(key: str, text: str):
    path = PATHS[key]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)