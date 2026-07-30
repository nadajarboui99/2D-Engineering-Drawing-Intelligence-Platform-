"""
OCR Pipeline
============
Flow: image → YOLO detection → crop patches → OCR → JSON

Usage:
    python run_ocr.py --config configs/ocr.yaml
    python run_ocr.py --config configs/ocr.yaml --ocr_model tesseract
    python run_ocr.py --config configs/ocr.yaml --task dimensions
    python run_ocr.py --config configs/ocr.yaml --image path/to/image.jpg
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
from PIL import Image
from pathlib import Path

# make models/ importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

# make detection module importable
DETECTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Table_dimensions_detection")
sys.path.insert(0, DETECTION_DIR)


# model factories

def load_detector(cfg: dict, task: str):
    from models.yolov11 import YOLOv11Detector
    weights_key = f"weights_{task}"
    weights = cfg["detection"][weights_key]
    # Resolve relative path from ocr/ folder
    weights = os.path.join(DETECTION_DIR, os.path.relpath(weights, "../Table_dimensions_detection"))
    if not os.path.exists(weights):
        print(f"[ERROR] Weights not found: {weights}")
        sys.exit(1)
    return YOLOv11Detector(weights=weights)


def load_ocr(cfg: dict, ocr_model_override: str = None):
    """
    Factory — add new OCR models here with a new elif block.
    """
    model_name = ocr_model_override or cfg["ocr"]["model"]

    if model_name == "easyocr":
        from easyocr_model import EasyOCRModel
        s = cfg["ocr"]["easyocr"]
        return EasyOCRModel(languages=s["languages"], gpu=s["gpu"])

    elif model_name == "tesseract":
        from tesseract_model import TesseractModel
        s = cfg["ocr"]["tesseract"]
        return TesseractModel(lang=s["lang"], config=s["config"])

    else:
        raise ValueError(f"Unknown OCR model '{model_name}'. Options: easyocr | tesseract")


# crop helper

def crop_patch(image: Image.Image, box: list, padding: int = 4) -> Image.Image:
    w, h = image.size
    x1 = max(0, int(box[0]) - padding)
    y1 = max(0, int(box[1]) - padding)
    x2 = min(w, int(box[2]) + padding)
    y2 = min(h, int(box[3]) + padding)
    return image.crop((x1, y1, x2, y2))


# process one image

def process_image(image_path: str, detector, ocr_model, cfg: dict,
                  task: str, output_dir: str) -> dict:

    image      = Image.open(image_path).convert("RGB")
    image_name = Path(image_path).stem

    preds  = detector.predict([np.array(image)],
                              conf_threshold=cfg["detection"]["conf_threshold"],
                              imgsz=cfg["detection"]["imgsz"])
    pred   = preds[0]
    boxes  = pred["boxes"].tolist()
    scores = pred["scores"].tolist()

    if not boxes:
        print(f"  [WARN] No detections in {image_name}")
        return {"image": image_name, "task": task, "detections": []}

    detections = []
    crops_dir  = os.path.join(output_dir, "crops", image_name)
    if cfg["io"]["save_crops"]:
        os.makedirs(crops_dir, exist_ok=True)

    for i, (box, score) in enumerate(zip(boxes, scores)):
        crop = crop_patch(image, box)
        text = ocr_model.read(crop)

        detections.append({
            "id":         i,
            "class":      task,
            "confidence": round(score, 4),
            "bbox":       [round(v, 2) for v in box],
            "text":       text,
        })

        if cfg["io"]["save_crops"]:
            crop.save(os.path.join(crops_dir, f"crop_{i:03d}.jpg"))

        print(f"  [{i+1}/{len(boxes)}] conf={score:.2f} → '{text}'")

    result = {
        "image":      image_name,
        "task":       task,
        "ocr_model":  cfg["ocr"]["model"],
        "detections": detections,
    }

    if cfg["io"]["save_json"]:
        json_path = os.path.join(output_dir, "json", f"{image_name}.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/ocr.yaml")
    parser.add_argument("--ocr_model", default=None, help="easyocr | tesseract")
    parser.add_argument("--task",      default=None, help="tables | dimensions")
    parser.add_argument("--image",     default=None, help="single image path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task = args.task or cfg["detection"]["task"]
    if args.ocr_model:
        cfg["ocr"]["model"] = args.ocr_model

    output_dir = cfg["io"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  TASK      : {task.upper()}")
    print(f"  OCR MODEL : {cfg['ocr']['model']}")
    print(f"{'='*55}\n")

    detector  = load_detector(cfg, task)
    ocr_model = load_ocr(cfg, args.ocr_model)

    if args.image:
        image_paths = [args.image]
    else:
        images_dir  = cfg["io"]["images_dir"]
        image_paths = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    print(f"Processing {len(image_paths)} images...\n")

    all_results = []
    for i, path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] {Path(path).name}")
        result = process_image(path, detector, ocr_model, cfg, task, output_dir)
        all_results.append(result)

    if cfg["io"]["save_json"]:
        out = os.path.join(output_dir, f"{task}_{cfg['ocr']['model']}_results.json")
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Combined results → {out}")

    total = sum(len(r["detections"]) for r in all_results)
    print(f"✅ Done. {total} detections processed.")


if __name__ == "__main__":
    main()