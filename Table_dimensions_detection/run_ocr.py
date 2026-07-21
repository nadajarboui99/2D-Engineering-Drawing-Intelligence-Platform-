"""
OCR Pipeline — Crop mode
=========================
Flow: image → YOLO detection → crop patches → OCR → JSON output

Usage:
    # Tables with EasyOCR
    python run_ocr.py --config configs/ocr.yaml

    # Switch OCR model
    python run_ocr.py --config configs/ocr.yaml --ocr_model tesseract

    # Switch detection task
    python run_ocr.py --config configs/ocr.yaml --task dimensions

    # Run on a single image
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Model factories ───────────────────────────────────────────────────────────

def load_detector(cfg: dict, task: str):
    from models.yolov11 import YOLOv11Detector
    weights_key = f"weights_{task}"
    weights = cfg["detection"][weights_key]
    if not os.path.exists(weights):
        print(f"[ERROR] Weights not found: {weights}")
        sys.exit(1)
    return YOLOv11Detector(weights=weights)


def load_ocr(cfg: dict, ocr_model_override: str = None):
    model_name = ocr_model_override or cfg["ocr"]["model"]

    if model_name == "easyocr":
        from ocr.easyocr_model import EasyOCRModel
        settings = cfg["ocr"]["easyocr"]
        return EasyOCRModel(languages=settings["languages"], gpu=settings["gpu"])

    elif model_name == "tesseract":
        from ocr.tesseract_model import TesseractModel
        settings = cfg["ocr"]["tesseract"]
        return TesseractModel(lang=settings["lang"], config=settings["config"])

    else:
        raise ValueError(f"Unknown OCR model '{model_name}'. Use: easyocr | tesseract")


# ── Crop helper ───────────────────────────────────────────────────────────────

def crop_patch(image: Image.Image, box: list, padding: int = 4) -> Image.Image:
    """
    Crop a region from the image given [x1, y1, x2, y2].
    Adds small padding and clamps to image boundaries.
    """
    w, h = image.size
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    return image.crop((x1, y1, x2, y2))


# ── Process one image ─────────────────────────────────────────────────────────

def process_image(image_path: str, detector, ocr_model, cfg: dict,
                  task: str, output_dir: str) -> dict:

    image = Image.open(image_path).convert("RGB")
    image_name = Path(image_path).stem

    # 1. Detect
    conf_thresh = cfg["detection"]["conf_threshold"]
    imgsz       = cfg["detection"]["imgsz"]
    preds       = detector.predict([np.array(image)],
                                   conf_threshold=conf_thresh, imgsz=imgsz)
    pred        = preds[0]  # single image

    boxes  = pred["boxes"].tolist()
    scores = pred["scores"].tolist()
    labels = pred["labels"].tolist()

    if not boxes:
        print(f"  [WARN] No detections in {image_name}")
        return {"image": image_name, "detections": []}

    # 2. Crop + OCR each box
    detections = []
    crops_dir  = os.path.join(output_dir, "crops", image_name)
    if cfg["io"]["save_crops"]:
        os.makedirs(crops_dir, exist_ok=True)

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        crop = crop_patch(image, box)

        # OCR
        text = ocr_model.read(crop)

        detection = {
            "id":         i,
            "class":      task,
            "confidence": round(score, 4),
            "bbox":       [round(v, 2) for v in box],
            "text":       text,
        }
        detections.append(detection)

        # Save crop
        if cfg["io"]["save_crops"]:
            crop_path = os.path.join(crops_dir, f"crop_{i:03d}.jpg")
            crop.save(crop_path)

        print(f"  [{i+1}/{len(boxes)}] conf={score:.2f} → '{text}'")

    result = {
        "image":      image_name,
        "task":       task,
        "ocr_model":  cfg["ocr"]["model"],
        "detections": detections,
    }

    # Save per-image JSON
    if cfg["io"]["save_json"]:
        json_path = os.path.join(output_dir, "json", f"{image_name}.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OCR Pipeline")
    parser.add_argument("--config",     default="configs/ocr.yaml")
    parser.add_argument("--ocr_model",  default=None,
                        help="Override OCR model: easyocr | tesseract")
    parser.add_argument("--task",       default=None,
                        help="Override task: tables | dimensions")
    parser.add_argument("--image",      default=None,
                        help="Run on a single image instead of a folder")
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

    # Load models
    detector  = load_detector(cfg, task)
    ocr_model = load_ocr(cfg, args.ocr_model)

    # Collect images
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
    for i, image_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] {Path(image_path).name}")
        result = process_image(image_path, detector, ocr_model, cfg,
                               task, output_dir)
        all_results.append(result)

    # Save combined JSON (ready to pass to VLM)
    if cfg["io"]["save_json"]:
        combined_path = os.path.join(output_dir, f"{task}_{cfg['ocr']['model']}_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Combined results saved to: {combined_path}")

    print(f"✅ Done. {sum(len(r['detections']) for r in all_results)} total detections processed.")


if __name__ == "__main__":
    main()
    