"""
VLM Feature Extraction Pipeline
================================
Compares 3 input modes against a fixed feature schema:
  1. whole_image      → raw full image only
  2. whole_image_ocr   → full image + OCR text (from whole-image OCR)
  3. cropped_ocr       → cropped patches + per-patch OCR text

Usage:
    # Run all configured modes on all images
    python run_vlm.py --config configs/vlm.yaml

    # Run a single mode
    python run_vlm.py --config configs/vlm.yaml --mode whole_image

    # Run on a single image
    python run_vlm.py --config configs/vlm.yaml --image_name drawing_001
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts"))

from prompt_builder import load_schema, build_prompt


# model factory

def load_vlm(cfg: dict, model_override: str = None):
    """
    Factory — add new VLMs here with a new elif block.
    """
    model_name = model_override or cfg["vlm"]["model"]

    if model_name == "claude":
        from claude_vlm import ClaudeVLM
        s = cfg["vlm"]["claude"]
        return ClaudeVLM(model=s["model_name"], max_tokens=s["max_tokens"])


    else:
        raise ValueError(f"Unknown VLM '{model_name}'. Add it to load_vlm() in run_vlm.py")


# OCR results loader

def load_ocr_results(json_path: str) -> dict:
    """Returns dict keyed by image name → OCR result for that image."""
    if not os.path.exists(json_path):
        print(f"[WARN] OCR results not found: {json_path}")
        return {}
    with open(json_path) as f:
        results = json.load(f)
    return {r["image"]: r for r in results}


def flatten_ocr_text(ocr_result: dict) -> str:
    """Flatten all detections' text into one string for whole_image_ocr mode."""
    if not ocr_result:
        return ""
    texts = [d["text"] for d in ocr_result.get("detections", []) if d.get("text")]
    return "\n".join(texts)


def format_cropped_text(ocr_result: dict) -> str:
    """Format per-patch text with bbox ids for cropped_ocr mode."""
    if not ocr_result:
        return ""
    lines = []
    for d in ocr_result.get("detections", []):
        lines.append(f"[patch {d['id']}] {d.get('text', '')}")
    return "\n".join(lines)


# process one image, one mode

def run_mode(image_name: str, mode: str, vlm, schema: list,
            cfg: dict, ocr_by_image: dict) -> dict:

    ocr_result = ocr_by_image.get(image_name, {})

    if mode == "whole_image":
        img_path = find_image_path(cfg["io"]["images_dir"], image_name)
        images = [Image.open(img_path).convert("RGB")]
        text_context = ""

    elif mode == "whole_image_ocr":
        img_path = find_image_path(cfg["io"]["images_dir"], image_name)
        images = [Image.open(img_path).convert("RGB")]
        text_context = flatten_ocr_text(ocr_result)

    elif mode == "cropped_ocr":
        crops_dir = os.path.join(cfg["io"]["crops_dir"], image_name)
        if not os.path.isdir(crops_dir):
            return {"error": f"No crops found for {image_name}"}
        crop_files = sorted(os.listdir(crops_dir))
        images = [Image.open(os.path.join(crops_dir, f)).convert("RGB") for f in crop_files]
        text_context = format_cropped_text(ocr_result)

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    prompt = build_prompt(schema, mode, text_context)
    extracted = vlm.extract(images, text_context, prompt)

    return extracted


def find_image_path(images_dir: str, image_name: str) -> str:
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(images_dir, image_name + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No image found for {image_name} in {images_dir}")


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/vlm.yaml")
    parser.add_argument("--model",      default=None, help="Override VLM: claude | ...")
    parser.add_argument("--mode",       default=None, help="Run a single mode only")
    parser.add_argument("--image_name", default=None, help="Run a single image (stem, no ext)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    schema = load_schema(cfg["schema_path"])
    vlm    = load_vlm(cfg, args.model)
    modes  = [args.mode] if args.mode else cfg["modes"]

    ocr_by_image = load_ocr_results(cfg["io"]["ocr_results_json"])

    if args.image_name:
        image_names = [args.image_name]
    else:
        image_names = list(ocr_by_image.keys())
        if not image_names:
            image_names = [Path(f).stem for f in os.listdir(cfg["io"]["images_dir"])
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    output_dir = cfg["io"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  VLM       : {cfg['vlm']['model']}")
    print(f"  MODES     : {modes}")
    print(f"  IMAGES    : {len(image_names)}")
    print(f"{'='*55}\n")

    all_results = []

    for image_name in image_names:
        print(f"\n--- {image_name} ---")
        for mode in modes:
            print(f"  [{mode}] extracting...")
            extracted = run_mode(image_name, mode, vlm, schema, cfg, ocr_by_image)

            result = {
                "image":     image_name,
                "mode":      mode,
                "vlm_model": cfg["vlm"]["model"],
                "extracted": extracted,
            }
            all_results.append(result)

            if cfg["io"]["save_json"]:
                out_path = os.path.join(output_dir, f"{image_name}_{mode}.json")
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

            print(f"    → {json.dumps(extracted, indent=2)[:200]}...")

    combined_path = os.path.join(output_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Done. Combined results → {combined_path}")


if __name__ == "__main__":
    main()