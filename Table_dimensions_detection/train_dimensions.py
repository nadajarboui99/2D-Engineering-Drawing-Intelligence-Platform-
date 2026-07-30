"""
Dimension Detection — Train & Evaluate
=======================================

Usage:
    # Train with YOLOv11 (default config)
    python train_dimensions.py --config configs/dimensions.yaml

    # Train with RT-DETR (change model.name in config or override here)
    python train_dimensions.py --config configs/dimensions.yaml --model rtdetr

    # Evaluate only (skip training)
    python train_dimensions.py --config configs/dimensions.yaml --eval_only --weights runs/dimensions/yolov11_n/weights/best.pt
"""

import os
import sys
import json
import argparse
import yaml
import torch

# make project root importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import build_dataloader
from utils.metrics import DetectionMetrics


# model factory
def load_model(cfg: dict, weights_override: str = None):
    """
    Instantiate a model from config.
    To add a new model: add a new elif block here. That's it.
    """
    model_name  = cfg["model"]["name"].lower()
    model_size  = cfg["model"].get("size", "n")
    num_classes = cfg["dataset"]["num_classes"]
    device      = cfg["training"].get("device", None)
    weights     = weights_override or cfg["model"].get("weights", None)

    if model_name == "yolov11":
        from models.yolov11 import YOLOv11Detector
        return YOLOv11Detector(model_size=model_size, num_classes=num_classes,
                               device=device, weights=weights)

    elif model_name == "rtdetr":
        from models.rtdetr import RTDETRDetector
        return RTDETRDetector(model_size=model_size, num_classes=num_classes,
                              device=device, weights=weights)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Add it to the load_model() factory in this script."
        )


# train
def train(cfg: dict, model):
    print(f"\n{'='*55}")
    print(f"  TASK     : {cfg['task'].upper()}")
    print(f"  MODEL    : {cfg['model']['name']} (size={cfg['model']['size']})")
    print(f"  EPOCHS   : {cfg['training']['epochs']}")
    print(f"  BATCH    : {cfg['training']['batch_size']}")
    print(f"{'='*55}\n")

    data_yaml = cfg["dataset"]["yolo_data_yaml"]
    if not os.path.exists(data_yaml):
        print(f"[ERROR] data.yaml not found at {data_yaml}")
        print("Run utils/convert_coco_to_yolo.py first to generate YOLO labels.")
        sys.exit(1)

    model.train(
        data_yaml  = data_yaml,
        epochs     = cfg["training"]["epochs"],
        batch      = cfg["training"]["batch_size"],
        imgsz      = cfg["training"]["imgsz"],
        lr         = cfg["training"]["lr"],
        project    = cfg["training"]["project"],
        name       = cfg["training"]["run_name"],
    )

    best_weights = os.path.join(
        cfg["training"]["project"],
        cfg["training"]["run_name"],
        "weights", "best.pt"
    )
    print(f"\n[TRAIN] Best weights saved to: {best_weights}")
    return best_weights


# evaluate
def evaluate(cfg: dict, model):
    print(f"\n[EVAL] Running evaluation on validation set...")

    val_loader, category_names = build_dataloader(
        images_dir       = cfg["dataset"]["val_images"],
        annotations_file = cfg["dataset"]["val_annotations"],
        batch_size       = 1,
        train            = False,
    )

    iou_thresh  = cfg["evaluation"]["iou_threshold"]
    conf_thresh = cfg["evaluation"]["conf_threshold"]
    metrics     = DetectionMetrics(iou_threshold=iou_thresh)

    for images, targets in val_loader:
        preds = model.predict(list(images), conf_threshold=conf_thresh,
                            imgsz=cfg["training"]["imgsz"])

        # Remap COCO category ids to 0-indexed to match YOLO output
        remapped_targets = []
        for t in list(targets):
            remapped = {
                "boxes":  t["boxes"],
                "labels": t["labels"] - 1,
            }
            remapped_targets.append(remapped)

        metrics.update(preds, remapped_targets)

    results = metrics.compute()
    metrics.print_report(results, category_names)

    if cfg["evaluation"].get("save_results", False):
        out_dir = cfg["evaluation"]["results_dir"]
        os.makedirs(out_dir, exist_ok=True)
        run_name = cfg["training"]["run_name"]
        out_path = os.path.join(out_dir, f"{run_name}_results.json")
        with open(out_path, "w") as f:
            json.dump({"model": cfg["model"]["name"],
                       "run":   run_name,
                       "task":  cfg["task"],
                       **results}, f, indent=2)
        print(f"[EVAL] Results saved to {out_path}")

    return results


# main
def main():
    parser = argparse.ArgumentParser(description="Dimension Detection — Train & Eval")
    parser.add_argument("--config",    default="configs/dimensions.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--model",     default=None,
                        help="Override model name (yolov11 | rtdetr)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, run evaluation only")
    parser.add_argument("--weights",   default=None,
                        help="Path to weights for eval_only mode")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["name"] = args.model
        cfg["training"]["run_name"] = f"{args.model}_{cfg['model']['size']}"

    model = load_model(cfg, weights_override=args.weights)

    if args.eval_only:
        if not args.weights:
            print("[ERROR] --eval_only requires --weights <path_to_.pt>")
            sys.exit(1)
        evaluate(cfg, model)
    else:
        best_weights = train(cfg, model)
        model = load_model(cfg, weights_override=best_weights)
        evaluate(cfg, model)


if __name__ == "__main__":
    main()