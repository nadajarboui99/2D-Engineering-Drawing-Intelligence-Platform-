#!/usr/bin/env python3
"""
Bootstrap the MASTER annotation set from the existing detection ground truth.

For each task it reads the COCO val.json (boxes + class already annotated for
detection) and writes one master record per image to:

    dataset/master/<task>/<image_stem>.json

with the detection boxes pre-filled. You then only add the NEW layers by hand:
  - `text`   on each region      (for OCR evaluation)
  - `features`  for the image     (for VLM evaluation)

Existing master files are left untouched unless you pass --force (so you never
lose annotation work). Run `build_from_master.py` afterwards to generate the
OCR / VLM ground-truth files the app reads.

Usage:
    python dataset/bootstrap_master.py                # both tasks
    python dataset/bootstrap_master.py --task tables
    python dataset/bootstrap_master.py --force        # overwrite existing masters
"""
import os
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COCO = {
    "tables":     os.path.join(ROOT, "Table_dimensions_detection", "data", "tables", "valid", "val.json"),
    "dimensions": os.path.join(ROOT, "Table_dimensions_detection", "data", "dimensions", "val", "val.json"),
}
SCHEMA_PATH = os.path.join(ROOT, "vlm", "configs", "feature_schema.yaml")

# Fallback if PyYAML isn't importable in this interpreter.
_DEFAULT_FEATURES = ["length", "width", "bends_count", "holes_count",
                     "hole_diameter", "cut_perimeter", "material", "standard"]


def _feature_names():
    try:
        import yaml
        with open(SCHEMA_PATH) as f:
            data = yaml.safe_load(f)
        return [feat["name"] for feat in data.get("features", [])] or _DEFAULT_FEATURES
    except Exception:
        return _DEFAULT_FEATURES


def _stem(file_name: str) -> str:
    return os.path.splitext(os.path.basename(file_name))[0]


def bootstrap_task(task: str, force: bool):
    coco_path = COCO[task]
    if not os.path.exists(coco_path):
        print(f"[skip] {task}: no COCO file at {coco_path}")
        return

    with open(coco_path) as f:
        coco = json.load(f)

    cat_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    anns_by_image = {}
    for a in coco.get("annotations", []):
        anns_by_image.setdefault(a["image_id"], []).append(a)

    out_dir = os.path.join(ROOT, "dataset", "master", task)
    os.makedirs(out_dir, exist_ok=True)
    feature_names = _feature_names()

    created, skipped = 0, 0
    for img in coco.get("images", []):
        stem = _stem(img["file_name"])
        out_path = os.path.join(out_dir, f"{stem}.json")
        if os.path.exists(out_path) and not force:
            skipped += 1
            continue

        regions = []
        for i, a in enumerate(sorted(anns_by_image.get(img["id"], []), key=lambda x: x.get("id", 0))):
            region = {
                "id":    i,
                "class": cat_name.get(a["category_id"], str(a["category_id"])),
                "bbox":  [round(v, 2) for v in a["bbox"]],   # COCO [x, y, w, h]
                "text":  "",                                  # ← FILL: OCR ground truth
            }
            if task == "dimensions":
                region["dim_type"] = ""                       # ← optional: linear / radial / angular ...
            regions.append(region)

        record = {
            "image":    stem,
            "task":     task,
            "width":    img.get("width"),
            "height":   img.get("height"),
            "regions":  regions,
            # ← FILL: VLM ground truth. value = correct value, text/bbox optional.
            "features": {name: {"value": None, "text": "", "bbox": None} for name in feature_names},
        }
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        created += 1

    print(f"[{task}] master records: {created} created, {skipped} kept → dataset/master/{task}/")


def main():
    ap = argparse.ArgumentParser(description="Bootstrap master annotations from COCO detection GT")
    ap.add_argument("--task", choices=["tables", "dimensions"], default=None, help="default: both")
    ap.add_argument("--force", action="store_true", help="overwrite existing master files")
    args = ap.parse_args()

    tasks = [args.task] if args.task else ["tables", "dimensions"]
    for t in tasks:
        bootstrap_task(t, args.force)


if __name__ == "__main__":
    main()
