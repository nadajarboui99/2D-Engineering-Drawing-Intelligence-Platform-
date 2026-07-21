#!/usr/bin/env python3
"""
Scaffold UNIFIED master files for the images you selected — you annotate them by hand.

Images can come from ANY source (online, val.json, scans...). Put them in
`dataset/selected_images/` and run this. For each image it writes/updates
`dataset/master/unified/<image-stem>.json`:

  - if the image happens to match the detection val.json, its dimension boxes are
    PRE-FILLED as a bonus (text still blank);
  - otherwise you get a BLANK master (no regions) to annotate fully by hand.

Either way you then add/fix boxes, type region text, and fill features. See
`dataset/master/_TEMPLATE.json` for the exact structure, or run --help-table for
a table-region snippet.

Safety: existing annotation is never overwritten — re-running only tops up
missing dimension boxes and preserves your text, table regions, and features.

Usage:
  python dataset/build_master.py                 # scaffold for every image in selected_images/
  python dataset/build_master.py --ids 24088841_A 26218_D   # also pull these from val.json + copy them in
  python dataset/build_master.py --help-table
"""
import os
import sys
import json
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTED_DIR   = os.path.join(ROOT, "dataset", "selected_images")
UNIFIED_DIR    = os.path.join(ROOT, "dataset", "master", "unified")
LEGACY_DIM_DIR = os.path.join(ROOT, "dataset", "master", "dimensions")
SCHEMA_PATH    = os.path.join(ROOT, "vlm", "configs", "feature_schema.yaml")

# Optional: detection val.json — only used to PRE-FILL dimension boxes when an image matches.
DIM_COCO   = os.path.join(ROOT, "Table_dimensions_detection", "data", "dimensions", "val", "val.json")
DIM_IMAGES = os.path.join(ROOT, "Table_dimensions_detection", "data", "dimensions", "val", "images")

IMG_EXT = (".jpg", ".jpeg", ".png")
_DEFAULT_FEATURES = ["length", "width", "bends_count", "holes_count",
                     "hole_diameter", "cut_perimeter", "material", "standard"]

TABLE_REGION_TEMPLATE = {"id": None, "class": "table", "bbox": [0, 0, 0, 0], "text": "", "cells": []}


def _feature_names():
    try:
        import yaml
        with open(SCHEMA_PATH) as f:
            return [x["name"] for x in yaml.safe_load(f).get("features", [])] or _DEFAULT_FEATURES
    except Exception:
        return _DEFAULT_FEATURES


def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def _image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return None, None


def _load_coco():
    """Returns (cat_map, anns_by_image, images_by_id) or empty structures if no val.json."""
    if not os.path.exists(DIM_COCO):
        return {}, {}, {}
    coco = json.load(open(DIM_COCO))
    cat = {c["id"]: c["name"] for c in coco.get("categories", [])}
    anns = {}
    for a in coco.get("annotations", []):
        anns.setdefault(a["image_id"], []).append(a)
    images = {img["id"]: img for img in coco.get("images", [])}
    return cat, anns, images


def _match(req_id, images):
    """Match an id to exactly one COCO image; returns image dict, None (no match), or '' (ambiguous)."""
    req = _stem(req_id)
    exact, prefix = [], []
    for img in images.values():
        fstem = _stem(img["file_name"])
        ename = _stem(img.get("extra", {}).get("name", "")) if img.get("extra") else ""
        if req == fstem or (ename and req == ename):
            exact.append(img)
        elif fstem.startswith(req) or (ename and ename.startswith(req)):
            prefix.append(img)
    if exact:
        return exact[0] if len(exact) == 1 else ""
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        return ""
    return None


def _prior(stem):
    """Prior annotation to preserve: {bbox_key: text}, [table regions], features, meta."""
    texts, tables, features, meta = {}, [], None, None
    for path in (os.path.join(UNIFIED_DIR, f"{stem}.json"),
                 os.path.join(LEGACY_DIM_DIR, f"{stem}.json")):
        if not os.path.exists(path):
            continue
        rec = json.load(open(path))
        for r in rec.get("regions", []):
            key = tuple(round(v, 1) for v in r.get("bbox", []))
            if (r.get("text") or "").strip() and key not in texts:
                texts[key] = r["text"]
            if r.get("class") == "table" and path.startswith(UNIFIED_DIR):
                tables.append(r)
        if path.startswith(UNIFIED_DIR):
            features = rec.get("features")
            meta = rec.get("meta")
    return texts, tables, features, meta


def build(args):
    feature_names = _feature_names()
    cat, anns, images = _load_coco()

    # Collect selected images: everything pasted in selected_images/, plus any --ids from val.json.
    selected = {}   # stem -> {"coco": img|None, "src": path|None, "size": (w,h)}
    if os.path.isdir(SELECTED_DIR):
        for f in sorted(os.listdir(SELECTED_DIR)):
            if f.lower().endswith(IMG_EXT):
                stem = _stem(f)
                m = _match(stem, images)
                selected[stem] = {"coco": m if m not in ("", None) else None,
                                  "src": os.path.join(SELECTED_DIR, f), "size": None}

    missing, ambiguous = [], []
    for rid in (args.ids or []):
        rid = rid.strip().rstrip(",")
        if not rid:
            continue
        m = _match(rid, images)
        if m is None:
            missing.append(rid)
        elif m == "":
            ambiguous.append(rid)
        else:
            selected.setdefault(_stem(m["file_name"]), {"coco": m, "src": None, "size": None})["coco"] = m

    if missing or ambiguous:
        if missing:
            print("[stop] --ids not found in the dimension val.json:")
            [print(f"    - {x}") for x in missing]
        if ambiguous:
            print("[stop] --ids matched more than one image (be more specific):")
            [print(f"    - {x}") for x in ambiguous]
        sys.exit(1)

    if not selected:
        sys.exit("[stop] No images selected. Paste image files into "
                 f"{SELECTED_DIR}/ (or pass --ids for val.json images), then re-run.")

    os.makedirs(UNIFIED_DIR, exist_ok=True)
    created = updated = prefilled = blank = 0

    for stem, info in sorted(selected.items()):
        img = info["coco"]

        # Make sure the image lives in selected_images/ so every stage runs on it.
        if info["src"] is None and img is not None:
            source = os.path.join(DIM_IMAGES, img["file_name"])
            if os.path.exists(source):
                os.makedirs(SELECTED_DIR, exist_ok=True)
                info["src"] = os.path.join(SELECTED_DIR, img["file_name"])
                if not os.path.exists(info["src"]):
                    shutil.copy2(source, info["src"])

        prior_texts, prior_tables, prior_feats, prior_meta = _prior(stem)

        regions = []
        if img is not None:  # bonus: pre-fill dimension boxes from val.json
            for i, a in enumerate(sorted(anns.get(img["id"], []), key=lambda x: x.get("id", 0))):
                bbox = [round(v, 2) for v in a["bbox"]]
                regions.append({"id": i, "class": cat.get(a["category_id"], "dimension"),
                                "bbox": bbox, "text": prior_texts.get(tuple(round(v, 1) for v in bbox), "")})
            prefilled += 1
        else:
            blank += 1

        for j, t in enumerate(prior_tables):     # keep table regions you already added
            t = dict(t); t["id"] = len(regions) + j
            regions.append(t)

        if img is not None:
            w, h = img.get("width"), img.get("height")
        else:
            w, h = _image_size(info["src"]) if info["src"] else (None, None)

        record = {
            "image": stem, "width": w, "height": h,
            "meta": prior_meta or {"standard": "unknown", "source_type": "unknown",
                                   "clutter": "med", "has_gdt": False, "difficulty": "med"},
            "regions": regions,
            "features": prior_feats or {n: {"value": None, "text": "", "bbox": None} for n in feature_names},
        }
        out = os.path.join(UNIFIED_DIR, f"{stem}.json")
        existed = os.path.exists(out)
        with open(out, "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        updated += 1 if existed else 0
        created += 0 if existed else 1

    print(f"[unified] {len(selected)} images ({created} created, {updated} updated): "
          f"{prefilled} pre-filled from val.json, {blank} blank (annotate boxes by hand).")
    print("Edit dataset/master/unified/*.json (see dataset/master/_TEMPLATE.json), then run "
          "`python dataset/build_from_master.py`.")


def main():
    ap = argparse.ArgumentParser(description="Scaffold unified master files for selected images")
    ap.add_argument("--ids", nargs="*", help="val.json image ids/filenames to also pull in")
    ap.add_argument("--help-table", action="store_true", help="print a table-region template and exit")
    args = ap.parse_args()
    if args.help_table:
        print("Add this to a unified file's \"regions\" list for each table you see:\n")
        print(json.dumps(TABLE_REGION_TEMPLATE, indent=2))
        print('\n"bbox" is [x, y, w, h]; "cells" stays [] for now (flat-text scoring).')
        return
    build(args)


if __name__ == "__main__":
    main()
