#!/usr/bin/env python3
"""
Project the UNIFIED master set into the per-stage ground-truth files the app reads.

Reads   dataset/master/unified/*.json   and writes, per region class:

  detection  ->  dataset/derived/unified_detection.coco.json   (boxes + class, both categories)
  OCR crops  ->  ocr/data/ground_truth/dimensions.json         (text of dimension regions)
                 ocr/data/ground_truth/tables.json             (text of table regions)
  OCR whole  ->  ocr/data/ground_truth/_whole_image.json       (set of ALL region texts per image)
  VLM        ->  vlm/data/ground_truth/dimensions.json
                 vlm/data/ground_truth/tables.json             (feature values; same set of images)
  dims parse ->  dataset/derived/dimensions_parsed.json        (value/tolerance/symbol per dim region)

Only annotated content is exported (empty text / null value skipped). Safe under
partial annotation and idempotent. Run whenever you edit the master files.

Usage:  python dataset/build_from_master.py
"""
import os
import sys
import json
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dataset"))
import normalize  # noqa: E402

UNIFIED_DIR = os.path.join(ROOT, "dataset", "master", "unified")


def _out(*parts):
    p = os.path.join(ROOT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build():
    files = sorted(glob.glob(os.path.join(UNIFIED_DIR, "*.json")))
    if not files:
        for parts in (("ocr", "data", "ground_truth", "dimensions.json"),
                      ("ocr", "data", "ground_truth", "tables.json"),
                      ("ocr", "data", "ground_truth", "_whole_image.json"),
                      ("vlm", "data", "ground_truth", "unified.json")):
            p = os.path.join(ROOT, *parts)
            if os.path.exists(p):
                _write(p, {})
        print(f"[build] no unified master files in {UNIFIED_DIR} — ground truth cleared.")
        return

    ocr_dim, ocr_tbl, ocr_whole = {}, {}, {}
    vlm_gt = {}
    dims_parsed = {}
    coco = {"categories": [{"id": 1, "name": "dimension"}, {"id": 2, "name": "table"}],
            "images": [], "annotations": []}
    cat_id = {"dimension": 1, "table": 2}
    ann_id = 0

    n_dim_txt = n_tbl_txt = n_feat = 0

    for img_i, path in enumerate(files):
        rec = json.load(open(path))
        stem = rec.get("image") or os.path.splitext(os.path.basename(path))[0]
        coco["images"].append({"id": img_i, "file_name": stem,
                               "width": rec.get("width"), "height": rec.get("height")})

        dim_texts, tbl_texts, whole = [], [], []
        parsed_rows = []
        for r in rec.get("regions", []):
            cls = r.get("class", "dimension")
            bbox = r.get("bbox", [0, 0, 0, 0])
            # detection GT: every region with a real box
            if bbox and any(v for v in bbox):
                coco["annotations"].append({"id": ann_id, "image_id": img_i,
                                             "category_id": cat_id.get(cls, 1),
                                             "bbox": bbox, "area": round(bbox[2] * bbox[3], 2),
                                             "iscrowd": 0, "segmentation": []})
                ann_id += 1
            text = (r.get("text") or "").strip()
            if not text:
                continue
            whole.append(text)
            if cls == "dimension":
                dim_texts.append(text)
                parsed = r.get("parsed") or normalize.parse_dimension(text)
                parsed_rows.append({"bbox": bbox, "text": text, "parsed": parsed})
            elif cls == "table":
                tbl_texts.append(text)

        if dim_texts:
            ocr_dim[stem] = dim_texts
            n_dim_txt += len(dim_texts)
        if tbl_texts:
            ocr_tbl[stem] = tbl_texts
            n_tbl_txt += len(tbl_texts)
        if whole:
            ocr_whole[stem] = whole
        if parsed_rows:
            dims_parsed[stem] = parsed_rows

        feats = {}
        for name, spec in (rec.get("features") or {}).items():
            val = spec.get("value") if isinstance(spec, dict) else spec
            feats[name] = None if val == "" else val
        if feats:
            vlm_gt[stem] = feats
            n_feat += sum(1 for v in feats.values() if v is not None)

    _write(_out("ocr", "data", "ground_truth", "dimensions.json"), ocr_dim)
    _write(_out("ocr", "data", "ground_truth", "tables.json"), ocr_tbl)
    _write(_out("ocr", "data", "ground_truth", "_whole_image.json"), ocr_whole)
    _write(_out("vlm", "data", "ground_truth", "unified.json"), vlm_gt)
    _write(_out("dataset", "derived", "unified_detection.coco.json"), coco)
    _write(_out("dataset", "derived", "dimensions_parsed.json"), dims_parsed)

    print(f"[build] {len(files)} unified images projected:")
    print(f"  detection : {len(coco['annotations'])} boxes ({sum(1 for a in coco['annotations'] if a['category_id']==1)} dim, "
          f"{sum(1 for a in coco['annotations'] if a['category_id']==2)} table)")
    print(f"  OCR text  : {n_dim_txt} dimension strings ({len(ocr_dim)} imgs), {n_tbl_txt} table strings ({len(ocr_tbl)} imgs)")
    print(f"  VLM       : {n_feat} feature values across {len(vlm_gt)} imgs")
    print("  -> ocr/data/ground_truth/{dimensions,tables}.json, vlm/data/ground_truth/unified.json, dataset/derived/")


if __name__ == "__main__":
    build()
