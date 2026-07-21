#!/usr/bin/env python3
"""
Annotation coverage report over the UNIFIED master set — tells you what's left
to annotate and the instance count N behind each metric.

Usage:  python dataset/coverage.py
"""
import os
import glob
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFIED_DIR = os.path.join(ROOT, "dataset", "master", "unified")


def main():
    files = sorted(glob.glob(os.path.join(UNIFIED_DIR, "*.json")))
    if not files:
        print(f"No unified master files yet in {UNIFIED_DIR}.")
        print("Select images (paste into dataset/selected_images/ or use --ids) then run build_master.py.")
        return

    imgs = len(files)
    imgs_with_dim = imgs_with_tbl = 0
    dim_total = dim_text = tbl_total = tbl_text = 0
    feat_total = feat_filled = imgs_with_feat = 0
    need_table = []   # images that have dimensions but no table region yet

    for path in files:
        rec = json.load(open(path))
        regs = rec.get("regions", [])
        dims = [r for r in regs if r.get("class") == "dimension"]
        tbls = [r for r in regs if r.get("class") == "table"]
        if dims:
            imgs_with_dim += 1
        if tbls:
            imgs_with_tbl += 1
        elif dims:
            need_table.append(rec.get("image", os.path.basename(path)))
        dim_total += len(dims)
        dim_text  += sum(1 for r in dims if (r.get("text") or "").strip())
        tbl_total += len(tbls)
        tbl_text  += sum(1 for r in tbls if (r.get("text") or "").strip())

        feats = rec.get("features") or {}
        filled = sum(1 for s in feats.values()
                     if (s.get("value") if isinstance(s, dict) else s) not in (None, ""))
        feat_total += len(feats)
        feat_filled += filled
        if filled:
            imgs_with_feat += 1

    def pct(a, b):
        return f"{(100*a/b):.0f}%" if b else "—"

    print(f"UNIFIED SET: {imgs} images\n")
    print(f"  images with a dimension region : {imgs_with_dim}/{imgs}")
    print(f"  images with a table region     : {imgs_with_tbl}/{imgs}")
    print()
    print(f"  dimension regions : {dim_total:>4}   text filled: {dim_text:>4} ({pct(dim_text, dim_total)})")
    print(f"  table regions     : {tbl_total:>4}   text filled: {tbl_text:>4} ({pct(tbl_text, tbl_total)})")
    print(f"  VLM features      : {feat_filled}/{feat_total} filled across {imgs_with_feat} images")
    print()
    print("  N per metric (statistical unit = instance):")
    print(f"    detection/OCR dimensions : N = {dim_text}")
    print(f"    detection/OCR tables     : N = {tbl_text}")
    print(f"    VLM (per image)          : N = {imgs_with_feat}")

    if need_table:
        print(f"\n  {len(need_table)} image(s) still need a table region added (if they contain one):")
        for name in need_table:
            print(f"    - {name}")


if __name__ == "__main__":
    main()
