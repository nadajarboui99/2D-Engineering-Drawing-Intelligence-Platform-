"""
Convert a COCO JSON annotation file to YOLO txt format.

YOLO expects one .txt file per image:
    <class_id> <x_center> <y_center> <width> <height>   (all normalized 0-1)

Run this once before training:
    python utils/convert_coco_to_yolo.py \
        --coco_json path/to/annotations.json \
        --images_dir path/to/images \
        --output_dir path/to/yolo_labels
"""

import os
import json
import argparse


def convert(coco_json: str, images_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    with open(coco_json) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    # Map original category_id → zero-indexed class id
    cat_ids = sorted(set(cat["id"] for cat in coco["categories"]))
    cat_map = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    names_path = os.path.join(output_dir, "categories.txt")
    cat_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
    with open(names_path, "w") as f:
        for cat_id in cat_ids:
            f.write(f"{cat_map[cat_id]}: {cat_names[cat_id]}\n")
    print(f"Category mapping saved to {names_path}")

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    converted, skipped = 0, 0
    for image_id, img_info in images.items():
        w, h = img_info["width"], img_info["height"]
        anns = anns_by_image.get(image_id, [])

        stem = os.path.splitext(img_info["file_name"])[0]
        label_path = os.path.join(output_dir, stem + ".txt")

        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        lines = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]   # COCO: top-left x,y + width, height
            if bw <= 0 or bh <= 0:
                skipped += 1
                continue
            cls_id = cat_map[ann["category_id"]]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        converted += 1

    print(f"Converted {converted} images. Skipped {skipped} invalid boxes.")
    print(f"Labels written to: {output_dir}")
    print("\nNext step — create a data.yaml like this:")
    print("--------------------------------------------")
    print("path: /absolute/path/to/dataset")
    print("train: images/train")
    print("val:   images/val")
    print(f"nc: {len(cat_ids)}")
    print(f"names: {[cat_names[cid] for cid in cat_ids]}")
    print("--------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json",   required=True)
    parser.add_argument("--images_dir",  required=True)
    parser.add_argument("--output_dir",  required=True)
    args = parser.parse_args()
    convert(args.coco_json, args.images_dir, args.output_dir)