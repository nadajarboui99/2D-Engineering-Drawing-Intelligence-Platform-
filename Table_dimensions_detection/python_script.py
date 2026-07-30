import json

def fix_categories(json_path, output_path):
    with open(json_path) as f:
        coco = json.load(f)

    # Fix category names: "Table" → "table"
    for cat in coco["categories"]:
        cat["name"] = cat["name"].lower()

    seen = {}
    id_remap = {}
    new_categories = []

    for cat in coco["categories"]:
        name = cat["name"]
        if name not in seen:
            seen[name] = cat["id"]
            new_categories.append(cat)
        # if duplicate, map old id → existing id
        id_remap[cat["id"]] = seen[name]

    for ann in coco["annotations"]:
        ann["category_id"] = id_remap[ann["category_id"]]

    coco["categories"] = new_categories

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Done. Categories now: {[c['name'] for c in new_categories]}")
    print(f"Saved to {output_path}")

fix_categories("data/tables/train/train.json", "data/tables/train/train.json")
fix_categories("data/tables/valid/val.json",   "data/tables/valid/val.json")

def fix_dimensions(json_path, output_path):
    with open(json_path) as f:
        coco = json.load(f)

    # Keep only the real class, remove "New-Drawings"
    real_categories = [c for c in coco["categories"] if c["name"] != "New-Drawings"]
    
    fake_ids = {c["id"] for c in coco["categories"] if c["name"] == "New-Drawings"}
    
    coco["annotations"] = [
        ann for ann in coco["annotations"] 
        if ann["category_id"] not in fake_ids
    ]
    
    coco["categories"] = real_categories

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Done. Categories now: {[c['name'] for c in real_categories]}")
    print(f"Remaining annotations: {len(coco['annotations'])}")

fix_dimensions("data/dimensions/train/train.json", 
               "data/dimensions/train/train.json")
fix_dimensions("data/dimensions/val/val.json",   
               "data/dimensions/val/val.json")