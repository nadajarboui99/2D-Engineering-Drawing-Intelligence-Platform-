"""
Shared COCO dataset loader for both tasks (tables / dimensions).
Works with any single-class or multi-class COCO JSON annotation file.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class COCODetectionDataset(Dataset):
    """
    Loads images and bounding boxes from a COCO-format JSON.

    Expected JSON structure:
        {
          "images":      [{ "id", "file_name", "width", "height" }, ...],
          "annotations": [{ "id", "image_id", "category_id", "bbox": [x,y,w,h], ... }, ...],
          "categories":  [{ "id", "name" }, ...]
        }
    """

    def __init__(self, images_dir: str, annotations_file: str, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        with open(annotations_file) as f:
            coco = json.load(f)

        # Build id → image info map
        self.images = {img["id"]: img for img in coco["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # Group annotations by image_id
        self.annotations = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            self.annotations.setdefault(iid, []).append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]

        # Load image
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Load annotations → convert COCO [x, y, w, h] to [x1, y1, x2, y2]
        anns = self.annotations.get(image_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def get_category_names(self):
        return self.categories


def get_transforms(train: bool):
    """Basic transforms — extend here if you need augmentation."""
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_variable_size(batch):
    return tuple(zip(*batch))


def build_dataloader(images_dir: str, annotations_file: str, batch_size: int,
                     train: bool = True, num_workers: int = 2):
    dataset = COCODetectionDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        transforms=get_transforms(train),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        collate_fn=collate_variable_size,
    )
    return loader, dataset.get_category_names()