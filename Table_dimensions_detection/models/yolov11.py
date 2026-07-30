"""
YOLOv11 wrapper — uniform interface for train / predict.

Install: pip install ultralytics
"""

import os
import torch
from ultralytics import YOLO
import numpy as np

class YOLOv11Detector:
    """
    Thin wrapper around Ultralytics YOLO to expose a consistent
    interface shared with other model wrappers (e.g. RTDETRDetector).
    """

    def __init__(self, model_size: str = "n", num_classes: int = 1,
                 device: str = None, weights: str = None):
        """
        Args:
            model_size: one of 'n', 's', 'm', 'l', 'x'  (nano → xlarge)
            num_classes: number of detection classes in your dataset
            device:      'cuda', 'cpu', or None (auto)
            weights:     path to a .pt checkpoint to resume from
        """
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if weights and os.path.exists(weights):
            print(f"[YOLOv11] Loading weights from {weights}")
            self.model = YOLO(weights)
        else:
            model_name = f"yolo11{model_size}.pt"
            print(f"[YOLOv11] Loading pretrained {model_name}")
            self.model = YOLO(model_name)

    def train(self, data_yaml: str, epochs: int = 50, imgsz: int = 640,
              batch: int = 16, lr: float = 0.01, project: str = "runs",
              name: str = "yolov11", **kwargs):
        """
        Train the model.
        data_yaml: path to a YOLO-format data.yaml
                   (YOLOv11 needs YOLO format — see convert_coco_to_yolo.py)
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr,
            device=self.device,
            project=project,
            name=name,
            **kwargs,
        )
        return results

    def predict(self, images, conf_threshold: float = 0.25, imgsz: int = 640):
        """
        Run inference on a list of PIL images or file paths.

        Returns list of dicts: {boxes [N,4], scores [N], labels [N]}
        compatible with DetectionMetrics.update()
        """
        # convert tensors to numpy arrays, YOLO doesn't accept tensors directly
        images_np = []
        for img in images:
            if hasattr(img, 'numpy'):
                # tensor [C, H, W] to numpy [H, W, C] uint8
                arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                images_np.append(arr)
            else:
                images_np.append(img)

        raw = self.model.predict(
            source=images_np,
            conf=conf_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        preds = []
        for r in raw:
            boxes  = torch.tensor(r.boxes.xyxy.cpu().numpy(),  dtype=torch.float32)
            scores = torch.tensor(r.boxes.conf.cpu().numpy(),  dtype=torch.float32)
            labels = torch.tensor(r.boxes.cls.cpu().numpy().astype(int), dtype=torch.int64)
            preds.append({"boxes": boxes, "scores": scores, "labels": labels})

        return preds

    def save(self, path: str):
        self.model.save(path)
        print(f"[YOLOv11] Model saved to {path}")