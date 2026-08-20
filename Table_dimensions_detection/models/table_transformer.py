"""
Table Transformer detector (Microsoft, via HuggingFace transformers) — a DETR
architecture trained on PubTables-1M for generic table detection.
Install: pip install transformers torch timm

Implements the same interface as YOLOv11Detector / RTDETRDetector:
    predict(images, conf_threshold, imgsz) -> [{"boxes", "scores", "labels"}]

Template: copy this file to plug in any HuggingFace object-detection model — only
__init__ and predict() need adapting.
"""
import numpy as np


class TableTransformerDetector:
    def __init__(self, weights: str = None, model_name: str = "microsoft/table-transformer-detection"):
        # lazy import, heavy deps load only when this model is actually used
        import torch
        from transformers import DetrImageProcessor, TableTransformerForObjectDetection
        self.torch = torch
        name = weights or model_name
        print(f"[TableTransformer] Loading {name} …")
        self.processor = DetrImageProcessor.from_pretrained(name)
        self.model = TableTransformerForObjectDetection.from_pretrained(name)
        self.model.eval()
        # TATR-detection has two classes: {0: 'table', 1: 'table rotated'}. We only
        # want upright tables; keep label 0 so 'table rotated' boxes don't count as
        # false positives. Resolved from config so it survives label-id changes.
        self.keep_labels = {i for i, lab in self.model.config.id2label.items()
                            if str(lab).strip().lower() == "table"} or {0}

    def predict(self, images, conf_threshold: float = 0.25, imgsz: int = 640):
        from PIL import Image
        out = []
        for img in images:
            pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img.convert("RGB")
            inputs = self.processor(images=pil, return_tensors="pt")
            with self.torch.no_grad():
                res = self.model(**inputs)
            target = self.torch.tensor([[pil.height, pil.width]])
            post = self.processor.post_process_object_detection(
                res, threshold=conf_threshold, target_sizes=target)[0]
            labels = post["labels"].cpu().numpy().astype(int)
            keep = np.array([l in self.keep_labels for l in labels], dtype=bool)
            out.append({
                "boxes":  post["boxes"].cpu().numpy()[keep],     # xyxy, 'table' only
                "scores": post["scores"].cpu().numpy()[keep],
                "labels": labels[keep],
            })
        return out
