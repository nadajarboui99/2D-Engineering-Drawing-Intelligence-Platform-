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
        from transformers import AutoModelForObjectDetection, AutoImageProcessor
        self.torch = torch
        name = weights or model_name
        print(f"[TableTransformer] Loading {name} …")
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForObjectDetection.from_pretrained(name)
        self.model.eval()

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
            out.append({
                "boxes":  post["boxes"].cpu().numpy(),     # xyxy
                "scores": post["scores"].cpu().numpy(),
                "labels": post["labels"].cpu().numpy().astype(int),
            })
        return out
