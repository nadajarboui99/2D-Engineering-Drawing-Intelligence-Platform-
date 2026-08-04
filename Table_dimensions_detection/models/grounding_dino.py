"""
Grounding DINO — open-vocabulary, text-prompted detector (HuggingFace).
Zero-shot: no training; you describe what to find with a text prompt.
Install: pip install transformers torch

Used here for DIMENSIONS via the prompt below. Implements the shared detector
interface: predict(images, conf_threshold, imgsz) -> [{"boxes","scores","labels"}].
"""
import numpy as np

DEFAULT_PROMPT = "dimension . measurement . numeric value . tolerance ."


class GroundingDINODetector:
    def __init__(self, weights: str = None,
                 model_name: str = "IDEA-Research/grounding-dino-base",
                 prompt: str = DEFAULT_PROMPT):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.prompt = prompt
        print(f"[GroundingDINO] Loading {model_name} …")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self.model.eval()

    def predict(self, images, conf_threshold: float = 0.25, imgsz: int = 640):
        from PIL import Image
        out = []
        for img in images:
            pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img.convert("RGB")
            inputs = self.processor(images=pil, text=self.prompt, return_tensors="pt")
            with self.torch.no_grad():
                res = self.model(**inputs)
            post = self.processor.post_process_grounded_object_detection(
                res, inputs["input_ids"],
                box_threshold=conf_threshold, text_threshold=0.25,
                target_sizes=[(pil.height, pil.width)])[0]
            boxes = post["boxes"].cpu().numpy()          # xyxy
            out.append({
                "boxes":  boxes,
                "scores": post["scores"].cpu().numpy(),
                "labels": np.zeros(len(boxes), dtype=int),  # single class: dimension
            })
        return out
