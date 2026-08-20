"""
Florence-2 — Microsoft's unified vision model (HuggingFace, trust_remote_code).
Zero-shot object detection via the "<OD>" task (or "<DENSE_REGION_CAPTION>").
Install: pip install transformers timm einops torch   (flash_attn optional)

Implements the shared detector interface. Florence's OD doesn't return per-box
scores, so scores default to 1.0. Same model can also OCR / caption — here it is
wired for detection.
"""
import numpy as np


class Florence2Detector:
    def __init__(self, weights: str = None,
                 model_name: str = "microsoft/Florence-2-large",
                 task_prompt: str = "<OD>"):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.torch = torch
        self.task = task_prompt
        print(f"[Florence-2] Loading {model_name} ({task_prompt}) …")
        # flash-attn is optional; the model works without it.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        # Newer transformers dropped forced_bos_token_id from the config default,
        # which Florence-2's bundled code still reads. Patch it on every config
        # object we can reach (top, text_config, and the language sub-model), and
        # on the config CLASS as a last resort.
        cfgs = [self.model.config,
                getattr(self.model.config, "text_config", None),
                getattr(getattr(self.model, "language_model", None), "config", None),
                getattr(self.model, "generation_config", None)]
        for cfg in cfgs:
            if cfg is None:
                continue
            try:
                if getattr(cfg, "forced_bos_token_id", None) is None:
                    cfg.forced_bos_token_id = 2
                type(cfg).forced_bos_token_id = 2   # class-level default fallback
            except Exception:
                pass

    def predict(self, images, conf_threshold: float = 0.25, imgsz: int = 640):
        from PIL import Image
        out = []
        for img in images:
            pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img.convert("RGB")
            inputs = self.processor(text=self.task, images=pil, return_tensors="pt")
            with self.torch.no_grad():
                gen = self.model.generate(input_ids=inputs["input_ids"],
                                          pixel_values=inputs["pixel_values"],
                                          max_new_tokens=1024, num_beams=3, do_sample=False)
            text = self.processor.batch_decode(gen, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(
                text, task=self.task, image_size=(pil.width, pil.height))
            boxes = (parsed.get(self.task) or {}).get("bboxes", [])   # xyxy, absolute
            arr = np.array(boxes, dtype=float).reshape(-1, 4)
            out.append({
                "boxes":  arr,
                "scores": np.ones(len(arr)),
                "labels": np.zeros(len(arr), dtype=int),
            })
        return out
