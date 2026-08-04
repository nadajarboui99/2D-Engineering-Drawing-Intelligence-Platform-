"""
GOT-OCR 2.0 — modern end-to-end (OCR-free) reader (HuggingFace).
Install: pip install transformers torch

Reads an image region and returns plain text. Implements BaseOCR.read().
"""
from PIL import Image
from models.base import BaseOCR


class GOTOCRModel(BaseOCR):
    def __init__(self, model_name: str = "stepfun-ai/GOT-OCR-2.0-hf"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        print(f"[GOT-OCR2] Loading {model_name} …")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name)
        self.model.eval()

    def read(self, image: Image.Image) -> str:
        img = image.convert("RGB")
        inputs = self.processor(img, return_tensors="pt")
        with self.torch.no_grad():
            gen = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=2048,
                tokenizer=self.processor.tokenizer, stop_strings="<|im_end|>")
        # Decode only the newly generated tokens.
        start = inputs["input_ids"].shape[1]
        text = self.processor.decode(gen[0, start:], skip_special_tokens=True)
        return (text or "").strip()
