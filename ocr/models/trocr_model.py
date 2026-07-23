"""
TrOCR wrapper (Microsoft, via HuggingFace transformers).
Install: pip install transformers torch pillow

TrOCR is a single-line recognizer — best on cropped text regions. On a whole
page it will only transcribe one line, so prefer it in crop mode.

Template: copy this file to add another HuggingFace OCR model — only __init__
and read() need changing.
"""
from PIL import Image
from models.base import BaseOCR


class TrOCRModel(BaseOCR):
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        # Lazy import: heavy deps only load when this model is actually used.
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        print(f"[TrOCR] Loading {model_name} …")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def read(self, image: Image.Image) -> str:
        img = image.convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
        ids = self.model.generate(pixel_values, max_new_tokens=64)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return (text or "").strip()
