"""
PaddleOCR wrapper — full detection + recognition pipeline.
Install: pip install paddlepaddle paddleocr

Good on dense documents; reads a whole region and returns all recognized text.

Template: copy this file to add another OCR engine — only __init__ and read()
need changing to match the engine's API.
"""
import numpy as np
from PIL import Image
from models.base import BaseOCR


class PaddleOCRModel(BaseOCR):
    def __init__(self, lang: str = "fr"):   # drawings are French; angle-cls for rotated text
        # Lazy import so the backend never touches paddle unless this model runs.
        from paddleocr import PaddleOCR
        print(f"[PaddleOCR] Loading (lang={lang}) …")
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def read(self, image: Image.Image) -> str:
        arr = np.array(image.convert("RGB"))
        result = self.ocr.ocr(arr, cls=True)
        lines = []
        for page in (result or []):
            for det in (page or []):
                # det = [box, (text, confidence)]
                if det and len(det) >= 2 and det[1]:
                    lines.append(det[1][0])
        return " ".join(lines).strip()
