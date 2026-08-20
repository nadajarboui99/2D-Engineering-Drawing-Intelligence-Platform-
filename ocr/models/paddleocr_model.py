"""
PaddleOCR wrapper — full detection + recognition pipeline (PP-OCRv5 / paddleocr 3.x).
Install: pip install paddlepaddle paddleocr

Good on dense documents; reads a whole region and returns all recognized text.

Template: copy this file to add another OCR engine — only __init__ and read()
need changing to match the engine's API.
"""
import numpy as np
from PIL import Image
from models.base import BaseOCR


class PaddleOCRModel(BaseOCR):
    def __init__(self, lang: str = "fr"):   # drawings are French; textline-orientation for rotated text
        # Lazy import so the backend never touches paddle unless this model runs.
        from paddleocr import PaddleOCR
        print(f"[PaddleOCR] Loading PP-OCRv5 (lang={lang}) …")
        # We feed already-cropped regions, so skip doc-level orientation/unwarping
        # (those are for full pages and can choke on tiny crops).
        self.ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )

    def read(self, image: Image.Image) -> str:
        arr = np.array(image.convert("RGB"))
        results = self.ocr.predict(arr)          # 3.x: list of OCRResult (dict-like)
        texts = []
        for res in (results or []):
            rec = None
            try:
                rec = res["rec_texts"]           # OCRResult supports __getitem__
            except Exception:
                rec = getattr(res, "get", lambda *_: None)("rec_texts")
            if rec:
                texts.extend([t for t in rec if t])
        return " ".join(texts).strip()
