"""
Surya OCR wrapper (surya-ocr 3.x / 0.22 API).

Surya is a dedicated, open-source OCR system with built-in layout + line
detection and a multilingual recognizer — strong on multi-line / structured
documents (its headline use case is tables and reading order). It runs locally
and only transcribes text; it is NOT the general-purpose stage-3 VLM.

Implements the shared OCR interface: read(image) -> str.
Install: pip install surya-ocr   (models download from HuggingFace on first use)
"""
import numpy as np
from PIL import Image
from models.base import BaseOCR


class SuryaOCRModel(BaseOCR):
    def __init__(self, langs=None):
        # Lazy import so the backend never touches surya unless this model runs.
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor
        print("[Surya] Loading recognition predictor …")
        self.manager = SuryaInferenceManager()
        self.rec = RecognitionPredictor(self.manager)

    def read(self, image: Image.Image) -> str:
        pil = image.convert("RGB")
        # full_page=True: detect layout + recognize in one pass over the crop.
        pages = self.rec([pil], full_page=True)
        texts = []
        for page in (pages or []):
            for line in getattr(page, "text_lines", []) or []:
                t = getattr(line, "text", None)
                if t:
                    texts.append(t)
        return " ".join(texts).strip()
