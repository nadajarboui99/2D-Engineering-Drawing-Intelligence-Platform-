"""
docTR wrapper (Mindee, python-doctr) — a classic two-stage dedicated OCR:
a text-detection model (DBNet/CRAFT-style) + a CRNN recognizer. Torch-native,
runs locally, transcription only — NOT a VLM. Multi-line capable, so it is a
legitimate candidate for both single-line callouts and multi-line table blocks.

Implements the shared OCR interface: read(image) -> str.
Install: pip install python-doctr[torch]   (weights download on first use)
"""
import numpy as np
from PIL import Image
from models.base import BaseOCR


class DocTRModel(BaseOCR):
    def __init__(self):
        # docTR downloads weights via urllib, which on macOS python.org builds
        # can't find the system CA bundle → point it at certifi's.
        import os
        try:
            import certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except Exception:
            pass
        # Lazy import so the backend never touches doctr unless this model runs.
        from doctr.models import ocr_predictor
        print("[docTR] Loading ocr_predictor (pretrained) …")
        self.model = ocr_predictor(pretrained=True)

    def read(self, image: Image.Image) -> str:
        arr = np.array(image.convert("RGB"))
        result = self.model([arr])              # list of pages (one here)
        words = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for w in line.words:
                        if w.value:
                            words.append(w.value)
        return " ".join(words).strip()
