"""
EasyOCR wrapper.
Install: pip install easyocr
"""

import numpy as np
from PIL import Image
from models.base import BaseOCR


class EasyOCRModel(BaseOCR):

    def __init__(self, languages: list = ["en"], gpu: bool = False):
        import easyocr
        print("[EasyOCR] Loading model...")
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def read(self, image: Image.Image) -> str:
        img_np = np.array(image)
        results = self.reader.readtext(img_np)
        return " ".join([r[1] for r in results]).strip()

    def read_with_confidence(self, image: Image.Image) -> list:
        img_np = np.array(image)
        results = self.reader.readtext(img_np)
        return [(r[1], round(r[2], 4)) for r in results]