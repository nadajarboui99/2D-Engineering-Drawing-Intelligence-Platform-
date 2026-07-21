"""
Tesseract wrapper.
Install: pip install pytesseract
Binary:  brew install tesseract
"""

from PIL import Image
from models.base import BaseOCR


class TesseractModel(BaseOCR):

    def __init__(self, lang: str = "eng", config: str = "--psm 6"):
        """
        --psm 6 → single block of text (good for crops)
        --psm 7 → single line
        --psm 8 → single word
        """
        import pytesseract
        self.pytesseract = pytesseract
        self.lang = lang
        self.config = config
        print("[Tesseract] Ready.")

    def read(self, image: Image.Image) -> str:
        return self.pytesseract.image_to_string(
            image, lang=self.lang, config=self.config
        ).strip()

    def read_with_confidence(self, image: Image.Image) -> list:
        data = self.pytesseract.image_to_data(
            image, lang=self.lang, config=self.config,
            output_type=self.pytesseract.Output.DICT
        )
        results = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            conf = int(data["conf"][i])
            if word and conf > 0:
                results.append((word, round(conf / 100, 4)))
        return results