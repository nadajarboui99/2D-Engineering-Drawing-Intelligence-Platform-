"""
Base OCR interface — all OCR models must implement this.
To add a new model: create a new file in ocr/models/ and inherit from BaseOCR.
"""

from abc import ABC, abstractmethod
from PIL import Image


class BaseOCR(ABC):

    @abstractmethod
    def read(self, image: Image.Image) -> str:
        """Takes a PIL image crop, returns extracted text as string."""
        pass

    def batch_read(self, images: list) -> list:
        """Default: loop. Override for true batching."""
        return [self.read(img) for img in images]