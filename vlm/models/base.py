"""
Base VLM interface — all VLM models must implement this.
To add a new model: create a new file in vlm/models/ and inherit from BaseVLM.
"""

from abc import ABC, abstractmethod
from PIL import Image


class BaseVLM(ABC):

    @abstractmethod
    def extract(self, images: list, text_context: str, prompt: str) -> dict:
        """
        Args:
            images:       list of PIL images (1 image for whole-image modes,
                          multiple for crop mode)
            text_context: OCR text to include in the prompt (can be empty string)
            prompt:       the full instruction prompt (built from schema)

        Returns:
            dict — parsed JSON with extracted features (or {"error": ...} on failure)
        """
        pass