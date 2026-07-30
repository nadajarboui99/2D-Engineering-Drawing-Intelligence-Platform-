"""
Image enhancement for OCR crops.
Applied before passing a crop to the OCR model.
"""
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np


def enhance(image: Image.Image, level: str = "basic") -> Image.Image:
    """
    level:
        "none"  → return as-is
        "basic" → upscale 2x + sharpen
        "full"  → upscale 4x + denoise + contrast + binarize
    """
    if level == "none":
        return image

    if level == "basic":
        w, h = image.size
        image = image.resize((w * 2, h * 2), Image.LANCZOS)
        image = image.filter(ImageFilter.SHARPEN)
        return image

    if level == "full":
        w, h = image.size
        image = image.resize((w * 4, h * 4), Image.LANCZOS)
        image = image.convert("L")
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = ImageEnhance.Contrast(image).enhance(2.0)
        image = image.filter(ImageFilter.SHARPEN)
        arr       = np.array(image)
        threshold = arr.mean()
        arr       = (arr > threshold).astype(np.uint8) * 255
        image     = Image.fromarray(arr).convert("RGB")
        return image

    return image