"""
VLM-as-OCR — uses a vision-language model (via the local Mastra service) to
transcribe an image's text. No local model weights; requires the Mastra service
running on MASTRA_URL (same one the VLM stage uses).

Implements BaseOCR.read().
"""
import os
import io
import json
import base64
import urllib.request
from PIL import Image
from models.base import BaseOCR

MASTRA_URL = os.environ.get("MASTRA_URL", "http://127.0.0.1:8787")
MAX_EDGE   = 1568   # keep tokens/cost sane; providers resize above this anyway


class VLMOCRModel(BaseOCR):
    def __init__(self, model: str = "claude", service_url: str = None):
        self.model = model
        self.url = (service_url or MASTRA_URL).rstrip("/")

    def read(self, image: Image.Image) -> str:
        img = image.convert("RGB")
        w, h = img.size
        scale = min(1.0, MAX_EDGE / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        body = json.dumps({"model": self.model, "image": data_url}).encode("utf-8")
        req = urllib.request.Request(self.url + "/ocr", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"VLM-OCR: Mastra service call failed ({e}). Is it running on {self.url}?")
        if out.get("error"):
            return ""
        return (out.get("text") or "").strip()
