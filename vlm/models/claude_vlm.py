"""
LiteLLM-based VLM wrapper.
Supports any model: Claude, GPT-4o, Gemini, Mistral, Ollama, etc.
Install: pip install litellm

Usage:
    vlm = LiteLLMWrapper(model="claude-sonnet-4-6")
    vlm = LiteLLMWrapper(model="gpt-4o")
    vlm = LiteLLMWrapper(model="gemini/gemini-1.5-pro")
    vlm = LiteLLMWrapper(model="ollama/llava")
"""

import os
import json
import base64
import io
import time
from PIL import Image
from models.base import BaseVLM


class LiteLLMWrapper(BaseVLM):

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1500):
        import litellm
        self.litellm    = litellm
        self.model      = model
        self.max_tokens = max_tokens
        # Resource usage of the MOST RECENT extract() call (tokens, latency, cost).
        # Read this from the caller right after extract() returns.
        self.last_meta  = {}

    def _image_to_base64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _usage_from(self, response, latency_s: float) -> dict:
        """Pull token counts + cost from a litellm response (best-effort)."""
        meta = {"latency_s": round(latency_s, 3),
                "input_tokens": None, "output_tokens": None,
                "total_tokens": None, "cost_usd": None}
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                meta["input_tokens"]  = getattr(usage, "prompt_tokens", None)
                meta["output_tokens"] = getattr(usage, "completion_tokens", None)
                meta["total_tokens"]  = getattr(usage, "total_tokens", None)
        except Exception:
            pass
        try:
            cost = self.litellm.completion_cost(completion_response=response)
            meta["cost_usd"] = round(float(cost), 6) if cost is not None else None
        except Exception:
            pass
        return meta

    def extract(self, images: list, text_context: str, prompt: str) -> dict:
        content = []

        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._image_to_base64(img)}"
                }
            })

        content.append({"type": "text", "text": prompt})

        self.last_meta = {"latency_s": None, "input_tokens": None,
                          "output_tokens": None, "total_tokens": None, "cost_usd": None}
        t0 = time.perf_counter()
        try:
            response = self.litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
            )
            self.last_meta = self._usage_from(response, time.perf_counter() - t0)
            raw_text = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            return json.loads(raw_text)

        except json.JSONDecodeError as e:
            return {"error": f"JSON parse failed: {e}", "raw": raw_text[:300]}
        except Exception as e:
            self.last_meta["latency_s"] = round(time.perf_counter() - t0, 3)
            return {"error": str(e)}


# Keep backward-compatible alias
ClaudeVLM = LiteLLMWrapper