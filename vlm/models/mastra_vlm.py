"""
Mastra-backed VLM wrapper.

Instead of calling a model SDK from Python (litellm), this posts to a local
Mastra (Node) service that owns the provider + API keys and returns the model's
text plus token usage. Same `extract()` contract as the old LiteLLMWrapper, so
the rest of the Python pipeline is unchanged.

The Mastra service is model-agnostic: whatever `model` string you pass is sent
straight through (e.g. "claude", "claude-opus-4-5", or another provider's id
once its @ai-sdk/* package is installed in the service).
"""

import os
import io
import json
import time
import base64
import urllib.request
from PIL import Image
from models.base import BaseVLM

MASTRA_URL = os.environ.get("MASTRA_URL", "http://127.0.0.1:8787")
MAX_EDGE   = 1568   # Anthropic (and most VLMs) resize above this anyway; keeps tokens/cost sane.

# rough price table in USD per 1M tokens, input then output, matched by prefix
# used only for the dashboard cost estimate, not billing-accurate
_PRICES = [
    ("claude-opus",    (15.0, 75.0)),
    ("claude-sonnet",  (3.0,  15.0)),
    ("claude-haiku",   (1.0,   5.0)),
    ("claude-fable",   (3.0,  15.0)),
    ("gpt-4o",         (2.5,  10.0)),
    ("gemini",         (1.25,  5.0)),
]


def _price_for(model: str):
    m = (model or "").lower()
    for prefix, price in _PRICES:
        if prefix in m:
            return price
    return None


class MastraVLMWrapper(BaseVLM):

    def __init__(self, model: str = "claude", max_tokens: int = 1500, service_url: str = None):
        self.model      = model
        self.max_tokens = max_tokens
        self.url        = (service_url or MASTRA_URL).rstrip("/")
        # resource usage of the most recent extract() call
        self.last_meta  = {}

    def _image_to_dataurl(self, image: Image.Image) -> str:
        img = image.convert("RGB")
        w, h = img.size
        scale = min(1.0, MAX_EDGE / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    def _cost(self, model, in_tok, out_tok):
        price = _price_for(model)
        if not price or in_tok is None or out_tok is None:
            return None
        cost = (in_tok / 1_000_000) * price[0] + (out_tok / 1_000_000) * price[1]
        return round(cost, 6)

    def extract(self, images: list, text_context: str, prompt: str) -> dict:
        payload = {
            "model":     self.model,
            "prompt":    prompt,
            "maxTokens": self.max_tokens,
            "images":    [self._image_to_dataurl(im) for im in images],
        }
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            self.url + "/extract", data=data,
            headers={"Content-Type": "application/json"},
        )

        self.last_meta = {"latency_s": None, "input_tokens": None,
                          "output_tokens": None, "total_tokens": None, "cost_usd": None}
        t0 = time.perf_counter()
        raw = ""
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            latency = time.perf_counter() - t0

            if body.get("error"):
                self.last_meta["latency_s"] = round(latency, 3)
                return {"error": body["error"]}

            usage  = body.get("usage") or {}
            in_tok = usage.get("input_tokens")
            out_tok = usage.get("output_tokens")
            self.last_meta = {
                "latency_s":     round(latency, 3),
                "input_tokens":  in_tok,
                "output_tokens": out_tok,
                "total_tokens":  usage.get("total_tokens"),
                "cost_usd":      self._cost(body.get("model", self.model), in_tok, out_tok),
            }

            raw = (body.get("text") or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # The model added preamble/prose around the JSON — recover the
                # outermost {...} object so a custom prompt can't break parsing.
                s, e = raw.find("{"), raw.rfind("}")
                if s != -1 and e != -1 and e > s:
                    return json.loads(raw[s:e + 1])
                raise

        except json.JSONDecodeError as e:
            return {"error": f"JSON parse failed: {e}", "raw": raw[:300]}
        except Exception as e:
            self.last_meta["latency_s"] = round(time.perf_counter() - t0, 3)
            return {"error": f"Mastra service call failed: {e}"}


LiteLLMWrapper = MastraVLMWrapper
ClaudeVLM      = MastraVLMWrapper
