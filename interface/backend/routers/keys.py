"""
API key management for VLM providers.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from core.api_keys import set_key, list_providers, get_key

router = APIRouter()

PROVIDERS = [
    {"id": "anthropic", "label": "Anthropic (Claude)",  "env": "ANTHROPIC_API_KEY",  "models": ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"]},
    {"id": "openai",    "label": "OpenAI (GPT-4o)",     "env": "OPENAI_API_KEY",     "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]},
    {"id": "google",    "label": "Google (Gemini)",     "env": "GEMINI_API_KEY",     "models": ["gemini/gemini-1.5-pro", "gemini/gemini-1.5-flash"]},
    {"id": "mistral",   "label": "Mistral",             "env": "MISTRAL_API_KEY",    "models": ["mistral/mistral-large-latest"]},
]


class KeyPayload(BaseModel):
    provider: str
    key:      str


@router.get("/providers")
def get_providers():
    saved = {p["provider"]: p["set"] for p in list_providers()}
    return [
        {**p, "configured": saved.get(p["id"], False)}
        for p in PROVIDERS
    ]


@router.post("/set")
def save_key(payload: KeyPayload):
    set_key(payload.provider, payload.key)
    return {"ok": True}


@router.get("/models")
def get_all_models():
    """Returns flat list of all available model strings across all providers."""
    saved = {p["provider"]: p["set"] for p in list_providers()}
    models = []
    for p in PROVIDERS:
        for m in p["models"]:
            models.append({
                "value":     m,
                "label":     m,
                "provider":  p["id"],
                "available": saved.get(p["id"], False),
            })
    return models