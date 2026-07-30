"""
Manages VLM API keys — stored in a local .keys.json file (never committed).
"""
import os
import json

KEYS_PATH = os.path.join(os.path.dirname(__file__), "..", ".keys.json")


def _load() -> dict:
    if not os.path.exists(KEYS_PATH):
        return {}
    with open(KEYS_PATH) as f:
        return json.load(f)


def _save(data: dict):
    with open(KEYS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(KEYS_PATH, 0o600)  # owner read/write only


def set_key(provider: str, key: str):
    data = _load()
    data[provider] = key
    _save(data)
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "google":    "GEMINI_API_KEY",
        "mistral":   "MISTRAL_API_KEY",
        "cohere":    "COHERE_API_KEY",
    }
    if provider in env_map:
        os.environ[env_map[provider]] = key


def get_key(provider: str) -> str | None:
    return _load().get(provider)


def list_providers() -> list:
    data = _load()
    return [{"provider": k, "set": bool(v)} for k, v in data.items()]


def load_all_to_env():
    """Call at startup to load all saved keys into environment."""
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "google":    "GEMINI_API_KEY",
        "mistral":   "MISTRAL_API_KEY",
        "cohere":    "COHERE_API_KEY",
    }
    data = _load()
    for provider, key in data.items():
        if provider in env_map and key:
            os.environ[env_map[provider]] = key