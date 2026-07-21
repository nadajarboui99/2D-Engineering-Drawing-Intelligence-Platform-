"""
Manages uploaded/registered model weights for detection.
Each entry: { id, name, task, path, source, added_at }
"""
import os
import json
from datetime import datetime

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "weights_registry.json")
WEIGHTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def _load() -> list:
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _save(data: list):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def register(name: str, task: str, path: str, source: str = "upload") -> dict:
    data  = _load()
    entry = {
        "id":       f"{task}_{name}_{int(datetime.utcnow().timestamp())}",
        "name":     name,
        "task":     task,
        "path":     path,
        "source":   source,
        "added_at": datetime.utcnow().isoformat(),
    }
    # Remove existing entry with same name+task
    data = [e for e in data if not (e["name"] == name and e["task"] == task)]
    data.append(entry)
    _save(data)
    return entry


def list_weights(task: str = None) -> list:
    data = _load()
    if task:
        data = [e for e in data if e["task"] == task]
    return sorted(data, key=lambda e: e["added_at"], reverse=True)


def get_weights_path(name: str, task: str) -> str | None:
    data = _load()
    for e in data:
        if e["name"] == name and e["task"] == task:
            return e["path"] if os.path.exists(e["path"]) else None
    return None


def delete_weights(entry_id: str):
    data = _load()
    data = [e for e in data if e["id"] != entry_id]
    _save(data)