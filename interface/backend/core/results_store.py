"""
Unified results store — every run from every module (detection, OCR, VLM)
gets logged here so the Results page can compare everything in one place.
"""
import os
import json
from datetime import datetime

STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "results_store.json")


def _load() -> list:
    if not os.path.exists(STORE_PATH):
        return []
    with open(STORE_PATH) as f:
        return json.load(f)


def _save(data: list):
    with open(STORE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def log_run(stage: str, task: str, model: str, metrics: dict, extra: dict = None):
    """
    stage:  "detection" | "ocr" | "vlm"
    task:   "tables" | "dimensions"
    model:  model name/run identifier
    metrics: dict of metric_name -> value (stage-specific)
    extra:  any extra metadata (mode, weights path, etc.)
    """
    data = _load()
    entry = {
        "id":        f"{stage}_{task}_{model}_{int(datetime.utcnow().timestamp())}",
        "stage":     stage,
        "task":      task,
        "model":     model,
        "metrics":   metrics,
        "extra":     extra or {},
        "timestamp": datetime.utcnow().isoformat(),
    }
    data.append(entry)
    _save(data)
    return entry


def get_runs(stage: str = None, task: str = None) -> list:
    data = _load()
    if stage:
        data = [r for r in data if r["stage"] == stage]
    if task:
        data = [r for r in data if r["task"] == task]
    return sorted(data, key=lambda r: r["timestamp"], reverse=True)


def get_best(stage: str, task: str, metric_key: str) -> dict | None:
    runs = get_runs(stage, task)
    if not runs:
        return None
    return max(runs, key=lambda r: r["metrics"].get(metric_key, 0))


def delete_run(run_id: str):
    data = _load()
    data = [r for r in data if r["id"] != run_id]
    _save(data)