"""
Dynamically finds the best weights for a given task by scanning runs/
and picking the run with the highest mAP50 from results.csv.
Falls back to the most recently modified best.pt if no results.csv found.
"""

import os
import csv
import glob


DETECTION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "Table_dimensions_detection")
)
RUNS_DIR = os.path.join(DETECTION_DIR, "runs", "detect")


def _task_keyword(task: str) -> str:
    return "table" if task == "tables" else "dim"


def _iter_run_dirs(task: str):
    """
    Yields (run_name, run_dir) for every run belonging to `task`, regardless of
    how deeply it is nested under runs/detect. Ultralytics writes weights to
    ``<project>/<run_name>/weights/best.pt`` and the project path can add extra
    nesting (e.g. runs/detect/runs/tables/yolov11_n/weights/best.pt), so we
    locate every ``best.pt`` and derive the task from its full path.
    """
    if not os.path.isdir(RUNS_DIR):
        return

    keyword = _task_keyword(task)
    for weights_path in glob.glob(os.path.join(RUNS_DIR, "**", "weights", "best.pt"), recursive=True):
        rel = os.path.relpath(weights_path, RUNS_DIR).lower()
        if keyword not in rel:
            continue
        run_dir  = os.path.dirname(os.path.dirname(weights_path))  # strip /weights/best.pt
        run_name = os.path.basename(run_dir)
        yield run_name, run_dir


def _read_metrics(run_dir: str) -> dict:
    """Reads final-epoch metrics from a run's results.csv (all zero if absent)."""
    metrics = {"map50": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "epochs": 0}
    results_csv = os.path.join(run_dir, "results.csv")
    if not os.path.exists(results_csv):
        return metrics
    try:
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = {k.strip(): v for k, v in rows[-1].items()}
            metrics["map50"]     = float(last.get("metrics/mAP50(B)", 0) or 0)
            metrics["precision"] = float(last.get("metrics/precision(B)", 0) or 0)
            metrics["recall"]    = float(last.get("metrics/recall(B)", 0) or 0)
            metrics["epochs"]    = len(rows)
            p, r = metrics["precision"], metrics["recall"]
            metrics["f1"] = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
    except Exception:
        pass
    return metrics


def find_best_weights(task: str) -> str | None:
    """
    Scans all run folders for the given task and returns the path
    to best.pt of the run with the highest mAP50. Falls back to the most
    recently modified best.pt when no run has usable results.csv metrics.
    """
    best_map  = -1.0
    best_path = None
    fallback_path  = None
    fallback_mtime = -1.0

    for run_name, run_dir in _iter_run_dirs(task):
        weights_path = os.path.join(run_dir, "weights", "best.pt")

        mtime = os.path.getmtime(weights_path)
        if mtime > fallback_mtime:
            fallback_mtime = mtime
            fallback_path  = weights_path

        map50 = _read_metrics(run_dir)["map50"]
        if map50 > best_map:
            best_map  = map50
            best_path = weights_path

    # best_map stays <= 0 only when no run had positive metrics — fall back to newest.
    return best_path if best_map > 0 else fallback_path


def list_all_runs(task: str) -> list[dict]:
    """Returns all runs for a task with their metrics, sorted by mAP50 desc."""
    results = []
    for run_name, run_dir in _iter_run_dirs(task):
        weights_path = os.path.join(run_dir, "weights", "best.pt")
        entry = {"run": run_name, "weights": weights_path}
        entry.update(_read_metrics(run_dir))
        results.append(entry)

    return sorted(results, key=lambda x: x["map50"], reverse=True)