from fastapi import APIRouter
from core.results_store import get_runs, get_best, delete_run, load_snapshot

router = APIRouter()

METRIC_KEYS = {
    "detection": "map50",
    "ocr":       "coverage",
    "vlm":       "fill_rate",
}


@router.get("/all")
def all_results(stage: str = None, task: str = None):
    return get_runs(stage, task)


@router.get("/best/{stage}/{task}")
def best_result(stage: str, task: str):
    key = METRIC_KEYS.get(stage, "map50")
    best = get_best(stage, task, key)
    return best or {}


@router.get("/summary")
def summary():
    """Best run per stage, across ALL tasks (tasks are now heterogeneous — some
    runs are 'all'/'both', not just tables/dimensions), plus the tasks present."""
    out = {}
    for stage in ["detection", "ocr", "vlm"]:
        key   = METRIC_KEYS[stage]
        runs  = get_runs(stage)
        best  = get_best(stage, None, key)
        out[stage] = {
            "best":  best,
            "tasks": sorted({r["task"] for r in runs}),
            "count": len(runs),
        }
    return out


@router.get("/snapshot/{run_id}")
def snapshot(run_id: str):
    """Full saved dashboard for one run (metrics + per-image visual payload)."""
    return load_snapshot(run_id)


@router.delete("/{run_id}")
def delete(run_id: str):
    delete_run(run_id)
    return {"ok": True}