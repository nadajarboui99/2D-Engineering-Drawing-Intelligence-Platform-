"""
Which annotated images are part of the EVALUATION set.

An annotation master carries a `status`: "complete" (ready to evaluate) or
"draft" (still being annotated — saved so work isn't lost, but excluded from
evaluation and ground truth). Missing status = complete (back-compat with
images annotated before drafts existed).
"""
import os
import json

ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UNIFIED_DIR = os.path.join(ROOT, "dataset", "master", "unified")


def is_draft(rec: dict) -> bool:
    return (rec or {}).get("status") == "draft"


def draft_stems() -> set:
    """Stems whose annotation is still a draft (exclude from evaluation)."""
    out = set()
    if not os.path.isdir(UNIFIED_DIR):
        return out
    for f in os.listdir(UNIFIED_DIR):
        if not f.endswith(".json") or f.startswith("_"):
            continue
        try:
            rec = json.load(open(os.path.join(UNIFIED_DIR, f)))
            if is_draft(rec):
                out.add(rec.get("image", f[:-5]))
        except Exception:
            pass
    return out
