"""
Annotation API — backs the in-app annotator.

Saves an annotated image into the unified dataset exactly like the master
format produced by dataset/build_master.py, so hand-editing JSON and using the
UI are interchangeable. On save it also regenerates the per-stage ground truth
(dataset/build_from_master.py) so results stay in sync.
"""
import os
import sys
import json
import shutil
import subprocess

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SELECTED_DIR = os.path.join(ROOT, "dataset", "selected_images")
UNIFIED_DIR  = os.path.join(ROOT, "dataset", "master", "unified")
BUILD_SCRIPT = os.path.join(ROOT, "dataset", "build_from_master.py")
IMG_EXT = (".jpg", ".jpeg", ".png")

META_DEFAULT = {"standard": "unknown", "source_type": "unknown",
                "clutter": "med", "has_gdt": False, "difficulty": "med"}

router = APIRouter()


def _find_image(stem: str):
    if os.path.isdir(SELECTED_DIR):
        for f in os.listdir(SELECTED_DIR):
            if os.path.splitext(f)[0] == stem and f.lower().endswith(IMG_EXT):
                return os.path.join(SELECTED_DIR, f)
    return None


@router.get("/list")
def list_annotations():
    """Every unified master, with quick annotation counts (for the 'edit' list)."""
    out = []
    if os.path.isdir(UNIFIED_DIR):
        for f in sorted(os.listdir(UNIFIED_DIR)):
            if not f.endswith(".json") or f.startswith("_"):
                continue
            rec = json.load(open(os.path.join(UNIFIED_DIR, f)))
            stem = rec.get("image", f[:-5])
            regs = rec.get("regions", [])
            feats = rec.get("features", {}) or {}
            out.append({
                "image": stem,
                "regions": len(regs),
                "with_text": sum(1 for r in regs if (r.get("text") or "").strip()),
                "features": sum(1 for s in feats.values()
                                if (s.get("value") if isinstance(s, dict) else s) not in (None, "")),
                "has_image": _find_image(stem) is not None,
                "status": rec.get("status", "complete"),   # missing = complete (back-compat)
            })
    return out


@router.get("/image/{stem}")
def get_image(stem: str):
    p = _find_image(stem)
    if not p:
        raise HTTPException(404, f"image not found for {stem}")
    return FileResponse(p)


@router.get("/master/{stem}")
def get_master(stem: str):
    p = os.path.join(UNIFIED_DIR, f"{stem}.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def _rebuild_gt():
    try:
        subprocess.run([sys.executable, BUILD_SCRIPT], cwd=ROOT, timeout=120, capture_output=True)
    except Exception:
        pass


@router.delete("/{stem}")
def delete_annotation(stem: str):
    removed = []
    master = os.path.join(UNIFIED_DIR, f"{stem}.json")
    if os.path.exists(master):
        os.remove(master)
        removed.append("master")
    img = _find_image(stem)
    if img:
        os.remove(img)
        removed.append("image")
    if not removed:
        raise HTTPException(404, f"no annotation found for {stem}")
    _rebuild_gt()
    return {"ok": True, "removed": removed}


@router.post("/save")
async def save(payload: str = Form(...), image: UploadFile = File(None)):
    data = json.loads(payload)
    stem = (data.get("image") or "").strip()
    if not stem:
        raise HTTPException(400, "missing image name")

    os.makedirs(SELECTED_DIR, exist_ok=True)
    os.makedirs(UNIFIED_DIR, exist_ok=True)

    if image is not None:
        ext = os.path.splitext(image.filename or "")[1].lower()
        if ext not in IMG_EXT:
            ext = ".png"
        with open(os.path.join(SELECTED_DIR, stem + ext), "wb") as f:
            shutil.copyfileobj(image.file, f)

    # ids by order, cells only for tables
    regions = []
    for i, r in enumerate(data.get("regions", [])):
        bbox = [round(float(v)) for v in (r.get("bbox") or [0, 0, 0, 0])][:4]
        reg = {"id": i, "class": r.get("class", "dimension"), "bbox": bbox, "text": r.get("text", "")}
        if reg["class"] == "table":
            reg["cells"] = r.get("cells", [])
        regions.append(reg)

    # Normalize features to {name: {value, text, bbox}}.
    feats = {}
    for k, v in (data.get("features") or {}).items():
        if isinstance(v, dict):
            feats[k] = {"value": v.get("value"), "text": v.get("text", ""), "bbox": v.get("bbox")}
        else:
            feats[k] = {"value": v, "text": "", "bbox": None}

    # "draft" = saved so work isn't lost, but NOT part of evaluation until the
    # user marks it complete. Anything else counts as complete.
    status = "draft" if data.get("status") == "draft" else "complete"
    record = {
        "image": stem,
        "width": data.get("width"),
        "height": data.get("height"),
        "status": status,
        "meta": data.get("meta") or dict(META_DEFAULT),
        "regions": regions,
        "features": feats,
    }
    with open(os.path.join(UNIFIED_DIR, f"{stem}.json"), "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    _rebuild_gt()

    return {"ok": True, "image": stem, "regions": len(regions), "status": status}
