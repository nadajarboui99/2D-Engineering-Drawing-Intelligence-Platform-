import sys
import os
import json

VLM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "vlm")
)
OCR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ocr")
)
DETECTION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "Table_dimensions_detection")
)

sys.path.insert(0, VLM_DIR)
sys.path.insert(0, os.path.join(VLM_DIR, "models"))
sys.path.insert(0, os.path.join(VLM_DIR, "prompts"))

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import List
from core.job_manager import create_job, run_job
from core.results_store import log_run
from core import vlm_eval

router = APIRouter()


class VLMConfig(BaseModel):
    vlm_model:  str        = "claude"
    modes:      List[str]  = ["whole_image", "whole_image_ocr", "cropped_ocr"]
    task:       str        = "tables"   # "tables" | "dimensions" | "both"
    ocr_model:  str        = "easyocr"


REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UNIFIED_DIR = os.path.join(REPO_ROOT, "dataset", "selected_images")


def _unified_ready() -> bool:
    return os.path.isdir(UNIFIED_DIR) and any(
        f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(UNIFIED_DIR))


def _get_images_dir(task: str) -> str:
    # Once a unified evaluation set is selected, every task/stage runs on it.
    if _unified_ready():
        return UNIFIED_DIR
    if task == "tables":
        return os.path.join(DETECTION_DIR, "data", "tables", "valid")
    return os.path.join(DETECTION_DIR, "data", "dimensions", "val", "images")


def _load_ocr_results(task: str, ocr_model: str) -> dict:
    # Crop-mode OCR (per task): detections carry both text AND bbox — used for the
    # "cropped_ocr" context (full image + crop text + box coordinates).
    # task "both" merges the tables + dimensions crop detections per image.
    if task == "both":
        merged = {}
        for t in ("tables", "dimensions"):
            for img, rec in _load_ocr_results(t, ocr_model).items():
                merged.setdefault(img, {"image": img, "detections": []})
                merged[img]["detections"].extend(rec.get("detections", []))
        return merged
    candidates = [
        os.path.join(OCR_DIR, "results", f"{task}_{ocr_model}_crop_results.json"),
        os.path.join(OCR_DIR, "results", f"{task}_{ocr_model}_results.json"),  # legacy
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r["image"]: r for r in data}


def _load_full_image_ocr(ocr_model: str) -> dict:
    # Whole-image OCR (task-agnostic, keyed "all"): the text read off the whole page.
    # Used for the "whole_image_ocr" context.
    path = os.path.join(OCR_DIR, "results", f"all_{ocr_model}_full_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r["image"]: [d.get("text", "") for d in r.get("detections", []) if d.get("text")]
            for r in data}


def _vlm_setup(cfg: VLMConfig):
    """One-time imports + shared resources for a run."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(VLM_DIR, ".env"))

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.api_keys import load_all_to_env
    load_all_to_env()

    # Force `from models.base import BaseVLM` to resolve against vlm/, not ocr/ or
    # Table_dimensions_detection/ which also ship a top-level `models` package.
    for d in (os.path.join(VLM_DIR, "prompts"), os.path.join(VLM_DIR, "models"), VLM_DIR):
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
    for m in ("models", "models.base", "base", "claude_vlm", "mastra_vlm", "prompt_builder"):
        sys.modules.pop(m, None)

    from prompt_builder import load_schema, build_prompt
    from mastra_vlm import MastraVLMWrapper
    from pathlib import Path

    images_dir = _get_images_dir(cfg.task if cfg.task != "both" else "dimensions")
    image_names = sorted({
        Path(f).stem for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    })
    return {
        "schema":       load_schema(os.path.join(VLM_DIR, "configs", "feature_schema.yaml")),
        "build_prompt": build_prompt,
        "vlm":          MastraVLMWrapper(model=cfg.vlm_model),
        "full_ocr":     _load_full_image_ocr(cfg.ocr_model),
        "images_dir":   images_dir,
        "image_names":  image_names,
    }


def _process_mode(mode: str, task_label: str, crop_task, cfg: VLMConfig, ctx: dict, job):
    """Run ONE input mode over every image. `task_label` is stored in the result
    (its class scope); `crop_task` selects which crop OCR feeds cropped_ocr."""
    from PIL import Image as PILImage
    from core.config_loader import read_text

    crop_ocr   = _load_ocr_results(crop_task, cfg.ocr_model) if mode == "cropped_ocr" else {}
    output_dir = os.path.join(VLM_DIR, "results", task_label)
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, image_name in enumerate(ctx["image_names"]):
        job.log(f"[{mode}] Image {i+1}/{len(ctx['image_names'])}: {image_name[:30]}...")

        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(ctx["images_dir"], image_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path:
            job.log(f"  [WARN] image not found: {image_name}")
            continue

        # Every mode SEES the full drawing; context modes add OCR text as TEXT context.
        images = [PILImage.open(img_path).convert("RGB")]

        if mode == "whole_image":
            text_context = ""
        elif mode == "whole_image_ocr":
            texts = ctx["full_ocr"].get(image_name, [])
            text_context = "\n".join(texts)
            if not texts:
                job.log(f"  [WARN] no whole-image OCR for {image_name} — run OCR (Full image) first")
        elif mode == "cropped_ocr":
            dets = crop_ocr.get(image_name, {}).get("detections", [])
            lines = []
            for d in dets:
                box = d.get("bbox")
                coord = ("[box " + ", ".join(str(round(v)) for v in box) + "]") if box else "[box n/a]"
                lines.append(f"{coord} {d.get('text','')}".strip())
            text_context = "\n".join(lines)
            if not dets:
                job.log(f"  [WARN] no crop OCR for {image_name} — run OCR (Cropped) first")
        else:
            continue

        custom_prompt = read_text(f"prompts_{mode}")
        prompt = custom_prompt if custom_prompt.strip() else ctx["build_prompt"](ctx["schema"], mode, text_context)

        extracted = ctx["vlm"].extract(images, text_context, prompt)

        result = {
            "image":     image_name,
            "task":      task_label,
            "mode":      mode,
            "vlm_model": cfg.vlm_model,
            "extracted": extracted,
            "usage":     dict(ctx["vlm"].last_meta),
        }
        results.append(result)
        with open(os.path.join(output_dir, f"{image_name}_{mode}.json"), "w") as f:
            json.dump(result, f, indent=2)

    return results


def _run_vlm(cfg: VLMConfig, job):
    """Whole-image modes are task-agnostic (run once, tagged 'all'); cropped_ocr
    runs once with the chosen crop task (tables | dimensions | both)."""
    ctx = _vlm_setup(cfg)
    combined = []
    for mode in cfg.modes:
        if mode == "cropped_ocr":
            combined += _process_mode(mode, task_label=cfg.task, crop_task=cfg.task, cfg=cfg, ctx=ctx, job=job)
        else:
            combined += _process_mode(mode, task_label="all", crop_task=None, cfg=cfg, ctx=ctx, job=job)
    return combined


@router.post("/run")
async def run_vlm(cfg: VLMConfig, background: BackgroundTasks):
    job = create_job(f"VLM · {cfg.vlm_model} · {', '.join(cfg.modes)}")

    def _run(job, cfg):
        combined_all = _run_vlm(cfg, job)

        # Save combined
        combined_path = os.path.join(VLM_DIR, "results", "all_results.json")
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)

        # Merge with existing results
        existing = []
        if os.path.exists(combined_path):
            with open(combined_path) as f:
                existing = json.load(f)

        # Overwrite entries with same image+mode+task
        key = lambda r: (r["image"], r["mode"], r.get("task",""))
        existing_map = {key(r): r for r in existing}
        for r in combined_all:
            existing_map[key(r)] = r

        with open(combined_path, "w") as f:
            json.dump(list(existing_map.values()), f, indent=2)

        # Log to unified results store, grouped by mode
        for mode in cfg.modes:
            mode_results = [r for r in combined_all if r["mode"] == mode and not r["extracted"].get("error")]
            if not mode_results:
                continue
            total_fields  = sum(len(r["extracted"]) for r in mode_results)
            filled_fields = sum(sum(1 for v in r["extracted"].values() if v is not None) for r in mode_results)
            fill_rate = (filled_fields / total_fields) if total_fields else 0
            res = _aggregate_resources(mode_results)
            task_label = mode_results[0].get("task", cfg.task)
            log_run(
                stage="vlm", task=task_label, model=f"{cfg.vlm_model}_{mode}",
                metrics={"fill_rate": round(fill_rate, 4), "images_processed": len(mode_results),
                         **res},
                extra={"mode": mode, "vlm_model": cfg.vlm_model},
            )

        job.log(f"All done. {len(combined_all)} results saved.")
        return {"count": len(combined_all)}

    background.add_task(run_job, job, _run, cfg=cfg)
    return {"job_id": job.job_id}


def _aggregate_resources(results: list) -> dict:
    """Average tokens / latency and total cost across a set of VLM results."""
    usages = [r.get("usage") or {} for r in results]
    def _avg(key):
        vals = [u.get(key) for u in usages if isinstance(u.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None
    costs = [u.get("cost_usd") for u in usages if isinstance(u.get("cost_usd"), (int, float))]
    return {
        "avg_input_tokens":  _avg("input_tokens"),
        "avg_output_tokens": _avg("output_tokens"),
        "avg_total_tokens":  _avg("total_tokens"),
        "avg_latency_s":     _avg("latency_s"),
        "total_cost_usd":    round(sum(costs), 6) if costs else None,
        "n_calls":           len(results),
    }


@router.get("/results")
def get_results(task: str = None, mode: str = None):
    path = os.path.join(VLM_DIR, "results", "all_results.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if task:
        data = [r for r in data if r.get("task") == task]
    if mode:
        data = [r for r in data if r.get("mode") == mode]
    return data


@router.get("/metrics")
def get_metrics(task: str = None, mode: str = None):
    """
    Feature-extraction accuracy vs the unified ground truth
    (vlm/data/ground_truth/unified.json). The VLM extracts one feature set per
    drawing, so scoring is task-independent — `mode` selects which run to score.
    `task` is accepted for backward-compat but not used for the answer key.
    """
    path = os.path.join(VLM_DIR, "results", "all_results.json")
    if not os.path.exists(path):
        gt = vlm_eval.load_unified_gt()
        return {"available": bool(gt), "mode": mode, "evaluated_images": 0, "gt_images": len(gt)}
    with open(path) as f:
        results = json.load(f)
    out = vlm_eval.evaluate(results, mode)
    # Resource usage is available even without ground truth.
    rows = [r for r in results if not (r.get("extracted") or {}).get("error")]
    if mode:
        rows = [r for r in rows if r.get("mode") == mode]
    out["resources"] = _aggregate_resources(rows)
    return out


@router.get("/detail")
def get_detail():
    """Per-image, per-field breakdown (extracted vs ground truth + verdict) for
    every mode — powers the image inspector so you can eyeball where a value was
    scored wrong (e.g. a right answer marked wrong on formatting)."""
    path = os.path.join(VLM_DIR, "results", "all_results.json")
    gt = vlm_eval.load_unified_gt()
    if not os.path.exists(path):
        return {"detail": {"whole_image": [], "whole_image_ocr": [], "cropped_ocr": []},
                "gt_images": sorted(gt.keys())}
    with open(path) as f:
        results = json.load(f)
    return {"detail": vlm_eval.detail(results, gt), "gt_images": sorted(gt.keys())}


@router.get("/compare")
def compare_modes(task: str = None):
    """Side-by-side accuracy + resource usage for every mode — answers
    'did context help, and which pipeline is best?'. All modes are scored against
    the same unified answer key, so it's apples-to-apples."""
    path = os.path.join(VLM_DIR, "results", "all_results.json")
    if not os.path.exists(path):
        return {"modes": []}
    with open(path) as f:
        results = json.load(f)
    out = []
    for mode in ("whole_image", "whole_image_ocr", "cropped_ocr"):
        acc  = vlm_eval.evaluate(results, mode)
        rows = [r for r in results
                if r.get("mode") == mode and not (r.get("extracted") or {}).get("error")]
        out.append({
            "mode": mode,
            "evaluated_images":   acc.get("evaluated_images", 0),
            "field_accuracy":     acc.get("field_accuracy"),
            "overall_accuracy":   acc.get("overall_accuracy"),
            "error_rate":         acc.get("error_rate"),
            "miss_rate":          acc.get("miss_rate"),
            "hallucination_rate": acc.get("hallucination_rate"),
            "numeric_mape":       acc.get("numeric_mape"),
            "exact_match":        acc.get("exact_match"),
            "resources":          _aggregate_resources(rows),
        })
    return {"modes": out}