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
from core.results_store import log_run, save_snapshot
from core import vlm_eval

router = APIRouter()


class VLMConfig(BaseModel):
    vlm_model:  str        = "claude"
    modes:      List[str]  = ["whole_image", "whole_image_ocr", "cropped_ocr"]
    task:       str        = "tables"   # "tables" | "dimensions" | "both"
    ocr_model:  str        = "easyocr"  # which OCR output feeds the context modes
    # cropped_ocr: which detector's crops to use. "" / "default" = the legacy
    # best-detector crop file; "gt" = ground-truth crops (ceiling); or a det_id
    # (e.g. "doclayout-yolo", "yolov11_n-5") to use that detector's tagged crops.
    crop_detector: str     = "default"


REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UNIFIED_DIR = os.path.join(REPO_ROOT, "dataset", "selected_images")


def _unified_ready() -> bool:
    return os.path.isdir(UNIFIED_DIR) and any(
        f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(UNIFIED_DIR))


def _get_images_dir(task: str) -> str:
    if _unified_ready():
        return UNIFIED_DIR
    if task == "tables":
        return os.path.join(DETECTION_DIR, "data", "tables", "valid")
    return os.path.join(DETECTION_DIR, "data", "dimensions", "val", "images")


def _crop_source_tag(crop_detector: str) -> str:
    """Map the composer's crop_detector choice to the OCR result-file source tag."""
    cd = (crop_detector or "default").strip()
    if cd in ("", "default", "best"):
        return "crop"                  # legacy best-detector crop file
    if cd == "gt":
        return "gtcrop"                # ground-truth crops (ceiling)
    return f"crop-{cd}"                # a specific detector's tagged crops


def _load_ocr_results(task: str, ocr_model: str, crop_detector: str = "default") -> dict:
    # crop-mode OCR per task, detections carry both text and bbox
    # task "both" merges the tables and dimensions crop detections per image
    if task == "both":
        merged = {}
        for t in ("tables", "dimensions"):
            for img, rec in _load_ocr_results(t, ocr_model, crop_detector).items():
                merged.setdefault(img, {"image": img, "detections": []})
                merged[img]["detections"].extend(rec.get("detections", []))
        return merged
    tag = _crop_source_tag(crop_detector)
    candidates = [
        os.path.join(OCR_DIR, "results", f"{task}_{ocr_model}_{tag}_results.json"),
    ]
    if tag == "crop":  # backward-compat with older untagged filename
        candidates.append(os.path.join(OCR_DIR, "results", f"{task}_{ocr_model}_results.json"))
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r["image"]: r for r in data}


def _load_full_image_ocr(ocr_model: str) -> dict:
    # whole-image OCR, task-agnostic and keyed "all"
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

    # resolve models.base against vlm/, not ocr/ or Table_dimensions_detection/
    for d in (os.path.join(VLM_DIR, "prompts"), os.path.join(VLM_DIR, "models"), VLM_DIR):
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
    for m in ("models", "models.base", "base", "claude_vlm", "mastra_vlm", "prompt_builder"):
        sys.modules.pop(m, None)

    from prompt_builder import load_schema, build_prompt
    from mastra_vlm import MastraVLMWrapper
    from pathlib import Path

    from core.eval_set import draft_stems
    _drafts = draft_stems()
    images_dir = _get_images_dir(cfg.task if cfg.task != "both" else "dimensions")
    image_names = sorted({
        Path(f).stem for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and Path(f).stem not in _drafts
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

    crop_ocr   = _load_ocr_results(crop_task, cfg.ocr_model, cfg.crop_detector) if mode == "cropped_ocr" else {}
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

        # A saved custom prompt only overrides the intro; the schema template and
        # the OCR context are always appended by build_prompt (so a custom prompt
        # can't silently drop them — that was the bug that made all modes behave
        # like "image only" and tanked accuracy).
        custom_prompt = read_text(f"prompts_{mode}")
        prompt = ctx["build_prompt"](ctx["schema"], mode, text_context, custom_instruction=custom_prompt)

        extracted = ctx["vlm"].extract(images, text_context, prompt)

        result = {
            "image":     image_name,
            "task":      task_label,
            "mode":      mode,
            "vlm_model": cfg.vlm_model,
            "extracted": extracted,
            "usage":     dict(ctx["vlm"].last_meta),
            # provenance: which upstream models produced this mode's context
            "context": {
                "ocr_model":     cfg.ocr_model if mode in ("whole_image_ocr", "cropped_ocr") else None,
                "crop_detector": cfg.crop_detector if mode == "cropped_ocr" else None,
            },
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

        combined_path = os.path.join(VLM_DIR, "results", "all_results.json")
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)

        existing = []
        if os.path.exists(combined_path):
            with open(combined_path) as f:
                existing = json.load(f)

        key = lambda r: (r["image"], r["mode"], r.get("task",""))
        existing_map = {key(r): r for r in existing}
        for r in combined_all:
            existing_map[key(r)] = r

        with open(combined_path, "w") as f:
            json.dump(list(existing_map.values()), f, indent=2)

        for mode in cfg.modes:
            mode_results = [r for r in combined_all if r["mode"] == mode and not r["extracted"].get("error")]
            if not mode_results:
                continue
            total_fields  = sum(len(r["extracted"]) for r in mode_results)
            filled_fields = sum(sum(1 for v in r["extracted"].values() if v is not None) for r in mode_results)
            fill_rate = (filled_fields / total_fields) if total_fields else 0
            res = _aggregate_resources(mode_results)
            task_label = mode_results[0].get("task", cfg.task)
            prov = (mode_results[0].get("context") or {})
            gt  = vlm_eval.load_unified_gt()
            det = [{"image": r["image"], **vlm_eval.image_detail(r.get("extracted") or {}, gt.get(r["image"], {}))}
                   for r in mode_results if r["image"] in gt]
            # Score THIS run's results for this mode (not the accumulated file), so
            # each logged run carries its own accuracy for the leaderboard.
            acc = vlm_eval.evaluate([r for r in combined_all if r["mode"] == mode], mode)
            acc_metrics = {k: acc.get(k) for k in ("field_accuracy", "hallucination_rate",
                           "miss_rate", "error_rate", "exact_match", "overall_accuracy", "numeric_mape")}
            entry = log_run(
                stage="vlm", task=task_label, model=f"{cfg.vlm_model}_{mode}",
                metrics={"fill_rate": round(fill_rate, 4), "images_processed": len(mode_results),
                         **acc_metrics, **res},
                extra={"mode": mode, "vlm_model": cfg.vlm_model,
                       "ocr_model": prov.get("ocr_model"), "crop_detector": prov.get("crop_detector")},
            )
            save_snapshot(entry["id"], {
                "stage": "vlm", "task": task_label, "model": f"{cfg.vlm_model} · {mode}", "mode": mode,
                "timestamp": entry["timestamp"],
                "metrics": {k: acc.get(k) for k in ("field_accuracy", "error_rate", "miss_rate",
                            "hallucination_rate", "numeric_mape", "exact_match", "overall_accuracy")},
                "view": {"mode": mode, "vlm_model": cfg.vlm_model, "detail": det, "resources": res,
                         "ocr_model": prov.get("ocr_model"), "crop_detector": prov.get("crop_detector")},
            })

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


@router.get("/available-context")
def available_context(task: str = "tables"):
    """What upstream OCR/detector outputs already exist on disk, so the VLM
    composer can offer only real, runnable pipeline combinations.
      - whole_image_ocr: OCR models with a whole-image result (keyed 'all')
      - cropped_ocr: (ocr_model, crop_detector) combos with a crop result for
        this task. crop_detector is 'default' (best trained), 'gt' (ground truth
        = ceiling), or a detector id (e.g. 'doclayout-yolo', 'yolov11_n-5')."""
    from core.ocr_registry import list_models as list_ocr
    ocr_ids = [m["id"] for m in list_ocr()]
    results_dir = os.path.join(OCR_DIR, "results")
    whole, seen = set(), set()
    if os.path.isdir(results_dir):
        files = set(os.listdir(results_dir))
        for ocr in ocr_ids:
            if f"all_{ocr}_full_results.json" in files:
                whole.add(ocr)
        tasks = ["tables", "dimensions"] if task == "both" else [task]
        for t in tasks:
            prefix, suffix = f"{t}_", "_results.json"
            for fn in files:
                if not (fn.startswith(prefix) and fn.endswith(suffix)):
                    continue
                middle = fn[len(prefix):-len(suffix)]
                for ocr in ocr_ids:
                    if middle.startswith(ocr + "_"):
                        tag = middle[len(ocr) + 1:]
                        if tag == "gtcrop":
                            seen.add((ocr, "gt"))
                        elif tag == "crop":
                            seen.add((ocr, "default"))
                        elif tag.startswith("crop-"):
                            seen.add((ocr, tag[5:]))
                        break
    return {
        "task": task,
        "whole_image_ocr": sorted(whole),
        "cropped_ocr": [{"ocr_model": o, "crop_detector": c} for o, c in sorted(seen)],
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