import sys
import os
import json

OCR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ocr")
)
DETECTION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "Table_dimensions_detection")
)

sys.path.insert(0, OCR_DIR)
sys.path.insert(0, os.path.join(OCR_DIR, "models"))
sys.path.insert(0, DETECTION_DIR)

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from core.job_manager import create_job, run_job
from core.weights_finder import find_best_weights
from core.results_store import log_run, save_snapshot
from core import ocr_eval
from core.image_enhancement import enhance

# fixed crop pre-processing: 2x upscale plus sharpen
CROP_ENHANCE = "basic"

router = APIRouter()


class OCRConfig(BaseModel):
    task:            str   = "tables"     # "tables" | "dimensions" | "both"
    ocr_model:       str   = "easyocr"
    mode:            str   = "crop"       # "crop" (detection crops) | "full" (whole image)
    crop_source:     str   = "detector"   # "detector" (model boxes) | "gt" (ground-truth boxes = OCR ceiling)
    conf_threshold:  float = 0.25
    imgsz:           int   = 640


def _prioritize_paths(*dirs):
    """Move dirs to the front of sys.path (ocr/vlm/detection all ship a top-level
    `models` package, so the correct one must win regardless of import order)."""
    for d in reversed(dirs):
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)


def _load_ocr(model_name: str):
    """Load any registered OCR engine by dynamically importing its wrapper class
    (declared in the registry). Adding an architecture = a wrapper file in
    ocr/models/ + a registry entry — no change here."""
    import importlib
    from core.ocr_registry import get_model

    # resolve models.base against ocr/, not vlm/ or Table_dimensions_detection/
    _prioritize_paths(OCR_DIR, os.path.join(OCR_DIR, "models"))
    for m in ("models", "models.base", "base", "easyocr_model", "tesseract_model",
              "trocr_model", "paddleocr_model"):
        sys.modules.pop(m, None)

    entry = get_model(model_name)
    if not entry:
        entry = {"easyocr": {"wrapper_module": "easyocr_model", "wrapper_class": "EasyOCRModel"},
                 "tesseract": {"wrapper_module": "tesseract_model", "wrapper_class": "TesseractModel"}}.get(model_name)
    if not entry or not entry.get("wrapper_module"):
        raise ValueError(f"Unknown or unwrapped OCR model: {model_name}")

    try:
        mod = importlib.import_module(entry["wrapper_module"])
        cls = getattr(mod, entry["wrapper_class"])
    except ImportError as e:
        raise RuntimeError(
            f"'{entry.get('label', model_name)}' isn't installed. Run: "
            f"{entry.get('install_cmd', 'see registry')}  (import error: {e})")
    # EasyOCR takes languages/gpu; others take no args.
    return cls(languages=["en"], gpu=False) if entry["wrapper_class"] == "EasyOCRModel" else cls()


def _read_regions(ocr, image):
    """Returns [(text, confidence)] for a whole image, using per-region output if available."""
    if hasattr(ocr, "read_with_confidence"):
        return ocr.read_with_confidence(image)
    text = ocr.read(image)
    return [(text, 1.0)] if text.strip() else []


REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
UNIFIED_DIR = os.path.join(REPO_ROOT, "dataset", "selected_images")
MASTER_DIR  = os.path.join(REPO_ROOT, "dataset", "master", "unified")


def _gt_boxes(task: str, stem: str):
    """Ground-truth boxes for a task from the master annotation (xyxy). Used for
    the 'OCR ceiling' crop source — crops the true regions, not the detector's."""
    target = "table" if task == "tables" else "dimension"
    path = os.path.join(MASTER_DIR, f"{stem}.json")
    if not os.path.exists(path):
        return []
    rec = json.load(open(path))
    out = []
    for r in rec.get("regions", []):
        if r.get("class") == target and r.get("bbox"):
            b = r["bbox"]
            out.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
    return out


def _unified_ready() -> bool:
    return os.path.isdir(UNIFIED_DIR) and any(
        f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(UNIFIED_DIR))


def _get_images_dir(task: str) -> str:
    if _unified_ready():
        return UNIFIED_DIR
    if task == "tables":
        return os.path.join(DETECTION_DIR, "data", "tables", "valid")
    return os.path.join(DETECTION_DIR, "data", "dimensions", "val", "images")


def _run_ocr_for_task(task: str, cfg: OCRConfig, job):
    import numpy as np
    from PIL import Image
    from pathlib import Path

    mode = cfg.mode if cfg.mode in ("crop", "full") else "crop"
    use_gt = mode == "crop" and cfg.crop_source == "gt"
    # File/namespace label: full | crop (detector boxes) | gtcrop (ground-truth boxes).
    label = "full" if mode == "full" else ("gtcrop" if use_gt else "crop")

    job.log(f"[{task}] Loading OCR model: {cfg.ocr_model}...")
    ocr = _load_ocr(cfg.ocr_model)

    images_dir  = _get_images_dir(task)
    output_dir  = os.path.join(OCR_DIR, "results")
    crops_dir   = os.path.join(output_dir, "crops")
    json_dir    = os.path.join(output_dir, "json", task, label)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(json_dir,  exist_ok=True)

    # Detector-box crops need trained weights; GT-box crops and whole-image do not.
    detector = None
    if mode == "crop" and not use_gt:
        weights = find_best_weights(task)
        if not weights:
            raise RuntimeError(f"No trained weights found for {task}. Train detection first, or use GT crops / full image.")
        job.log(f"[{task}] Loading detector from {weights}...")
        _prioritize_paths(DETECTION_DIR, os.path.join(DETECTION_DIR, "models"))
        for m in ("models", "models.yolov11", "models.rtdetr", "models.base"):
            sys.modules.pop(m, None)
        if "rtdetr" in weights.lower():
            from models.rtdetr import RTDETRDetector
            detector = RTDETRDetector(weights=weights)
        else:
            from models.yolov11 import YOLOv11Detector
            detector = YOLOv11Detector(weights=weights)
    elif use_gt:
        job.log(f"[{task}] Using GROUND-TRUTH boxes as crop source (OCR ceiling).")

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    job.log(f"[{task}] Processing {len(image_files)} images ({label} mode)...")
    all_results = []

    for i, fname in enumerate(image_files):
        img_path = os.path.join(images_dir, fname)
        image    = Image.open(img_path).convert("RGB")
        stem     = Path(fname).stem
        detections = []

        if mode == "crop":
            if use_gt:
                boxes  = _gt_boxes(task, stem)
                scores = [1.0] * len(boxes)
            else:
                pred   = detector.predict([np.array(image)], conf_threshold=cfg.conf_threshold, imgsz=cfg.imgsz)[0]
                boxes  = pred["boxes"].tolist()
                scores = pred["scores"].tolist()

            img_crops_dir = os.path.join(crops_dir, label, task, stem)
            os.makedirs(img_crops_dir, exist_ok=True)

            for j, (box, score) in enumerate(zip(boxes, scores)):
                w, h = image.size
                x1 = max(0, int(box[0]) - 4)
                y1 = max(0, int(box[1]) - 4)
                x2 = min(w, int(box[2]) + 4)
                y2 = min(h, int(box[3]) + 4)
                crop = image.crop((x1, y1, x2, y2))
                text = ocr.read(enhance(crop, CROP_ENHANCE))

                crop.save(os.path.join(img_crops_dir, f"crop_{j:03d}.jpg"))
                detections.append({
                    "id": j, "class": task,
                    "confidence": round(score, 4),
                    "bbox": [round(v, 2) for v in box],
                    "text": text,
                })
        else:  # full image, OCR the whole drawing, one detection per text region
            for j, (text, conf) in enumerate(_read_regions(ocr, image)):
                detections.append({
                    "id": j, "class": task,
                    "confidence": round(float(conf), 4),
                    "bbox": None,
                    "text": text,
                })

        result = {
            "image":      stem,
            "task":       task,
            "ocr_model":  cfg.ocr_model,
            "mode":       label,
            "detections": detections,
        }
        all_results.append(result)

        with open(os.path.join(json_dir, f"{stem}.json"), "w") as f:
            json.dump(result, f, indent=2)

        if (i + 1) % 5 == 0:
            job.log(f"[{task}] {i+1}/{len(image_files)} images processed...")

    combined = os.path.join(output_dir, f"{task}_{cfg.ocr_model}_{label}_results.json")
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)

    all_dets = [d for r in all_results for d in r.get("detections", [])]
    with_text = sum(1 for d in all_dets if (d.get("text") or "").strip())
    avg_conf  = (sum(d.get("confidence", 0) for d in all_dets) / len(all_dets)) if all_dets else 0
    coverage  = (with_text / len(all_dets)) if all_dets else 0

    predicted_by_image = {r["image"]: [d.get("text", "") for d in r["detections"]] for r in all_results}
    gt = ocr_eval.load_whole_image_gt() if mode == "full" else None
    acc = ocr_eval.evaluate(predicted_by_image, task, gt=gt)
    with open(os.path.join(output_dir, f"{task}_{cfg.ocr_model}_{label}_metrics.json"), "w") as f:
        json.dump(acc, f, indent=2)

    run_metrics = {"coverage": round(coverage, 4), "avg_confidence": round(avg_conf, 4), "total_patches": len(all_dets)}
    if acc.get("available") and acc.get("evaluated_images"):
        run_metrics.update({"cer": acc["cer"], "wer": acc["wer"], "exact_match": acc["exact_match"], "f1": acc["f1"]})
        job.log(f"[{task}] Accuracy vs ground truth — CER {acc['cer']} · WER {acc['wer']} · exact {acc['exact_match']}")

    entry = log_run(
        stage="ocr", task=task, model=f"{cfg.ocr_model}_{label}",
        metrics=run_metrics,
        extra={"conf_threshold": cfg.conf_threshold, "mode": label, "crop_source": cfg.crop_source},
    )
    gt_union = ocr_eval.load_whole_image_gt()
    pred_by  = {r["image"]: [d.get("text", "") for d in r["detections"]] for r in all_results}
    snap_detail = [{"image": r["image"], **ocr_eval.whole_text_detail(pred_by[r["image"]], gt_union.get(r["image"], []))}
                   for r in all_results]
    snap_agg = ocr_eval.evaluate_whole(pred_by, gt_union)
    save_snapshot(entry["id"], {
        "stage": "ocr", "task": task, "model": f"{cfg.ocr_model} · {label}", "approach": label,
        "timestamp": entry["timestamp"],
        "metrics": {k: snap_agg.get(k) for k in ("word_coverage", "char_coverage", "word_precision",
                                                 "matched_words", "n_gt_words", "n_pred_words", "evaluated_images")},
        "view": {"approach": label, "ocr_model": cfg.ocr_model, "detail": snap_detail},
    })

    job.log(f"[{task}] Done. Results saved to {combined}")
    return all_results


@router.post("/run")
async def run_ocr(cfg: OCRConfig, background: BackgroundTasks):
    tasks = ["tables", "dimensions"] if cfg.task == "both" else [cfg.task]
    job   = create_job(f"OCR · {cfg.task} · {cfg.ocr_model}")

    def _run(job, cfg, tasks):
        all_results = {}
        for task in tasks:
            results = _run_ocr_for_task(task, cfg, job)
            all_results[task] = len(results)
        return all_results

    background.add_task(run_job, job, _run, cfg=cfg, tasks=tasks)
    return {"job_id": job.job_id}


@router.get("/results/{task}/{model}")
def get_results(task: str, model: str, mode: str = "crop"):
    path = os.path.join(OCR_DIR, "results", f"{task}_{model}_{mode}_results.json")
    if not os.path.exists(path):
        legacy = os.path.join(OCR_DIR, "results", f"{task}_{model}_results.json")
        path = legacy if os.path.exists(legacy) else path
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@router.get("/metrics/{task}/{model}")
def get_metrics(task: str, model: str, mode: str = "crop"):
    """
    Accuracy metrics vs ground truth. Computed live from saved OCR results +
    ocr/data/ground_truth/<task>.json, so adding ground truth later needs no re-run.
    """
    results = get_results(task, model, mode)
    predicted_by_image = {r["image"]: [d.get("text", "") for d in r.get("detections", [])] for r in results}
    gt = ocr_eval.load_whole_image_gt() if mode == "full" else None
    return ocr_eval.evaluate(predicted_by_image, task, gt=gt)


def _load_full_pred(model: str) -> dict:
    """Whole-page OCR predictions per image (the 'full image' approach)."""
    path = os.path.join(OCR_DIR, "results", f"all_{model}_full_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r["image"]: [d.get("text", "") for d in r.get("detections", []) if d.get("text")]
            for r in data}


def _load_crop_pred(model: str, label: str = "crop") -> dict:
    """Crop-OCR predictions per image, MERGED across tasks. label = 'crop'
    (detector boxes) or 'gtcrop' (ground-truth boxes = OCR ceiling)."""
    merged = {}
    for task in ("tables", "dimensions"):
        path = os.path.join(OCR_DIR, "results", f"{task}_{model}_{label}_results.json")
        if not os.path.exists(path) and label == "crop":
            path = os.path.join(OCR_DIR, "results", f"{task}_{model}_results.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data:
            merged.setdefault(r["image"], []).extend(
                [d.get("text", "") for d in r.get("detections", []) if d.get("text")])
    return merged


@router.get("/detail")
def ocr_detail(model: str = "easyocr"):
    """Per-image + aggregate comparison of the OCR approaches (whole image vs
    detector crops vs ground-truth crops). Uses WHOLE-TEXT coverage (word +
    character) so table blocks and whole-page OCR are scored correctly."""
    gt         = ocr_eval.load_whole_image_gt()
    approaches = {"full":   _load_full_pred(model),
                  "gtcrop": _load_crop_pred(model, "gtcrop"),
                  "crop":   _load_crop_pred(model, "crop")}

    def build(pred_by_image):
        per = []
        for img in sorted(gt):
            if img not in pred_by_image:
                continue
            per.append({"image": img, **ocr_eval.whole_text_detail(pred_by_image[img], gt[img])})
        return per

    detail  = {k: build(v) for k, v in approaches.items()}
    compare = {k: ocr_eval.evaluate_whole(v, gt) for k, v in approaches.items()}
    return {"gt_images": sorted(gt.keys()), "detail": detail, "compare": compare,
            "available": bool(gt)}


@router.get("/crops/{task}/{image_name}")
def list_crops(task: str, image_name: str):
    crops_dir = os.path.join(OCR_DIR, "results", "crops", task, image_name)
    if not os.path.isdir(crops_dir):
        return []
    return sorted(os.listdir(crops_dir))


@router.get("/crops-detail/{image_name}")
def crops_detail(image_name: str, model: str = "easyocr", source: str = "detector"):
    """Every crop produced for one drawing (across tasks), with its OCR text,
    confidence, box and an image URL — to judge crop quality. source = 'detector'
    (model boxes) or 'gt' (ground-truth boxes)."""
    label = "gtcrop" if source == "gt" else "crop"
    out = []
    for task in ("tables", "dimensions"):
        path = os.path.join(OCR_DIR, "results", f"{task}_{model}_{label}_results.json")
        if not os.path.exists(path) and label == "crop":
            path = os.path.join(OCR_DIR, "results", f"{task}_{model}_results.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        rec = next((r for r in data if r["image"] == image_name), None)
        if not rec:
            continue
        crops_dir = os.path.join(OCR_DIR, "results", "crops", label, task, image_name)
        for d in rec.get("detections", []):
            fname = f"crop_{d['id']:03d}.jpg"
            if not os.path.exists(os.path.join(crops_dir, fname)):
                continue
            out.append({
                "task": task, "id": d["id"], "text": d.get("text", ""),
                "confidence": d.get("confidence"), "bbox": d.get("bbox"),
                "url": f"/ocr/crop-file/{label}/{task}/{image_name}/{fname}",
            })
    return {"image": image_name, "crops": out, "count": len(out)}


@router.get("/crop-file/{label}/{task}/{image_name}/{filename}")
def crop_file(label: str, task: str, image_name: str, filename: str):
    path = os.path.join(OCR_DIR, "results", "crops", label, task, image_name, filename)
    if not os.path.exists(path):
        return {"error": "not found"}
    return FileResponse(path)