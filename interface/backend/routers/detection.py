import sys
import os
import csv
import asyncio

DETECTION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "Table_dimensions_detection")
)
sys.path.insert(0, DETECTION_DIR)

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from core.job_manager import create_job, run_job
from core.weights_finder import find_best_weights, list_all_runs
from core.results_store import log_run, get_runs, save_snapshot
from core.weights_registry import list_weights, get_weights_path
from core import detection_registry

REPO_ROOT    = os.path.dirname(DETECTION_DIR)
UNIFIED_DIR  = os.path.join(REPO_ROOT, "dataset", "master", "unified")
SELECTED_DIR = os.path.join(REPO_ROOT, "dataset", "selected_images")

router = APIRouter()


def _load_detector(weights: str):
    """Load a detector for the checkpoint, forcing `models` to resolve against
    Table_dimensions_detection/ (ocr/ and vlm/ also ship a `models` package)."""
    det_models = os.path.join(DETECTION_DIR, "models")
    for d in (DETECTION_DIR, det_models):
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
    for m in ("models", "models.yolov11", "models.rtdetr", "models.base"):
        sys.modules.pop(m, None)
    if "rtdetr" in weights.lower():
        from models.rtdetr import RTDETRDetector
        return RTDETRDetector(weights=weights)
    from models.yolov11 import YOLOv11Detector
    return YOLOv11Detector(weights=weights)


def _load_arch_detector(entry: dict):
    """Load a custom-architecture detector by dynamically importing its wrapper
    class from Table_dimensions_detection/models/ (registry-declared)."""
    import importlib
    det_models = os.path.join(DETECTION_DIR, "models")
    for d in (DETECTION_DIR, det_models):
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
    for m in ("models", "models.yolov11", "models.rtdetr", "models.base", entry["wrapper_module"]):
        sys.modules.pop(m, None)
    try:
        mod = importlib.import_module(entry["wrapper_module"])
        cls = getattr(mod, entry["wrapper_class"])
    except Exception as e:
        raise RuntimeError(
            f"'{entry.get('label')}' isn't installed. Run: {entry.get('install_cmd','see registry')} "
            f"(error: {e})")
    return cls(weights=entry.get("weights"))


class DetectionConfig(BaseModel):
    task:       str = "tables"
    model_name: str = "yolov11"
    model_size: str = "n"
    epochs:     int = 100
    batch_size: int = 16
    imgsz:      int = 640
    device:     str = "mps"


@router.get("/best-weights/{task}")
def get_best_weights(task: str):
    registered = list_weights(task)
    if registered:
        return {"path": registered[0]["path"], "found": True, "name": registered[0]["name"]}
    path = find_best_weights(task)
    return {"path": path, "found": path is not None, "name": None}


@router.get("/weights/{task}")
def get_weights_list(task: str):
    """
    Returns selectable weights for this task: user-registered/uploaded weights
    first, then trained runs auto-discovered under runs/detect (deduped by path).
    """
    registered = list_weights(task)
    known_paths = {os.path.abspath(e["path"]) for e in registered}

    discovered = []
    for r in list_all_runs(task):
        if os.path.abspath(r["weights"]) in known_paths:
            continue
        discovered.append({
            "id":     f"run_{r['run']}",
            "name":   f"{r['run']} (mAP {r['map50']:.3f})" if r["map50"] else r["run"],
            "task":   task,
            "path":   r["weights"],
            "source": "trained",
        })

    # custom-architecture models have no .pt path, they load via their wrapper
    arch_models = [{
        "id":        f"arch_{m['id']}",
        "arch_id":   m["id"],
        "name":      m["label"] + ("" if m["installed"] else "  (install required)"),
        "task":      task,
        "path":      None,
        "source":    "custom-arch",
        "arch":      m["arch"],
        "available": m["installed"],
        "install_cmd": m["install_cmd"],
    } for m in detection_registry.list_models(task)]

    return registered + discovered + arch_models


@router.get("/results/{task}")
def get_results(task: str):
    return list_all_runs(task)


@router.post("/run/{task}")
async def run_detection(task: str, cfg: DetectionConfig, background: BackgroundTasks):
    job = create_job(f"Detection · {task} · {cfg.model_name}-{cfg.model_size}")

    def _run(job, task, cfg):
        import importlib.util
        import sys

        script = "train_tables.py" if task == "tables" else "train_dimensions.py"
        script_path = os.path.join(DETECTION_DIR, script)

        sys.argv = [
            script,
            "--config", os.path.join(DETECTION_DIR, "configs", f"{'tables' if task == 'tables' else 'dimensions'}.yaml"),
            "--model",  cfg.model_name,
        ]

        import yaml
        cfg_path = os.path.join(DETECTION_DIR, "configs", f"{'tables' if task == 'tables' else 'dimensions'}.yaml")
        with open(cfg_path) as f:
            data = yaml.safe_load(f)

        data["model"]["name"]          = cfg.model_name
        data["model"]["size"]          = cfg.model_size
        data["training"]["epochs"]     = cfg.epochs
        data["training"]["batch_size"] = cfg.batch_size
        data["training"]["imgsz"]      = cfg.imgsz
        data["training"]["device"]     = cfg.device
        data["training"]["run_name"]   = f"{cfg.model_name}_{cfg.model_size}"

        with open(cfg_path, "w") as f:
            yaml.dump(data, f)

        job.log(f"Config updated. Starting {task} training with {cfg.model_name}-{cfg.model_size}...")

        prev_cwd = os.getcwd()
        os.chdir(DETECTION_DIR)
        try:
            spec = importlib.util.spec_from_file_location("train_script", script_path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        finally:
            os.chdir(prev_cwd)

        best = find_best_weights(task)
        job.log(f"Training complete. Best weights: {best}")
        return {"task": task, "best_weights": best}

    background.add_task(run_job, job, _run, task=task, cfg=cfg)
    return {"job_id": job.job_id}


class EvalConfig(BaseModel):
    weights_path: str | None = None
    arch_id:      str | None = None   # custom-architecture model (registry id), single-task only
    imgsz:        int | None = None


@router.post("/eval/{task}")
async def eval_detection(task: str, background: BackgroundTasks, cfg: EvalConfig = EvalConfig()):
    job = create_job(f"Eval · {task}")

    def _run(job, task, cfg):
        best = cfg.weights_path or find_best_weights(task)
        if not best:
            raise RuntimeError(f"No trained weights found for {task}. Train first.")
        if not os.path.exists(best):
            raise RuntimeError(f"Weights file not found: {best}")
        job.log(f"Using weights: {best}")

        import importlib.util
        script = "train_tables.py" if task == "tables" else "train_dimensions.py"
        script_path = os.path.join(DETECTION_DIR, script)
        cfg_path    = os.path.join(DETECTION_DIR, "configs", f"{'tables' if task == 'tables' else 'dimensions'}.yaml")

        if cfg.imgsz:
            import yaml
            with open(cfg_path) as f:
                data = yaml.safe_load(f)
            data["training"]["imgsz"] = cfg.imgsz
            with open(cfg_path, "w") as f:
                yaml.dump(data, f)

        model_name = "rtdetr" if "rtdetr" in best.lower() else "yolov11"
        job.log(f"Using model architecture: {model_name}")

        sys.argv = [script, "--config", cfg_path, "--eval_only", "--weights", best, "--model", model_name]

        # train scripts use dataset paths relative to Table_dimensions_detection/
        prev_cwd = os.getcwd()
        os.chdir(DETECTION_DIR)
        try:
            spec = importlib.util.spec_from_file_location("train_script", script_path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        finally:
            os.chdir(prev_cwd)

        job.log("Evaluation complete.")
        runs = list_all_runs(task)
        if runs:
            top = runs[0]
            log_run(
                stage="detection", task=task, model=top["run"],
                metrics={"map50": top["map50"], "precision": top["precision"], "recall": top["recall"]},
                extra={"weights": best, "epochs": top["epochs"]},
            )
        return runs

    background.add_task(run_job, job, _run, task=task, cfg=cfg)
    return {"job_id": job.job_id}


@router.post("/eval-annotated/{task}")
async def eval_annotated(task: str, background: BackgroundTasks, cfg: EvalConfig = EvalConfig()):
    """Evaluate a trained model on the ANNOTATED dataset (boxes you drew), not the
    training val set. Computes mAP@0.5 / precision / recall / F1 against your GT."""
    job = create_job(f"Eval on annotated · {task}")

    def _run(job, task, cfg):
        import glob, json
        import numpy as np
        from PIL import Image
        from core.detection_eval import evaluate as det_eval

        targets = (["table", "dimension"] if task == "both"
                   else ["dimension"] if task == "dimensions" else ["table"])

        masters = sorted(glob.glob(os.path.join(UNIFIED_DIR, "*.json")))
        if not masters:
            raise RuntimeError("No annotated images found (dataset/master/unified is empty).")

        # Ground truth boxes per (class, image), and a flat per-image list for overlay.
        gt_cls = {t: {} for t in targets}
        gt_viz = {}
        for p in masters:
            rec  = json.load(open(p))
            if rec.get("status") == "draft":
                continue   # unfinished annotation — not part of evaluation
            stem = rec.get("image") or os.path.splitext(os.path.basename(p))[0]
            for r in rec.get("regions", []):
                cls = r.get("class")
                if cls in targets and r.get("bbox"):
                    b = r["bbox"]
                    box = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
                    gt_cls[cls].setdefault(stem, []).append(box)
                    gt_viz.setdefault(stem, []).append({"box": [round(v, 1) for v in box], "cls": cls})
        if not gt_viz:
            raise RuntimeError(f"No {' or '.join(targets)} boxes annotated yet — nothing to score.")

        models = {}
        for t in targets:
            t_task = "tables" if t == "table" else "dimensions"
            if cfg.arch_id and task != "both":
                entry = detection_registry.get_model(cfg.arch_id)
                if not entry:
                    raise RuntimeError(f"Unknown architecture model: {cfg.arch_id}")
                models[t] = (_load_arch_detector(entry), entry["label"])
                job.log(f"{t} model: {entry['label']} (custom arch)")
            else:
                w = (cfg.weights_path if (task != "both" and cfg.weights_path) else None) or find_best_weights(t_task)
                if not w or not os.path.exists(w):
                    raise RuntimeError(f"No trained weights found for {t_task}.")
                models[t] = (_load_detector(w), w)
                job.log(f"{t} model: {w}")

        def _find_img(stem):
            for f in os.listdir(SELECTED_DIR):
                if os.path.splitext(f)[0] == stem and f.lower().endswith((".jpg", ".jpeg", ".png")):
                    return os.path.join(SELECTED_DIR, f)
            return None

        imgsz = cfg.imgsz or 640
        stems = sorted(gt_viz.keys())
        preds_cls = {t: {} for t in targets}   # class → {stem: [(x1,y1,x2,y2,score)]}
        preds_viz = {}                          # stem → [{box, score, cls}]
        img_dims  = {}

        for stem in stems:
            ip = _find_img(stem)
            if not ip:
                job.log(f"[warn] image file missing for {stem}")
                continue
            im = Image.open(ip).convert("RGB")
            img_dims[stem] = im.size
            for t, (model, _) in models.items():
                out = model.predict([np.array(im)], conf_threshold=0.001, imgsz=imgsz)[0]
                boxes, scores = out["boxes"].tolist(), out["scores"].tolist()
                preds_cls[t][stem] = [(b[0], b[1], b[2], b[3], s) for b, s in zip(boxes, scores)]
                top = sorted(
                    [{"box": [round(v, 1) for v in b], "score": round(s, 4), "cls": t}
                     for b, s in zip(boxes, scores) if s >= 0.05],
                    key=lambda d: d["score"], reverse=True)[:120]
                preds_viz.setdefault(stem, []).extend(top)
                job.log(f"{stem} [{t}]: {len(boxes)} raw preds vs {len(gt_cls[t].get(stem, []))} annotated")

        per_class = {}
        tp = fp = fn = n_gt = n_pred = 0
        maps = []
        for t in targets:
            m = det_eval(preds_cls[t], gt_cls[t], iou_thr=0.5, conf=0.25)
            per_class[t] = m
            tp += m.get("tp", 0); fp += m.get("fp", 0); fn += m.get("fn", 0)
            n_gt += m.get("n_gt", 0); n_pred += m.get("n_pred", 0)
            if m.get("map50") is not None:
                maps.append(m["map50"])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
        metrics = {
            "available": True,
            "map50": round(sum(maps) / len(maps), 4) if maps else 0.0,
            "precision": round(prec, 4), "recall": round(rec_, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt, "n_pred": n_pred,
            "conf_threshold": 0.25, "iou_threshold": 0.5,
            "per_class": {t: {k: per_class[t].get(k) for k in ("map50","precision","recall","f1","tp","fp","fn","n_gt")} for t in targets},
            "images": [{"image": s, "width": img_dims.get(s, (1000, 1000))[0], "height": img_dims.get(s, (1000, 1000))[1],
                        "gt": gt_viz.get(s, []), "pred": preds_viz.get(s, []),
                        "n_pred_total": sum(len(preds_cls[t].get(s, [])) for t in targets)} for s in stems if s in img_dims],
        }
        job.log(f"Annotated eval — mAP50 {metrics['map50']} · P {metrics['precision']} · R {metrics['recall']} · F1 {metrics['f1']} (TP {tp}/FP {fp}/FN {fn})")

        def _label(w):
            # w is either a weights path (YOLO/RT-DETR) or an arch model label.
            return os.path.basename(os.path.dirname(os.path.dirname(w))) if (isinstance(w, str) and w.endswith(".pt")) else w
        model_label = " + ".join(_label(w) for _, w in models.values())
        entry = log_run(
            stage="detection", task=task, model=f"{model_label} · annotated",
            metrics={"map50": metrics["map50"], "precision": metrics["precision"],
                     "recall": metrics["recall"], "f1": metrics["f1"]},
            extra={"on": "annotated", "n_gt": n_gt, "tp": tp, "fp": fp, "fn": fn, "images": len(stems)},
        )
        out_dir = os.path.join(DETECTION_DIR, "results")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"annotated_{task}.json"), "w") as f:
            json.dump({**metrics, "model_label": model_label}, f)
        save_snapshot(entry["id"], {"stage": "detection", "task": task, "model": model_label,
                                    "timestamp": entry["timestamp"], "metrics": {k: metrics[k] for k in ("map50","precision","recall","f1","tp","fp","fn","n_gt","n_pred")},
                                    "view": {"images": metrics["images"], "per_class": metrics.get("per_class"),
                                             "conf_threshold": metrics["conf_threshold"], "iou_threshold": metrics["iou_threshold"]}})
        return metrics

    background.add_task(run_job, job, _run, task=task, cfg=cfg)
    return {"job_id": job.job_id}


@router.get("/annotated-result/{task}")
def get_annotated_result(task: str):
    """Last saved annotated-eval result (metrics + per-image boxes) so the page
    auto-loads the overlay without re-running — like OCR/VLM."""
    import json as _json
    path = os.path.join(DETECTION_DIR, "results", f"annotated_{task}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return _json.load(f)


@router.post("/detail/{task}")
async def eval_annotated_detail(task: str, background: BackgroundTasks, cfg: EvalConfig = EvalConfig()):
    """Like eval-annotated, but returns per-image PREDICTED + GROUND-TRUTH boxes
    so the UI can overlay them on the drawing — to eyeball box quality and which
    annotated regions were missed."""
    job = create_job(f"Detection detail · {task}")

    def _run(job, task, cfg):
        import glob, json
        import numpy as np
        from PIL import Image

        target  = "dimension" if task == "dimensions" else "table"
        masters = sorted(glob.glob(os.path.join(UNIFIED_DIR, "*.json")))
        if not masters:
            raise RuntimeError("No annotated images found (dataset/master/unified is empty).")

        gt_by_image = {}
        for p in masters:
            rec  = json.load(open(p))
            if rec.get("status") == "draft":
                continue   # unfinished annotation — not part of evaluation
            stem = rec.get("image") or os.path.splitext(os.path.basename(p))[0]
            gt_by_image[stem] = [[r["bbox"][0], r["bbox"][1], r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3]]
                                 for r in rec.get("regions", []) if r.get("class") == target and r.get("bbox")]

        weights = cfg.weights_path or find_best_weights(task)
        if not weights or not os.path.exists(weights):
            raise RuntimeError("No trained weights found for this task.")
        job.log(f"Model: {weights}")
        model = _load_detector(weights)

        def _find_img(stem):
            for f in os.listdir(SELECTED_DIR):
                if os.path.splitext(f)[0] == stem and f.lower().endswith((".jpg", ".jpeg", ".png")):
                    return os.path.join(SELECTED_DIR, f)
            return None

        imgsz  = cfg.imgsz or 640
        images = []
        for stem, gt in gt_by_image.items():
            ip = _find_img(stem)
            if not ip:
                continue
            im = Image.open(ip).convert("RGB")
            w, h = im.size
            out = model.predict([np.array(im)], conf_threshold=0.001, imgsz=imgsz)[0]
            boxes, scores = out["boxes"].tolist(), out["scores"].tolist()
            preds = sorted(
                [{"box": [round(v, 1) for v in b], "score": round(s, 4)}
                 for b, s in zip(boxes, scores) if s >= 0.05],
                key=lambda d: d["score"], reverse=True)[:120]
            images.append({"image": stem, "width": w, "height": h,
                           "gt": gt, "pred": preds, "n_pred_total": len(boxes)})
            job.log(f"{stem}: {len(boxes)} raw preds ({len(preds)} shown) vs {len(gt)} annotated {target}(s)")

        return {"task": task, "target": target,
                "weights": os.path.basename(os.path.dirname(os.path.dirname(weights))),
                "images": images}

    background.add_task(run_job, job, _run, task=task, cfg=cfg)
    return {"job_id": job.job_id}