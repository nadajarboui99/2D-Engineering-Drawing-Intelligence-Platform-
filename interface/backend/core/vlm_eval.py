"""
VLM feature-extraction evaluation against ground-truth feature values.

Ground truth lives in ONE file per task:

    vlm/data/ground_truth/<task>.json

keyed by image stem, each value being the correct feature values (a subset of
the fields in vlm/configs/feature_schema.yaml):

    {
      "drawing_001": { "length": 120.0, "width": 60.0, "holes_count": 4,
                       "material": "Al 6061", "standard": "ISO 2768" },
      "drawing_002": { "length": 88.5, "holes_count": 2 }
    }

The answer key stores the FULL schema per image, with null meaning "truly absent
in this drawing". That lets each of the 15 fields fall into one outcome:
  present GT -> correct | wrong | missed (returned null)
  absent  GT -> hallucinated (invented a value) | abstained (correctly null)

Metrics per (mode):
  - field_accuracy     : correct / fields-that-should-have-a-value
  - error_rate         : wrong  / fields-that-should-have-a-value
  - miss_rate          : missed / fields-that-should-have-a-value
  - hallucination_rate : hallucinated / fields-that-should-be-empty
  - overall_accuracy   : (correct + abstained) / all 15 fields
  - exact_match        : images with zero wrong/missed/hallucinated
Numbers are compared with a relative tolerance; strings are normalized.
"""
import os
import json

VLM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "vlm")
)
GT_DIR = os.path.join(VLM_DIR, "data", "ground_truth")

NUMERIC_REL_TOL = 0.02   # 2% relative tolerance for measurements
NUMERIC_ABS_TOL = 0.5    # ...or within 0.5 absolute (whichever is looser)


def _norm_str(s) -> str:
    return " ".join(str(s).lower().split())


def _to_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", ".").strip())
    except (ValueError, AttributeError):
        return None


def _scalar_correct(pred, gt) -> bool:
    if pred is None:
        return False
    gt_num = _to_number(gt)
    if gt_num is not None:
        pred_num = _to_number(pred)
        if pred_num is None:
            return False
        return abs(pred_num - gt_num) <= max(NUMERIC_ABS_TOL, NUMERIC_REL_TOL * abs(gt_num))
    return _norm_str(pred) == _norm_str(gt)


def _as_list(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [p.strip() for p in v.split(",") if p.strip()]
    return [v]


def _field_correct(pred, gt) -> bool:
    # List fields (e.g. circular_hole_diameters): order-independent multiset match.
    if isinstance(gt, list):
        pred_list = _as_list(pred)
        if pred_list is None or len(pred_list) != len(gt):
            return False
        remaining = list(pred_list)
        for g in gt:
            idx = next((i for i, p in enumerate(remaining) if _scalar_correct(p, g)), None)
            if idx is None:
                return False
            remaining.pop(idx)
        return True
    return _scalar_correct(pred, gt)


def _load_gt_path(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        out = {}
        for row in data:
            stem = row.get("image") or row.get("image_id") or row.get("id")
            if stem is not None:
                feats = {k: v for k, v in row.items()
                         if k not in ("image", "image_id", "id")}
                out[str(stem)] = feats
        return out
    return dict(data)


def load_gt(task: str) -> dict:
    """Returns {image_stem: {field: value}} or {} if no ground truth exists."""
    return _load_gt_path(os.path.join(GT_DIR, f"{task}.json"))


def load_unified_gt() -> dict:
    """The VLM extracts ONE unified feature set per drawing, so scoring uses a
    single task-independent answer key. Falls back to the legacy per-task files
    (which were identical copies) if unified.json isn't there yet."""
    gt = _load_gt_path(os.path.join(GT_DIR, "unified.json"))
    if gt:
        return gt
    return load_gt("tables") or load_gt("dimensions")


def _pred_present(pred) -> bool:
    return pred is not None and pred != "" and pred != []


def _classify(pred, gt_val) -> str:
    """One of: correct | wrong | missed | hallucinated | abstained."""
    if gt_val is not None:
        if _field_correct(pred, gt_val):
            return "correct"
        return "missed" if not _pred_present(pred) else "wrong"
    return "hallucinated" if _pred_present(pred) else "abstained"


def image_detail(extracted: dict, gt_feats: dict) -> dict:
    """Per-field verdict table + per-image metrics for one image (extracted vs GT)."""
    fields = []
    counts = {"correct": 0, "wrong": 0, "missed": 0, "hallucinated": 0, "abstained": 0}
    present = absent = 0
    ape_sum = 0.0
    ape_count = 0
    for field, gt_val in gt_feats.items():
        pred = extracted.get(field)
        verdict = _classify(pred, gt_val)
        counts[verdict] += 1
        if gt_val is not None:
            present += 1
            gt_num = _to_number(gt_val)
            pred_num = _to_number(pred) if _pred_present(pred) else None
            if gt_num is not None and pred_num is not None:
                ape_sum += min(abs(pred_num - gt_num) / max(abs(gt_num), 1e-9), 2.0)
                ape_count += 1
        else:
            absent += 1
        fields.append({"name": field, "gt": gt_val, "pred": pred, "verdict": verdict})

    metrics = {
        "field_accuracy":     round(counts["correct"] / present, 4) if present else None,
        "error_rate":         round(counts["wrong"] / present, 4) if present else None,
        "miss_rate":          round(counts["missed"] / present, 4) if present else None,
        "hallucination_rate": round(counts["hallucinated"] / absent, 4) if absent else None,
        "numeric_mape":       round(ape_sum / ape_count, 4) if ape_count else None,
        "counts": counts, "present_fields": present, "absent_fields": absent,
    }
    return {"fields": fields, "metrics": metrics}


def detail(results, gt: dict = None) -> dict:
    """Per-image, per-field breakdown for every mode — powers the image inspector.
    Returns {mode: [ {image, fields:[{name,gt,pred,verdict}], metrics} ]}."""
    if gt is None:
        gt = load_unified_gt()
    out = {"whole_image": [], "whole_image_ocr": [], "cropped_ocr": []}
    if not gt:
        return out
    for mode in out:
        seen = set()
        for r in results:
            if r.get("mode") != mode or (r.get("extracted") or {}).get("error"):
                continue
            stem = r["image"]
            if stem in seen or stem not in gt or not gt[stem]:
                continue
            seen.add(stem)
            d = image_detail(r.get("extracted") or {}, gt[stem])
            out[mode].append({"image": stem, **d})
    return out


def evaluate(results, mode: str = None, gt: dict = None) -> dict:
    """
    results: list of {"image", "task", "mode", "extracted": {...}} (the VLM run output)
    Scores against the unified per-drawing answer key. `mode` filters which results
    are scored; `available` is False when no GT exists.
    """
    if gt is None:
        gt = load_unified_gt()
    if not gt:
        return {"available": False, "mode": mode}

    rows = [r for r in results if not (r.get("extracted") or {}).get("error")]
    if mode:
        rows = [r for r in rows if r.get("mode") == mode]

    correct = wrong = missed = hallucinated = abstained = 0
    present_total = absent_total = 0
    images = 0
    exact_images = 0
    per_field_totals = {}   # field -> [correct, present_total]
    seen = set()            # one score per image
    ape_sum = 0.0           # numeric closeness: sum of |pred-gt|/|gt| over numeric attempts
    ape_count = 0

    for r in rows:
        stem = r["image"]
        if stem in seen or stem not in gt:
            continue
        gt_feats = gt[stem]
        if not gt_feats:
            continue
        seen.add(stem)
        images += 1
        extracted = r.get("extracted") or {}
        image_perfect = True

        for field, gt_val in gt_feats.items():
            pred = extracted.get(field)
            pred_present = pred is not None and pred != "" and pred != []

            if gt_val is not None:
                present_total += 1
                slot = per_field_totals.setdefault(field, [0, 0])
                slot[1] += 1

                gt_num, pred_num = _to_number(gt_val), (_to_number(pred) if pred_present else None)
                if gt_num is not None and pred_num is not None:
                    ape_sum += min(abs(pred_num - gt_num) / max(abs(gt_num), 1e-9), 2.0)
                    ape_count += 1

                if _field_correct(pred, gt_val):
                    correct += 1
                    slot[0] += 1
                elif not pred_present:
                    missed += 1
                    image_perfect = False
                else:
                    wrong += 1
                    image_perfect = False
            else:
                absent_total += 1
                if pred_present:
                    hallucinated += 1
                    image_perfect = False
                else:
                    abstained += 1

        if image_perfect:
            exact_images += 1

    if images == 0:
        return {"available": True, "mode": mode, "evaluated_images": 0, "gt_images": len(gt),
                "note": "Ground truth found, but no VLM results match its image names."}

    total_fields = present_total + absent_total
    return {
        "available": True,
        "mode": mode,
        "evaluated_images": images,
        "gt_images": len(gt),
        "field_accuracy":     round(correct / present_total, 4) if present_total else 0.0,
        "overall_accuracy":   round((correct + abstained) / total_fields, 4) if total_fields else 0.0,
        "error_rate":         round(wrong / present_total, 4) if present_total else 0.0,
        "miss_rate":          round(missed / present_total, 4) if present_total else 0.0,
        "hallucination_rate": round(hallucinated / absent_total, 4) if absent_total else 0.0,
        "numeric_mape":       round(ape_sum / ape_count, 4) if ape_count else None,
        "numeric_scored":     ape_count,
        "exact_match":        round(exact_images / images, 4),
        "counts": {
            "correct": correct, "wrong": wrong, "missed": missed,
            "hallucinated": hallucinated, "abstained": abstained,
            "present_fields": present_total, "absent_fields": absent_total,
        },
        "per_field": {k: round(v[0] / v[1], 4) if v[1] else 0.0 for k, v in per_field_totals.items()},
        "scored_fields": total_fields,
    }
