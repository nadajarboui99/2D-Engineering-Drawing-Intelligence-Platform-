"""
Self-contained detection evaluation: compares predicted boxes to ground-truth
boxes (single class) and returns mAP@0.5, precision, recall, F1.

Used to evaluate a trained model on the *annotated* dataset (the boxes you drew
in dataset/master/unified/), independently of the training-time results.csv.
"""


def _iou(a, b):
    # boxes in xyxy
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match(preds, gts, iou_thr):
    """preds: [(x1,y1,x2,y2,score)] for ONE image (any order). gts: [(x1,y1,x2,y2)].
    Returns list of (score, is_tp) in the given pred order, plus matched-GT count."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][4])
    used = [False] * len(gts)
    out = [None] * len(preds)
    for i in order:
        p = preds[i]
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = _iou(p[:4], g)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
            out[i] = (p[4], 1)
        else:
            out[i] = (p[4], 0)
    return out, sum(used)


def evaluate(preds_by_image, gt_by_image, iou_thr=0.5, conf=0.25):
    """
    preds_by_image: {image: [(x1,y1,x2,y2,score), ...]}
    gt_by_image:    {image: [(x1,y1,x2,y2), ...]}   (only images with GT are scored)
    """
    n_gt = sum(len(v) for v in gt_by_image.values())
    if n_gt == 0:
        return {"available": False, "note": "No ground-truth boxes for this class."}

    # ---- AP@iou (all predictions, all scores) ----
    scored = []          # (score, is_tp) across all images
    for img, gts in gt_by_image.items():
        preds = preds_by_image.get(img, [])
        marks, _ = _match(preds, gts, iou_thr)
        scored.extend(marks)
    scored.sort(key=lambda x: -x[0])

    cum_tp = cum_fp = 0
    rec_pts, pre_pts = [], []
    for _, is_tp in scored:
        cum_tp += is_tp
        cum_fp += (1 - is_tp)
        rec_pts.append(cum_tp / n_gt)
        pre_pts.append(cum_tp / (cum_tp + cum_fp))

    # all-point interpolation (VOC style)
    mrec = [0.0] + rec_pts + [rec_pts[-1] if rec_pts else 0.0]
    mpre = [0.0] + pre_pts + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mrec) - 1))

    # ---- headline precision/recall/F1 at a confidence threshold ----
    tp = fp = matched_gt = 0
    for img, gts in gt_by_image.items():
        preds = [p for p in preds_by_image.get(img, []) if p[4] >= conf]
        marks, mg = _match(preds, gts, iou_thr)
        tp += sum(m[1] for m in marks)
        fp += sum(1 - m[1] for m in marks)
        matched_gt += mg
    fn = n_gt - matched_gt
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "available": True,
        "map50": round(ap, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "n_gt": n_gt,
        "n_pred": sum(len(v) for v in preds_by_image.values()),
        "conf_threshold": conf,
        "iou_threshold": iou_thr,
    }
