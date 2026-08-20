"""
OCR evaluation against ground-truth text.

Ground truth lives in one file per task:

    ocr/data/ground_truth/<task>.json

keyed by image stem (the file name without extension), each value being the
list of true text strings present in that image:

    {
      "drawing_001": ["45.5", "M6", "R10", "ISO 2768"],
      "drawing_002": ["120", "Ø8", "Steel 1045"]
    }

The SAME ground truth is used for both OCR modes (whole image and detection
crops): the answer key is "what text is truly in the image", which does not
depend on how the text was extracted. Both modes yield a list of predicted
strings, so the metric is identical.

Metrics (corpus-level):
  - CER  : total char edit-distance / total ground-truth chars   (lower better)
  - WER  : total word edit-distance / total ground-truth words   (lower better)
  - exact_match : fraction of GT strings matched exactly (normalized)
  - recall / precision / f1 : string-level, a GT string counts as "found"
    when its best predicted match has CER <= MATCH_CER_THRESHOLD.
"""
import os
import json
from collections import Counter

OCR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ocr")
)
GT_DIR = os.path.join(OCR_DIR, "data", "ground_truth")

MATCH_CER_THRESHOLD = 0.5  # a GT string is "found" if best match is <=50% char error


def _norm(s: str) -> str:
    """Normalize a string for comparison: lowercase, collapse whitespace."""
    return " ".join(str(s).lower().split())


def _edit_distance(a, b) -> int:
    """Levenshtein distance over any two sequences (strings or token lists)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _load_gt_file(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    # Accept either {stem: [...]} or [{"image": stem, "texts": [...]}, ...]
    if isinstance(data, list):
        out = {}
        for row in data:
            stem = row.get("image") or row.get("image_id") or row.get("id")
            if stem is not None:
                out[str(stem)] = row.get("texts", row.get("text", []))
        return out
    return {k: (v if isinstance(v, list) else [v]) for k, v in data.items()}


def load_gt(task: str) -> dict:
    """Per-class ground truth (crop mode): the texts you annotated for that task."""
    return _load_gt_file(os.path.join(GT_DIR, f"{task}.json"))


def load_whole_image_gt() -> dict:
    """Union of ALL annotated texts per image (whole-image mode)."""
    return _load_gt_file(os.path.join(GT_DIR, "_whole_image.json"))


def _score_image(pred_texts, gt_texts) -> dict:
    """Greedily match each GT string to its best remaining predicted string."""
    preds = [_norm(t) for t in pred_texts if str(t).strip()]
    gts   = [_norm(t) for t in gt_texts if str(t).strip()]

    acc = {
        "char_edits": 0, "gt_chars": 0,
        "word_edits": 0, "gt_words": 0,
        "exact": 0, "found": 0,
        "n_gt": len(gts), "n_pred": len(preds),
    }
    used = [False] * len(preds)
    matched_pred_idxs = set()

    for gt in gts:
        best_i, best_d = -1, None
        for i, p in enumerate(preds):
            if used[i]:
                continue
            d = _edit_distance(gt, p)
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        best = preds[best_i] if best_i >= 0 else ""
        if best_i >= 0:
            used[best_i] = True

        acc["char_edits"] += min(_edit_distance(gt, best), len(gt))
        acc["gt_chars"]   += max(1, len(gt))
        acc["word_edits"] += min(_edit_distance(gt.split(), best.split()), len(gt.split()))
        acc["gt_words"]   += max(1, len(gt.split()))
        if best == gt:
            acc["exact"] += 1
        if len(gt) and (_edit_distance(gt, best) / len(gt)) <= MATCH_CER_THRESHOLD:
            acc["found"] += 1
            if best_i >= 0:
                matched_pred_idxs.add(best_i)

    acc["matched_pred"] = len(matched_pred_idxs)
    return acc


def _tokens(text) -> list:
    """Normalize text into comparable word tokens (lowercase, comma→dot, strip
    edge punctuation). Keeps '45.5' / 'M6' whole; folds '90,68' → '90.68'."""
    out = []
    for tok in str(text).split():
        t = tok.strip().lower().replace(",", ".").strip(".,;:()[]{}\"'`°|")
        if t:
            out.append(t)
    return out


def _lcs_len(a: str, b: str) -> int:
    """Longest common subsequence length (character-level containment)."""
    if not a or not b:
        return 0
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def whole_text_detail(pred_texts, gt_texts) -> dict:
    """Whole-text scoring: treat ALL annotated text as one body and ALL OCR output
    as one body, then measure how much of the annotation was captured. Correct for
    table blocks (a paragraph) and whole-image OCR, where per-string matching fails.
    """
    gt_joined   = " ".join(str(t) for t in gt_texts if str(t).strip())
    pred_joined = " ".join(str(t) for t in pred_texts if str(t).strip())
    gt_toks     = _tokens(gt_joined)
    pred_toks   = _tokens(pred_joined)

    # Word coverage: order-independent multiset overlap.
    avail = Counter(pred_toks)
    matched_flags, matched = [], 0
    for w in gt_toks:
        if avail.get(w, 0) > 0:
            avail[w] -= 1
            matched += 1
            matched_flags.append(True)
        else:
            matched_flags.append(False)

    gt_norm, pred_norm = " ".join(gt_toks), " ".join(pred_toks)
    lcs = _lcs_len(gt_norm, pred_norm)
    return {
        "gt_text": gt_joined,
        "pred_text": pred_joined,
        "gt_tokens": gt_toks,
        "matched_flags": matched_flags,
        "metrics": {
            "word_coverage":  round(matched / len(gt_toks), 4) if gt_toks else 0.0,
            "char_coverage":  round(lcs / len(gt_norm), 4) if gt_norm else 0.0,
            "word_precision": round(matched / len(pred_toks), 4) if pred_toks else 0.0,
            "n_gt_words": len(gt_toks), "n_pred_words": len(pred_toks), "matched_words": matched,
            "_lcs": lcs, "_gt_chars": len(gt_norm),
        },
    }


def evaluate_whole(predicted_by_image: dict, gt: dict = None) -> dict:
    """Corpus-level whole-text coverage across all images (block-aware)."""
    if gt is None:
        gt = load_whole_image_gt()
    if not gt:
        return {"available": False}
    tot_gt = tot_matched = tot_pred = lcs_sum = gt_chars = imgs = 0
    for img, gts in gt.items():
        if img not in predicted_by_image:
            continue
        imgs += 1
        m = whole_text_detail(predicted_by_image[img], gts)["metrics"]
        tot_gt += m["n_gt_words"]; tot_matched += m["matched_words"]; tot_pred += m["n_pred_words"]
        lcs_sum += m["_lcs"]; gt_chars += m["_gt_chars"]
    if imgs == 0:
        return {"available": True, "evaluated_images": 0, "gt_images": len(gt)}
    # Also compute the classic string-level CER / WER / exact-match (per annotated
    # string, best-matched) so the accuracy of what was read is reported too.
    strv = evaluate(predicted_by_image, "all", gt=gt)
    return {
        "available": True, "evaluated_images": imgs, "gt_images": len(gt),
        "word_coverage":  round(tot_matched / tot_gt, 4) if tot_gt else 0.0,
        "char_coverage":  round(lcs_sum / gt_chars, 4) if gt_chars else 0.0,
        "word_precision": round(tot_matched / tot_pred, 4) if tot_pred else 0.0,
        "cer": strv.get("cer"), "wer": strv.get("wer"), "exact_match": strv.get("exact_match"),
        "n_gt_words": tot_gt, "n_pred_words": tot_pred, "matched_words": tot_matched,
    }


def image_detail(pred_texts, gt_texts) -> dict:
    """Per-string breakdown for one image: which annotated texts were found vs
    missed, each with its best OCR match + CER, plus extra (unmatched) predictions."""
    preds       = [str(t) for t in pred_texts if str(t).strip()]
    preds_norm  = [_norm(t) for t in preds]
    gts         = [str(t) for t in gt_texts if str(t).strip()]
    used        = [False] * len(preds_norm)
    matched_idx = set()

    gt_rows = []
    char_edits = gt_chars = found = 0
    for gt in gts:
        gtn = _norm(gt)
        best_i, best_d = -1, None
        for i, p in enumerate(preds_norm):
            if used[i]:
                continue
            d = _edit_distance(gtn, p)
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        bestn = preds_norm[best_i] if best_i >= 0 else ""
        best  = preds[best_i] if best_i >= 0 else ""
        if best_i >= 0:
            used[best_i] = True
        dist = _edit_distance(gtn, bestn)
        cer  = min(dist, len(gtn)) / max(1, len(gtn))
        is_found = len(gtn) > 0 and (dist / len(gtn)) <= MATCH_CER_THRESHOLD
        if is_found:
            found += 1
            if best_i >= 0:
                matched_idx.add(best_i)
        char_edits += min(dist, len(gtn))
        gt_chars   += max(1, len(gtn))
        gt_rows.append({"text": gt, "match": best, "found": is_found, "cer": round(cer, 3)})

    pred_rows = [{"text": preds[i], "matched": i in matched_idx} for i in range(len(preds))]
    n_gt, n_pred = len(gts), len(preds)
    metrics = {
        "recall":    round(found / n_gt, 4) if n_gt else 0.0,
        "precision": round(len(matched_idx) / n_pred, 4) if n_pred else 0.0,
        "cer":       round(char_edits / gt_chars, 4) if gt_chars else 0.0,
        "n_gt": n_gt, "n_pred": n_pred, "found": found,
    }
    return {"gt": gt_rows, "pred": pred_rows, "metrics": metrics}


def evaluate(predicted_by_image: dict, task: str, gt: dict = None) -> dict:
    """
    predicted_by_image: {image_stem: [predicted_texts]}
    gt: optional ground-truth override (e.g. the whole-image union); defaults to
    the per-class GT for `task`. `available` is False when no ground truth exists.
    """
    if gt is None:
        gt = load_gt(task)
    if not gt:
        return {"available": False, "task": task}

    total = {"char_edits": 0, "gt_chars": 0, "word_edits": 0, "gt_words": 0,
             "exact": 0, "found": 0, "n_gt": 0, "n_pred": 0, "matched_pred": 0}
    evaluated_images = 0

    for stem, gt_texts in gt.items():
        if stem not in predicted_by_image:
            continue
        evaluated_images += 1
        s = _score_image(predicted_by_image[stem], gt_texts)
        for k in total:
            total[k] += s[k]

    if evaluated_images == 0:
        return {"available": True, "task": task, "evaluated_images": 0,
                "note": "Ground truth found, but no predicted OCR results match its image names."}

    cer = total["char_edits"] / total["gt_chars"] if total["gt_chars"] else 0.0
    wer = total["word_edits"] / total["gt_words"] if total["gt_words"] else 0.0
    exact = total["exact"] / total["n_gt"] if total["n_gt"] else 0.0
    recall = total["found"] / total["n_gt"] if total["n_gt"] else 0.0
    precision = total["matched_pred"] / total["n_pred"] if total["n_pred"] else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "available": True,
        "task": task,
        "evaluated_images": evaluated_images,
        "gt_images": len(gt),
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "exact_match": round(exact, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_gt_strings": total["n_gt"],
        "n_pred_strings": total["n_pred"],
    }
