"""
Shared evaluation metrics for object detection.

Computes per-class and overall:
  - Precision, Recall, F1
  - mAP@0.5

Works for both table detection and dimension detection scripts.
"""

import torch
import numpy as np
from collections import defaultdict


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    boxes: [N, 4] in [x1, y1, x2, y2] format
    Returns: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / union_area.clamp(min=1e-6)


class DetectionMetrics:
    """
    Accumulates predictions and ground truths across batches,
    then computes Precision, Recall, F1, and mAP@0.5.

    Usage:
        metrics = DetectionMetrics(iou_threshold=0.5)
        for batch in dataloader:
            preds = model(batch)
            metrics.update(preds, targets)
        results = metrics.compute()
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.all_preds = []   # list of dicts: {boxes, scores, labels}
        self.all_targets = [] # list of dicts: {boxes, labels}

    def update(self, predictions: list, targets: list):
        """
        predictions: list of dicts with keys 'boxes' [N,4], 'scores' [N], 'labels' [N]
        targets:     list of dicts with keys 'boxes' [M,4], 'labels' [M]
        """
        self.all_preds.extend(predictions)
        self.all_targets.extend(targets)

    def compute(self) -> dict:
        all_classes = set()
        for t in self.all_targets:
            all_classes.update(t["labels"].tolist())

        per_class_results = {}
        aps = []

        for cls_id in sorted(all_classes):
            tp_list, fp_list, scores_list = [], [], []
            n_gt = 0

            for pred, target in zip(self.all_preds, self.all_targets):
                gt_mask = target["labels"] == cls_id
                gt_boxes = target["boxes"][gt_mask]
                n_gt += gt_boxes.shape[0]

                pred_mask = pred["labels"] == cls_id
                pred_boxes = pred["boxes"][pred_mask]
                pred_scores = pred["scores"][pred_mask]

                if pred_boxes.shape[0] == 0:
                    continue

                order = pred_scores.argsort(descending=True)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]

                matched_gt = set()
                for pb, ps in zip(pred_boxes, pred_scores):
                    scores_list.append(ps.item())
                    if gt_boxes.shape[0] > 0:
                        ious = box_iou(pb.unsqueeze(0), gt_boxes)[0]
                        best_iou, best_gt = ious.max(0)
                        if best_iou >= self.iou_threshold and best_gt.item() not in matched_gt:
                            tp_list.append(1)
                            fp_list.append(0)
                            matched_gt.add(best_gt.item())
                        else:
                            tp_list.append(0)
                            fp_list.append(1)
                    else:
                        tp_list.append(0)
                        fp_list.append(1)

            if not scores_list:
                per_class_results[cls_id] = {"precision": 0, "recall": 0, "f1": 0, "ap": 0}
                aps.append(0)
                continue

            order = np.argsort(scores_list)[::-1]
            tp_arr = np.array(tp_list)[order]
            fp_arr = np.array(fp_list)[order]

            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(fp_arr)

            precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
            recalls    = tp_cum / (n_gt + 1e-6)

            # AP via 11-point interpolation
            ap = 0
            for t in np.linspace(0, 1, 11):
                mask = recalls >= t
                ap += precisions[mask].max() if mask.any() else 0
            ap /= 11

            # Final precision / recall at last point
            final_precision = precisions[-1] if len(precisions) else 0
            final_recall    = recalls[-1]    if len(recalls)    else 0
            f1 = (2 * final_precision * final_recall /
                  (final_precision + final_recall + 1e-6))

            per_class_results[cls_id] = {
                "precision": round(float(final_precision), 4),
                "recall":    round(float(final_recall),    4),
                "f1":        round(float(f1),              4),
                "ap":        round(float(ap),              4),
            }
            aps.append(ap)

        map50 = float(np.mean(aps)) if aps else 0.0

        return {
            "mAP@0.5":   round(map50, 4),
            "per_class": per_class_results,
        }

    def print_report(self, results: dict, category_names: dict = None):
        print("\n" + "=" * 50)
        print(f"  mAP@0.5 : {results['mAP@0.5']:.4f}")
        print("=" * 50)
        print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP@0.5':>10}")
        print("-" * 50)
        for cls_id, r in results["per_class"].items():
            name = category_names.get(cls_id, str(cls_id)) if category_names else str(cls_id)
            print(f"{name:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                  f"{r['f1']:>10.4f} {r['ap']:>10.4f}")
        print("=" * 50 + "\n")