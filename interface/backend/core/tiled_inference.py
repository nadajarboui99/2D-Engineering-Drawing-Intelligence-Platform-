"""
Sliced (tiled) inference — the SAHI technique, implemented locally so it works
with any detector wrapper that exposes:

    model.predict([np_image, ...], conf_threshold=float, imgsz=int)
        -> [ {"boxes": Tensor[N,4] xyxy, "scores": Tensor[N], "labels": ...}, ... ]

Why this exists
---------------
Detectors resize the WHOLE image to a fixed square (imgsz) before inference.
On a 7000px scan, a 21px dimension line shrinks to a few pixels and vanishes.
Instead of feeding one shrunk image, we cut the drawing into overlapping tiles,
run the detector on each tile at (near) native resolution, map the boxes back to
full-image coordinates, and merge duplicates with NMS. Tiny objects are now seen
at full size; big objects (tables) are recovered by an extra full-image pass.
"""

import numpy as np


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(boxes, iou_thr=0.5):
    """boxes: [(x1,y1,x2,y2,score)]. Greedy NMS, highest score wins."""
    order = sorted(boxes, key=lambda b: -b[4])
    kept = []
    for b in order:
        if all(_iou(b, k) < iou_thr for k in kept):
            kept.append(b)
    return kept


def _tile_origins(size, tile, overlap):
    """Start coordinates along one axis so tiles of width `tile` cover `size`
    with the given fractional overlap. Always includes an edge-aligned last tile."""
    if size <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    origins = list(range(0, size - tile + 1, step))
    if origins[-1] != size - tile:
        origins.append(size - tile)
    return origins


def sliced_predict(model, pil_image, conf_threshold=0.001, tile=1024,
                   overlap=0.2, imgsz=None, merge_iou=0.5, add_full_image=True):
    """
    Run tiled inference on one PIL image.

    Returns [(x1,y1,x2,y2,score)] in FULL-image coordinates, after NMS.

    tile        edge length of each square crop (native pixels)
    overlap     fraction shared between neighbouring tiles (so objects on a
                seam land fully inside at least one tile)
    imgsz       inference size per tile; defaults to `tile` (no downscaling)
    add_full_image  also run one whole-image pass and merge it in, to recover
                objects larger than a tile (e.g. tables spanning the sheet)
    """
    arr = np.array(pil_image)
    H, W = arr.shape[:2]
    imgsz = imgsz or tile

    crops, offsets = [], []
    for y0 in _tile_origins(H, tile, overlap):
        for x0 in _tile_origins(W, tile, overlap):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            crops.append(arr[y0:y1, x0:x1])
            offsets.append((x0, y0))

    all_boxes = []
    # Batch the tiles through the wrapper; fall back to one-by-one if a wrapper
    # can't take a list of differently-sized crops.
    try:
        outs = model.predict(crops, conf_threshold=conf_threshold, imgsz=imgsz)
    except Exception:
        outs = [model.predict([c], conf_threshold=conf_threshold, imgsz=imgsz)[0] for c in crops]

    for out, (ox, oy) in zip(outs, offsets):
        boxes = out["boxes"].tolist()
        scores = out["scores"].tolist()
        for b, s in zip(boxes, scores):
            all_boxes.append((b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, s))

    if add_full_image:
        fo = model.predict([arr], conf_threshold=conf_threshold, imgsz=max(imgsz, 1024))[0]
        for b, s in zip(fo["boxes"].tolist(), fo["scores"].tolist()):
            all_boxes.append((b[0], b[1], b[2], b[3], s))

    return _nms(all_boxes, iou_thr=merge_iou)
