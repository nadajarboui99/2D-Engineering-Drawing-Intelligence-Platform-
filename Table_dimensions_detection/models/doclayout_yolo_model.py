"""
DocLayout-YOLO — a YOLOv10 detector trained on diverse document layouts
(DocStructBench). Unlike Table Transformer (trained on PDF/paper tables), it
learns tables as page REGIONS alongside titles, figures, text — which is much
closer to how a title-block / BOM sits on an engineering drawing.

Zero-shot for us (no training on your data). We keep only the 'table' class.

Install: pip install doclayout-yolo huggingface_hub
Weights download from HuggingFace on first use (cached).

Implements the shared interface:
    predict(images, conf_threshold, imgsz) -> [{"boxes","scores","labels"}]
"""
import numpy as np

_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"


class DocLayoutYOLODetector:
    def __init__(self, weights: str = None):
        from doclayout_yolo import YOLOv10
        if not weights:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(repo_id=_REPO, filename=_FILE)
        print(f"[DocLayout-YOLO] Loading {weights} …")
        self.model = YOLOv10(weights)
        # find the class id whose name is exactly 'table'
        names = self.model.names
        self.table_ids = {int(i) for i, n in names.items() if str(n).strip().lower() == "table"}

    def predict(self, images, conf_threshold: float = 0.25, imgsz: int = 1024):
        from PIL import Image
        out = []
        for img in images:
            arr = np.array(img.convert("RGB")) if isinstance(img, Image.Image) else np.asarray(img)
            res = self.model.predict(arr, imgsz=imgsz, conf=conf_threshold, verbose=False)[0]
            boxes, scores = [], []
            for b in res.boxes:
                cls_id = int(b.cls.item())
                if cls_id in self.table_ids:
                    boxes.append(b.xyxy.cpu().numpy().reshape(-1)[:4])
                    scores.append(float(b.conf.item()))
            out.append({
                "boxes":  np.array(boxes, dtype="float32").reshape(-1, 4),
                "scores": np.array(scores, dtype="float32"),
                "labels": np.zeros(len(boxes), dtype=int),
            })
        return out
