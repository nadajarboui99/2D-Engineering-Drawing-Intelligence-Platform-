# Technical Guide — Engineering-Drawing Evaluation Platform
*Written for the graduation defense and report. Explains the architecture, every
important file, the key code you must be able to defend, and which models were
used, why, and how they were implemented.*

---

## 1. What the system is (one paragraph)

A benchmarking platform that evaluates a **three-stage pipeline for digitizing 2D
mechanical engineering drawings** — **Detection → OCR → VLM** — against a
**hand-annotated held-out ground truth**. The guiding principle is
**generalization over peak scores**: we prefer generic / zero-shot models that
work on *any* mechanical drawing, not models overfit to our few sheets. It is a
**FastAPI (Python) backend** + **React (Vite) frontend**; every model plugs in
through a **registry + wrapper** pattern so architectures can be swapped without
touching the evaluation code.

```
     Annotation (you draw boxes + type the text + feature values)
                              │
                dataset/  →  one annotation becomes several "answer keys"
                              │
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │  DETECTION   │→ │     OCR      │→ │     VLM      │
   │ find boxes   │   │ read the text│   │ extract values│
   └──────────────┘   └──────────────┘   └──────────────┘
          │                  │                  │
          └──────── each stage scored, logged, shown in the web UI ────────┘
```

The three stages are **independent and independently scored** — a deliberate
design choice so each model’s contribution isn’t double-counted (this is why,
for example, a VLM is *not* allowed to be tested as an OCR engine).

---

## 2. Repository map

| Folder | Role |
|---|---|
| `interface/backend/` | FastAPI server: routers (HTTP endpoints) + core (logic). |
| `interface/frontend/` | React SPA (Vite + Tailwind) — the dashboard. |
| `Table_dimensions_detection/` | Detection stage: model wrappers + trained YOLO weights (`runs/`). |
| `ocr/` | OCR stage: model wrappers (`ocr/models/`) + ground truth + results. |
| `vlm/` | VLM stage: wrappers + the Node **Mastra** microservice + feature schema. |
| `dataset/` | The ground-truth engine: your annotations → machine answer keys. |

**Backend split — the pattern to explain in the defense:**
- `routers/*.py` = the HTTP layer (validate request → start a job → return job id).
- `core/*.py` = the actual logic (evaluation math, registries, job manager).
- Model wrappers live *next to their stage’s code* (`Table_dimensions_detection/models/`,
  `ocr/models/`, `vlm/models/`), each a thin adapter to a shared interface.

---

## 3. The backend, file by file

### 3.1 `main.py` — the entry point
Creates the FastAPI app, enables CORS for the Vite dev server
(`localhost:5173`), and mounts each router under a prefix
(`/detection`, `/ocr`, `/vlm`, `/jobs`, `/results`, …). Loads API keys into the
environment on startup (`core/api_keys.py`).

### 3.2 `core/job_manager.py` — asynchronous jobs
Evaluations are slow (model loading, tiling, many crops), so they never run
inside the HTTP request. Instead:
- `create_job(name)` → returns a `Job` with a UUID, a `logs` list, a `status`.
- `run_job(job, fn, **kwargs)` → runs `fn` in a **thread pool executor**
  (`loop.run_in_executor`) so the event loop isn’t blocked; captures the return
  value into `job.result` and flips `status` to `done`/`failed`.
- The frontend **polls `GET /jobs/{id}`** to stream live logs and pick up the
  result. Jobs are kept in an in-memory dict.

**Defense point:** this is a classic *async job + polling* pattern — the request
returns instantly with a `job_id`, the heavy work runs in the background, the UI
polls for progress. No websockets needed.

### 3.3 The registry + wrapper pattern (the architectural core)
This is the single most important idea to understand.

**Every model is a thin wrapper class implementing one method**, and a
**registry entry** tells the backend how to load it. Adding a model = *one
wrapper file + one registry line*, with **no change to the evaluation code**.

Shared interfaces (the “contract”):
- **Detection** — `predict(images, conf_threshold, imgsz) -> [{"boxes","scores","labels"}]`
  (see `Table_dimensions_detection/models/yolov11.py`).
- **OCR** — `read(image: PIL.Image) -> str` (see `ocr/models/base.py`, `BaseOCR`).
- **VLM** — `extract(image, …)` (see `vlm/models/base.py`, `BaseVLM`).

Registries:
- `core/detection_registry.py` — `BUILTIN_ARCHS` (Table Transformer, Grounding
  DINO, DocLayout-YOLO, …) plus user-added JSON entries. Each entry declares
  `wrapper_module`, `wrapper_class`, `check_import` (deps that must import for the
  model to be “available”), and `install_cmd`.
- `core/ocr_registry.py` — the same idea for OCR (easyocr, tesseract, trocr,
  paddleocr, got-ocr2, doctr, vlm-ocr).

**Dynamic loading — the subtle part worth calling out.** All three stages ship a
top-level package literally named `models` (`Table_dimensions_detection/models`,
`ocr/models`, `vlm/models`). To load the *right* one at runtime, the loaders
(`_load_arch_detector` in `routers/detection.py`, `_load_ocr` in `routers/ocr.py`)
**re-prioritize `sys.path`** and **purge stale `sys.modules`** entries before
`importlib.import_module(...)`. Without this, Python would cache the first
`models` package it saw and the wrong wrapper would load. *This is a real
engineering detail examiners like — it shows you understand Python’s import
system.*

### 3.4 `core/detection_eval.py` — how detection is scored
- `_iou(a, b)` — intersection-over-union of two xyxy boxes.
- `_match(preds, gts, iou_thr)` — greedy matching: sort predictions by
  confidence, each matches the best unused ground-truth box with IoU ≥ threshold;
  otherwise it’s a false positive.
- `evaluate(...)` returns:
  - **mAP@0.5** — average precision via **VOC all-point interpolation** of the
    precision–recall curve.
  - **Precision / Recall / F1** at a chosen confidence threshold.
  - **Best-F1 / Best-conf** — a sweep over the whole PR curve returning the
    highest achievable F1 and the threshold that reaches it. *This is the
    headline metric for model comparison because it’s threshold-independent.*

**Defense point:** IoU threshold 0.5 is the standard PASCAL-VOC operating point;
Best-F1 removes the arbitrary confidence cut so two models are compared fairly.

### 3.5 `core/tiled_inference.py` — the “SAHI” fix
Detectors resize the whole image to a fixed square (`imgsz`) before inference.
On a 7000-px scan, a 21-px dimension line shrinks to a few pixels and vanishes.
`sliced_predict(...)`:
1. Cuts the drawing into **overlapping tiles** (`_tile_origins` computes the grid
   with a fractional overlap so no object is split at a seam).
2. Runs the detector on each tile **at native resolution**.
3. Maps every tile’s boxes back to full-image coordinates.
4. Adds one **full-image pass** to recover objects larger than a tile (tables).
5. Merges duplicates from overlaps with **non-max suppression** (`_nms`).

**Defense point:** this is the published **SAHI** technique (Sliced Aided
Hyper Inference); it is the root-cause fix for tiny objects on large scans, and
it is model-agnostic (works through the same `predict` contract for any detector).

### 3.6 `core/ocr_eval.py` — how OCR is scored
- The evaluation protocol is **gtcrop**: crop each region from the *ground-truth*
  box and OCR it. This isolates **OCR quality from detection error** — the
  cleanest way to compare readers.
- `_edit_distance` — Levenshtein (used for both characters and word tokens).
- `_score_image` — greedily matches each ground-truth string to its best
  predicted string, accumulating character/word edits.
- Corpus metrics: **CER** (char error rate — the headline, lower = better),
  **WER**, **exact-match**, plus string-level precision/recall/F1.
- `whole_text_detail` — for multi-line blocks: order-independent **word
  coverage** (multiset overlap) and **char coverage** (LCS containment), used for
  tables/whole-page where per-string matching is unfair.

**Defense point:** CER is the standard OCR metric; exact-match is too harsh for
multi-line tables, which is why CER is the number we rank on.

### 3.7 `core/vlm_eval.py` — how the VLM stage is scored
The ground truth stores the **full 15-field schema per image**, with `null`
meaning “truly absent”. Every field falls into a **verdict**:
- present GT → **correct** / **wrong** / **missed** (returned null),
- absent GT → **hallucinated** (invented a value) / **abstained** (correctly null).
Metrics: `field_accuracy`, `error_rate`, `miss_rate`, **`hallucination_rate`**,
`overall_accuracy`, `exact_match`. Numbers use a **2 % relative / 0.5 absolute
tolerance**; lists match order-independently.

**Defense point:** keeping `null` in the ground truth is what lets us measure
**hallucination** — a VLM inventing a dimension is a different, worse failure than
missing one, and this taxonomy separates them.

### 3.8 `core/results_store.py` — history + reproducibility
- `log_run(stage, task, model, metrics, extra)` — appends a run to
  `results_store.json` with a filename-safe id.
- `save_snapshot / load_snapshot` — persists a run’s **full visual payload**
  (per-image boxes / text / verdicts) to `run_snapshots/<id>.json` so the Results
  page can reopen the entire dashboard for any past run — reproducible history.

### 3.9 `core/eval_set.py` — draft vs. complete
Annotations can be saved half-finished; `draft_stems()` excludes drafts so
unfinished images never leak into the evaluation set.

---

## 4. The models — what, why, and how (the heart of the defense)

The selection rule is **hypothesis-driven, not brute force**: read what a model
*is* and what it was *trained on*, match it against the input, and test only to
confirm. “Why not test every model?” → because a test without a hypothesis has no
scientific value.

Two properties are matched: **(1) input shape it’s designed for** (single-line vs
page-level/layout) and **(2) training domain + character set**.

### 4.1 Detection

| Model | What / trained on | Result | Verdict |
|---|---|---|---|
| **Trained YOLOv11** (`yolov11.py`) | Ultralytics YOLO fine-tuned on our dimension boxes | **dimensions winner, Best-F1 0.71, precision 0.94** | Domain-fit; nano model, one weak sheet |
| **DocLayout-YOLO** (`doclayout_yolo_model.py`) | YOLOv10 trained on document layouts (DocStructBench), has a native `table` class | **tables winner** | Layout regions ≈ title blocks |
| **Grounding DINO** (`grounding_dino.py`, `grounding_dino_tables.py`) | Open-vocabulary, text-prompted detector | weak (dense tiny objects) | Wrong tool: built for few salient objects |
| **Table Transformer** (`table_transformer.py`) | DETR trained **only** on PubTables-1M (scientific-paper tables) | ~0 on our tables | **Domain mismatch, not a bug** — a title block ≠ a paper table |

**How detection is implemented:** each wrapper exposes `predict(...)`; YOLO/RT-DETR
load a `.pt` file; the HuggingFace models (Table Transformer, Grounding DINO)
lazy-import `transformers`, run their processor + model, and post-process to xyxy
boxes. `max_det` was raised to 1000 (drawings have hundreds of callouts).
Table Transformer’s wrapper filters to the `table` class so `table rotated`
boxes don’t count as false positives.

### 4.2 OCR

| Model | What / trained on | gtcrop CER | Verdict |
|---|---|---|---|
| **TrOCR** (`trocr_model.py`) | Transformer **single-line** recognizer | **dimensions 0.081** | Winner on short callouts; fails tables (single-line) |
| **docTR** (`doctr_model.py`) | DBNet detector + **CRNN** recognizer (Mindee) | **tables 0.261** | Winner on multi-line blocks; fails dimensions (its detector can’t fire on tiny crops) |
| **PaddleOCR** (`paddleocr_model.py`) | PP-OCRv5 detector+recognizer | 0.173 / 0.321 | Best all-rounder, beats no incumbent |
| **EasyOCR / Tesseract / GOT-OCR2** | general OCR engines | mid | Baselines |
| **Surya** (`surya_model.py`, *not registered*) | now **VLM-backed** (llama.cpp) | — | **Excluded** — a VLM belongs to stage 3 |

**How OCR is implemented:** each wrapper implements `read(image) -> str`, lazy-
importing its engine. The eval loops over gt-box crops, calls `read`, and scores
with `ocr_eval`. Final OCR solution = **routing**: TrOCR for dimension crops,
docTR for table crops (no single model wins both — opposite architectures).

### 4.3 VLM (stage built, not yet benchmarked)
`vlm/models/mastra_vlm.py` posts the image + prompt to a local **Mastra** Node
service (`vlm/mastra-service/`) that owns the provider keys and returns text +
token usage. This keeps API keys out of Python and makes the stage
**model-agnostic** (Claude by default; any provider once its `@ai-sdk/*` package
is added). The wrapper resizes images to ≤1568 px and estimates cost from a price
table (for the dashboard only).

---

## 5. The data / ground-truth engine (`dataset/`)
This folder is why anything can be scored.
- `master/unified/*.json` — **the only files you hand-edit**: one per drawing,
  holding boxes, the text in each box, and the 15 feature values.
- `build_from_master.py` — splits each annotation into **several answer keys**:
  detection COCO, OCR-dimensions, OCR-tables, OCR-whole-page, VLM features.
- `normalize.py` — the matching brain: makes `⌀25`, `Ø25`, `DIA 25` equal and
  `25,5` = `25.5`, and splits `⌀25±0.1 mm` into value/tolerance/symbol/unit.

**Defense point:** the same annotation feeds all three stages, so the stages are
measured against one consistent source of truth.

---

## 6. The frontend (`interface/frontend/src/`)
- `App.jsx` — a single-page app; a page-state variable swaps between pages
  (no router library). Pages: **Detection, OCR, VLM, Pipeline, Results, Annotate,
  Settings**.
- `api/client.js` — every backend endpoint in one place, plus `pollJob` (the
  polling loop that drives live logs).
- `pages/*.jsx` — each stage’s page: pick a model, set the protocol, run, and see
  metrics + a per-image visual (SVG box overlays for detection, token chips for
  OCR, a verdict table for VLM).
- `components/runViews.jsx` — turns a saved **snapshot** into its dashboard, so
  the Results page can redraw any past run.
- `components/ui.jsx` — shared widgets (Panel, Select, MetricCard, Btn, …).

**Defense point:** the UI never computes metrics — it only renders what the
backend returns and polls jobs. All logic is server-side and testable.

---

## 7. Likely defense questions (and crisp answers)

- **Why not one model for everything?** Because model design dictates fit: a
  single-line recognizer (TrOCR) physically can’t read a multi-line table; a
  page detector (docTR) can’t fire on a 32-px crop. We proved this by prediction
  then measurement.
- **Why did Table Transformer fail if it was “trained on tables”?** It was
  trained on *scientific-paper* tables (PubTables-1M); a mechanical title block is
  a different visual object. Not a code bug — a domain mismatch (we verified the
  implementation against Microsoft’s repo).
- **Why is the confidence threshold not part of the comparison?** It only trades
  precision for recall; **Best-F1** sweeps all thresholds, so we compare models on
  a threshold-independent number.
- **Why tiling?** Whole-image resize destroys tiny objects on huge scans; SAHI
  tiling runs at native resolution and recovers them.
- **Why evaluate OCR on ground-truth crops (gtcrop)?** To isolate OCR quality
  from detection error — otherwise a good reader is punished for a bad detector.
- **How do you measure hallucination?** The answer key keeps `null` for absent
  features, so inventing a value is a distinct, measurable verdict.
- **How is a new model added?** One wrapper file implementing the stage interface
  + one registry entry; the evaluation code is untouched.

---

## 8. Final results (as benchmarked)

| Stage | Task | Winner | Metric |
|---|---|---|---|
| Detection | Tables | **DocLayout-YOLO** | layout-native |
| Detection | Dimensions | **Trained YOLOv11** | Best-F1 0.71, P 0.94 |
| OCR | Dimensions | **TrOCR** | CER 0.081 |
| OCR | Tables | **docTR** | CER 0.261 |
| VLM | — | Claude via Mastra | stage built, not yet benchmarked |
