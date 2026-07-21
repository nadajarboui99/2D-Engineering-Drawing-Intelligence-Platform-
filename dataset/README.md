# Unified evaluation dataset — single source of truth

**One image set, annotated for BOTH classes (dimension + table), scored by every
stage.** No more separate tables/dimensions image sources — that would confound
task difficulty with dataset difficulty, and the VLM whole-image condition must
read tables and dimensions from the *same* drawing.

```
dataset/
  selected_images/          ← paste your chosen eval images here (or build_master copies them)
  master/unified/           ← one master file per image (BOTH classes) — you annotate these
  derived/                  ← auto-generated: detection COCO, parsed dimensions
  _archive/tables_master/   ← old per-task tables master (kept, not used)
  normalize.py              ← shared text/dimension normalization used by ALL scoring
  build_master.py           ← selected images  → unified master files (pulls dim boxes)
  build_from_master.py      ← unified masters   → per-stage ground truth
  coverage.py               ← annotation progress + N per metric
```

## Workflow

### 1. Select the evaluation images
Either paste image files into `dataset/selected_images/`, **or** pass ids:
```bash
python dataset/build_master.py --ids 24088841_A 26218_D 26172-VDI_B
# or, after pasting into selected_images/:
python dataset/build_master.py
```
This finds each image's **dimension boxes in the detection val.json** and writes
`dataset/master/unified/<image>.json` with the boxes pre-filled (text blank).
Images picked by id are also copied into `selected_images/` so every stage runs
on exactly this set. If an id isn't in the val.json / on disk, it **stops** and
lists the misses instead of guessing.

### 2. Annotate each unified file
Open `dataset/master/unified/<image>.json` and:
- **Dimension regions** (already have box + class): type the `text`, e.g. `"⌀25±0.1"`.
- **Table regions** (not present yet — add them): for each table you see, add a
  region. Get the template with:
  ```bash
  python dataset/build_master.py --help-table
  ```
  ```json
  { "id": <next>, "class": "table", "bbox": [x, y, w, h],
    "text": "PART LIST 3 M6 BOLT ...", "cells": [] }
  ```
  Tables use **flat text** for now (`cells` stays `[]`; structured/TEDS scoring is
  a later add-on — the hook is there so it won't need a rewrite).
- **VLM features**: fill `features.<name>.value` with the correct value.

Fill only what you have — blank `text` / `null` `value` are simply skipped.

### One master record
```json
{
  "image": "24088841_A_...",
  "width": 1280, "height": 1280,
  "meta": { "standard":"unknown","source_type":"unknown","clutter":"med",
            "has_gdt":false,"difficulty":"med" },
  "regions": [
    { "id":0, "class":"dimension", "bbox":[x,y,w,h], "text":"⌀25±0.1" },
    { "id":1, "class":"table",     "bbox":[x,y,w,h], "text":"PART LIST ...", "cells":[] }
  ],
  "features": { "length": {"value":120.0,"text":"120","bbox":null}, ... }
}
```

### 3. Build the ground truth + check progress
```bash
python dataset/build_from_master.py     # writes the per-stage GT files
python dataset/coverage.py              # what's left to annotate, N per metric
```
`build_from_master.py` projects the unified set, per region class, into:

| Consumer | File |
|----------|------|
| Detection (both classes) | `dataset/derived/unified_detection.coco.json` |
| OCR crops — dimensions | `ocr/data/ground_truth/dimensions.json` |
| OCR crops — tables | `ocr/data/ground_truth/tables.json` |
| OCR whole-image (union) | `ocr/data/ground_truth/_whole_image.json` |
| VLM features | `vlm/data/ground_truth/{dimensions,tables}.json` |
| Dimension parsed fields | `dataset/derived/dimensions_parsed.json` |

### 4. See results
Open the OCR / VLM pages and run. Both stages read images from
`dataset/selected_images/`, and scoring is restricted to whatever you've
annotated (unannotated images/features are skipped).

## Notes
- **Raw images and the original `val.json` are untouched.** The unified master is
  seeded from the dimension `val.json`; its dimension boxes match it.
- Everything is **idempotent** and safe under partial annotation — re-running
  `build_master.py` only adds missing dimension boxes and never clears your text,
  table regions, or features.
- `normalize.py` is the single place that defines text/number/symbol equivalence
  (`⌀`==`Ø`==`DIA`, `R`==`RAD`, `±`, decimal comma, case/space) and dimension
  parsing (value / tolerance / symbol / unit). Import it — don't re-implement.
