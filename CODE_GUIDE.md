# Code guide

What every file in this project does. For install and run steps see
[README.md](README.md).

## The big picture

One run has four steps. Everything else is plumbing around them.

```
you annotate drawings   ->   dataset/ turns that into answer keys
                                      |
     detection finds boxes  ->  OCR reads them  ->  VLM extracts values
                                      |
                     each stage scored, logged, shown in the web page
```

If you only read three files, read these:

1. [vlm/configs/feature_schema.yaml](vlm/configs/feature_schema.yaml) for what is
   being extracted.
2. [dataset/build_from_master.py](dataset/build_from_master.py) for how one
   annotation becomes six answer keys.
3. [interface/backend/core/vlm_eval.py](interface/backend/core/vlm_eval.py) for
   how a wrong answer is told apart from an invented one.

## dataset, the answer key

This folder is the heart of the project. Nothing gets scored without it.

| File | What it does |
|---|---|
| [build_master.py](dataset/build_master.py) | Makes a blank annotation file for each image in `selected_images/`. If an image also exists in the old detection val.json, it pre-fills the dimension boxes as a head start. Never overwrites work you already did. |
| [build_from_master.py](dataset/build_from_master.py) | Takes your one annotation file per image and splits it into six answer keys, one per consumer: detection COCO, OCR dimensions, OCR tables, OCR whole page, VLM features, parsed dimensions. Run after every edit. |
| [coverage.py](dataset/coverage.py) | Prints what is still blank and how many examples sit behind each metric. |
| [normalize.py](dataset/normalize.py) | Decides when two bits of text mean the same thing, so `⌀25`, `Ø25` and `DIA 25` all match, and `25,5` equals `25.5`. Also splits a callout like `⌀25±0.1 mm` into value, tolerance, symbol and unit. Run `python dataset/normalize.py` for its self test. |
| `master/unified/*.json` | Your actual annotations. One file per drawing with boxes, the text in each box, and the 15 feature values. This is the only file you hand edit. |
| `master/_TEMPLATE.json` | Shows the shape of the above. |
| `selected_images/` | Your drawing images. |
| `derived/` | Auto generated, do not edit. |

Why it is built this way: one image set annotated for both dimensions and tables.
So when detection scores worse than OCR, you know it is the task and not a
different pile of images.

## Table_dimensions_detection, stage 1, find the boxes

| File | What it does |
|---|---|
| [train_tables.py](Table_dimensions_detection/train_tables.py) and [train_dimensions.py](Table_dimensions_detection/train_dimensions.py) | Nearly identical. Read a config, build a model, train, then evaluate. `--eval_only --weights x.pt` skips training. The dimensions one also shifts COCO category ids by 1 to match YOLO. |
| [models/yolov11.py](Table_dimensions_detection/models/yolov11.py) | Wraps Ultralytics YOLO behind `train()` and `predict()`. |
| [models/rtdetr.py](Table_dimensions_detection/models/rtdetr.py) | Same interface, RT-DETR instead. Swappable by one config line. |
| [models/table_transformer.py](Table_dimensions_detection/models/table_transformer.py) | Microsoft's table detector from HuggingFace. Off the shelf, no training. Same `predict()` shape so it drops into the same slots. |
| [utils/dataset.py](Table_dimensions_detection/utils/dataset.py) | Loads COCO JSON as a PyTorch dataset and converts boxes from `[x,y,w,h]` to `[x1,y1,x2,y2]`. |
| [utils/metrics.py](Table_dimensions_detection/utils/metrics.py) | Precision, recall, F1 and mAP@0.5 using 11 point interpolation. Used during training only. |
| [utils/convert_coco_to_yolo.py](Table_dimensions_detection/utils/convert_coco_to_yolo.py) | One time prep. YOLO wants one `.txt` per image with normalized centre coordinates, this writes them. |
| `configs/tables.yaml` and `configs/dimensions.yaml` | Model, size, epochs, batch, image size, device. Tables default to YOLOv11-n at 640px, dimensions to RT-DETR-x at 1280px. |
| [python_script.py](Table_dimensions_detection/python_script.py) | A one off fixer for messy annotations. Lowercases `Table` to `table`, merges the duplicate category that creates, drops a junk `New-Drawings` class. Already did its job. |

## ocr, stage 2, read the text

| File | What it does |
|---|---|
| [models/base.py](ocr/models/base.py) | The contract. Every engine implements `read(image) -> str`. |
| [models/easyocr_model.py](ocr/models/easyocr_model.py) | EasyOCR. Also has `read_with_confidence()`, which is what whole page mode uses to get one entry per text region. |
| [models/tesseract_model.py](ocr/models/tesseract_model.py) | Tesseract with `--psm 6`, which treats the crop as one block. Needs the separate Tesseract program installed. |
| [models/trocr_model.py](ocr/models/trocr_model.py) | HuggingFace TrOCR. A single line reader, so good on crops and bad on full pages. |
| [models/paddleocr_model.py](ocr/models/paddleocr_model.py) | PaddleOCR, strong on dense documents. |
| [run_ocr.py](ocr/run_ocr.py) | Standalone version: detect, crop, read, save JSON. The app does not use this, it has its own copy of the logic. |
| `configs/ocr.yaml` | Which engine, which weights, confidence threshold. |
| `data/ground_truth/` | Written by `build_from_master.py`. `dimensions.json` and `tables.json` per class, `_whole_image.json` is everything on the page merged. |

The last three engines import lazily on purpose, so the backend never touches
`paddle` or `transformers` unless you actually pick that engine.

## vlm, stage 3, ask Claude

| File | What it does |
|---|---|
| [configs/feature_schema.yaml](vlm/configs/feature_schema.yaml) | The most important config in the project. The 15 values to extract with type, unit and description. The prompt and the scoring are both generated from this, so adding a feature here needs no code change. |
| [prompts/prompt_builder.py](vlm/prompts/prompt_builder.py) | Turns that schema into a prompt with a JSON template. Three variants for the three input modes. The crop one adds rules about engineering notation and warns about 0 versus O and 1 versus I. |
| [models/base.py](vlm/models/base.py) | The contract. `extract(images, text_context, prompt) -> dict`. |
| [models/mastra_vlm.py](vlm/models/mastra_vlm.py) | The one in use. Shrinks the image to 1568px, base64s it, posts to the local Node service, strips markdown fences off the reply, parses JSON. Also records tokens, latency and an estimated cost. |
| [mastra-service/server.mjs](vlm/mastra-service/server.mjs) | Small Express server on port 8787. The only place that talks to Anthropic. Reads the key from `vlm/.env` and exits if it is missing. |
| [models/claude_vlm.py](vlm/models/claude_vlm.py) | The older litellm path, replaced by the Mastra one. Only the standalone script still uses it. |
| [run_vlm.py](vlm/run_vlm.py) | Standalone runner for the three modes. The app does not use it. |
| `data/ground_truth/unified.json` | The feature answer key, one entry per drawing. |

Why the Node middleman: it keeps all the model provider code in one file, so the
Python side stays pure evaluation logic.

## interface/backend, the server

[main.py](interface/backend/main.py) loads saved API keys, then mounts 13
routers. CORS is locked to `localhost:5173`.

### core, the logic

| File | What it does |
|---|---|
| [job_manager.py](interface/backend/core/job_manager.py) | Training takes minutes, so every long task becomes a background job with an id and a log. The page polls `/jobs/{id}`. Jobs live in memory, so a restart loses them. |
| [detection_eval.py](interface/backend/core/detection_eval.py) | Scores boxes against your boxes. mAP over all confidences, plus precision, recall and F1 at 0.25. This is why you sometimes see mAP 0.59 next to precision 0.0, the two use different thresholds. |
| [ocr_eval.py](interface/backend/core/ocr_eval.py) | Two scoring styles. Per string with edit distance gives CER and WER. Whole text treats the page as one bag of words and measures coverage, which is the fair way to score a table block. |
| [vlm_eval.py](interface/backend/core/vlm_eval.py) | Sorts every field into correct, wrong, missed, hallucinated or abstained. The hallucinated bucket is why the answer key stores explicit nulls. Numbers pass within 2 percent or 0.5 absolute. |
| [results_store.py](interface/backend/core/results_store.py) | Appends every run to `results_store.json`, and saves a full snapshot per run so you can reopen the whole dashboard later. |
| [weights_finder.py](interface/backend/core/weights_finder.py) | Hunts every `best.pt` under `runs/`, reads its `results.csv`, picks the best mAP. Falls back to the newest file if no run has real numbers. |
| [ocr_registry.py](interface/backend/core/ocr_registry.py) and [detection_registry.py](interface/backend/core/detection_registry.py) | The plug in lists. Each entry names a wrapper class, a pip command and an import to test. That is how the app knows PaddleOCR exists but is not installed. |
| [weights_registry.py](interface/backend/core/weights_registry.py) | Tracks `.pt` files you uploaded. |
| [image_enhancement.py](interface/backend/core/image_enhancement.py) | Crop clean up before OCR. `basic` is 2x upscale plus sharpen and is always used. `full` adds binarizing, which can wipe faint text. |
| [config_loader.py](interface/backend/core/config_loader.py) | One place that knows where every YAML and prompt file lives. |
| [api_keys.py](interface/backend/core/api_keys.py) | Saves keys to `.keys.json` at file mode 600, loads them into env at startup. |

### routers, the endpoints

| File | What it does |
|---|---|
| [detection.py](interface/backend/routers/detection.py) | Biggest router. Train, evaluate on the training set, and evaluate on your annotations with box overlays. Also holds the `sys.path` juggling that stops the three `models` folders shadowing each other. |
| [ocr.py](interface/backend/routers/ocr.py) | Runs the three approaches, saves crops as images so you can look at them, and computes metrics live so adding answers later needs no re run. |
| [vlm.py](interface/backend/routers/vlm.py) | Runs the modes, merges into `all_results.json`, and `/compare` puts accuracy next to tokens and cost per mode. |
| [annotation.py](interface/backend/routers/annotation.py) | Saves what you draw in the same format as the hand written files, then re runs `build_from_master.py` so scores stay current. |
| [results.py](interface/backend/routers/results.py) | Lists runs, reopens snapshots, deletes runs. |
| [jobs.py](interface/backend/routers/jobs.py) | Job status polling. |
| [features.py](interface/backend/routers/features.py) and [prompts.py](interface/backend/routers/prompts.py) | Edit the feature list and the prompts from the browser. |
| [keys.py](interface/backend/routers/keys.py) | API key save and provider list. |
| [weights.py](interface/backend/routers/weights.py) | Upload a `.pt` file. |
| [ocr_models.py](interface/backend/routers/ocr_models.py) and [models.py](interface/backend/routers/models.py) | Install an engine by running its pip command. |
| [pipeline.py](interface/backend/routers/pipeline.py) | Meant to chain all three stages. Currently broken, it imports two functions that no longer exist under those names. |

## interface/frontend, the web page

| File | What it does |
|---|---|
| [App.jsx](interface/frontend/src/App.jsx) | Sidebar and page switching. No router library, just state. |
| [api/client.js](interface/frontend/src/api/client.js) | Every backend call in one object, plus `pollJob` which watches a job until it finishes. |
| [components/ui.jsx](interface/frontend/src/components/ui.jsx) | Card, Panel, Btn, Tabs, Badge, JobLog and friends. |
| [components/runViews.jsx](interface/frontend/src/components/runViews.jsx) | Turns a saved snapshot back into its dashboard. Shared so one run and a side by side comparison render identically. |
| [pages/Annotate.jsx](interface/frontend/src/pages/Annotate.jsx) | The annotator. Draw boxes on an SVG, type text, fill features. Survives tab switches and saves a draft to localStorage so a reload does not lose work. |
| [pages/Detection.jsx](interface/frontend/src/pages/Detection.jsx) | Pick a model, evaluate, see boxes drawn on your drawing. Green found, red missed, purple table, blue dimension. |
| [pages/OCR.jsx](interface/frontend/src/pages/OCR.jsx) | The three approaches with a compare tab. Words show as chips, green if OCR got it. Also shows the actual crops. |
| [pages/VLM.jsx](interface/frontend/src/pages/VLM.jsx) | Run the modes, edit the prompt inline, then a per field table of expected against extracted, coloured by verdict. |
| [pages/Results.jsx](interface/frontend/src/pages/Results.jsx) | Every run ever, with filters and sorting. Tick two to compare. Only lets you compare things that are actually comparable. |
| [pages/Settings.jsx](interface/frontend/src/pages/Settings.jsx) | API keys. |
| [pages/Other.jsx](interface/frontend/src/pages/Other.jsx) | Two pages in one file: edit the feature schema, and add a model. Also holds a `PipelinePage` that nothing imports. |
| [pages/Pipeline.jsx](interface/frontend/src/pages/Pipeline.jsx) | The one actually used for the full pipeline. |

## Files you can ignore

| File | Why |
|---|---|
| `__init__.py` | Empty package markers. |
| `interface/backend/weights/weights_registry.py` | Copy of the core one. That folder is also where uploaded weights land. |
| `Table_dimensions_detection/run_ocr.py` and its `configs/ocr.yaml` | Older copies of the files in `ocr/`. |
| `vlm/models/claude_vlm.py` | Replaced by `mastra_vlm.py`. |
| `dataset/_archive/` | Old per task annotations, kept but unused. |
| `PipelinePage` inside `Other.jsx` | Dead export, `Pipeline.jsx` is the live one. |
