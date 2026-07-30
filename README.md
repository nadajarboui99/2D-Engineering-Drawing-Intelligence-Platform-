# 2D Engineering Drawing Intelligence Platform

This project reads 2D mechanical engineering drawings and pulls out the useful
values from them: lengths, widths, hole counts, hole diameters, material, part
name, drawing number, and so on.

There is more than one way to do that, and it is not obvious which way is best.
So this is a **test bench**. It runs three different methods on the same set of
drawings, scores each one against answers you filled in by hand, and shows the
results side by side.

## The three methods

| Stage | What it does |
|-------|--------------|
| **Detection** | Finds the boxes on the drawing: where the dimension labels are, where the tables are. Uses YOLOv11, RT-DETR, or Table Transformer. |
| **OCR** | Reads the text inside those boxes. Uses EasyOCR, Tesseract, TrOCR, or PaddleOCR. |
| **VLM** | Sends the whole drawing to Claude and asks it to return the values as JSON. |

The VLM stage runs in three ways so you can see if extra help changes the answer:
the image alone, the image plus the text OCR read off the whole page, and the
image plus the text OCR read from each box with its coordinates.

## How scoring works

You annotate a few drawings once, by hand, in the app. For each drawing you draw
the boxes, type the text you see in each one, and fill in the 15 feature values.
That becomes the answer key.

Every stage is then graded against that same answer key, so the numbers are
comparable. Detection gets mAP, precision and recall. OCR gets word and
character coverage. The VLM gets accuracy per field, and it also gets counted
when it invents a value that is not on the drawing.

## What you need installed

| Tool | Version | Needed for |
|------|---------|-----------|
| Python | 3.10 or newer | everything (tested on 3.11) |
| Node.js | 18 or newer | the web page and the Claude service (tested on 22) |
| Tesseract | any | only if you want to use the Tesseract OCR engine |
| Anthropic API key | | only if you want to run the VLM stage |

Tesseract is a separate program, not a Python package. On Mac:

```bash
brew install tesseract
```

## Install

Three installs, one per part. Run these from the project folder.

**1. Python side**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows the activate line is `venv\Scripts\activate` instead.

**2. Web page**

```bash
cd interface/frontend
npm install
cd ../..
```

**3. Claude service**

```bash
cd vlm/mastra-service
npm install
cd ../..
```

## Add your API key

Only needed for the VLM stage. Make a file at `vlm/.env` with this line:

```
ANTHROPIC_API_KEY=sk-ant-...
```

This file is already in `.gitignore`, so it will not be committed.

There is also an API keys page in the app. That page saves the key for the
Python side, but the Claude service reads `vlm/.env` only. So if the VLM stage
says it cannot reach the model, check that this file exists.

## Run it

You need three terminals open at the same time. Activate the Python venv in the
first one.

**Terminal 1, the backend**

```bash
cd interface/backend
uvicorn main:app --reload --port 8000
```

**Terminal 2, the web page**

```bash
cd interface/frontend
npm run dev
```

**Terminal 3, the Claude service**

```bash
cd vlm/mastra-service
npm start
```

Then open http://localhost:5173 in your browser.

| Part | Address |
|------|---------|
| Web page | http://localhost:5173 |
| Backend | http://localhost:8000 |
| Claude service | http://127.0.0.1:8787 |

The backend only accepts requests from port 5173, so use `npm run dev` rather
than serving the built files somewhere else.

## What to do first

A fresh copy of this project has no trained models and no training images. Those
files are large, so they are not committed. This changes what you can run right
away:

- **Annotate** works now. Put your drawing images in `dataset/selected_images/`,
  open the Annotate page, and fill them in.
- **VLM** works now, as soon as you have an API key.
- **OCR** works now if you pick "ground truth boxes" as the crop source, because
  that uses the boxes you drew. Using detector boxes needs a trained model first.
- **Detection** needs either a trained model or a `.pt` file you upload on the
  Detection page. Training also needs a dataset in
  `Table_dimensions_detection/data/`.

A normal first run looks like this: annotate two or three drawings, run OCR on
ground truth boxes, run the VLM, then compare them on the Results page.

## Working from the command line instead

The app does all of this for you, but the scripts also run on their own:

```bash
python dataset/build_master.py        # make blank annotation files for your images
python dataset/build_from_master.py   # turn your annotations into the answer key
python dataset/coverage.py            # show what is still missing
```

Run `build_from_master.py` after editing annotation files by hand. The app runs
it for you when you save from the Annotate page.

## Folders

| Folder | What is in it |
|--------|---------------|
| `dataset/` | Your images, your annotations, and the scripts that build the answer key |
| `Table_dimensions_detection/` | The detection stage: model wrappers, training, config |
| `ocr/` | The OCR stage: one wrapper file per engine |
| `vlm/` | The VLM stage: the feature list, the prompts, and the Claude service |
| `interface/backend/` | FastAPI server that runs the stages and does the scoring |
| `interface/frontend/` | The React web page |

Two files are worth knowing about:

- `vlm/configs/feature_schema.yaml` is the list of 15 values to extract. Add or
  remove a feature there and the prompts and scoring follow automatically.
- `dataset/normalize.py` decides when two pieces of text count as the same
  thing, so `⌀25`, `Ø25` and `DIA 25` all match. Every stage uses it.

## If something goes wrong

| Problem | Reason |
|---------|--------|
| Web page loads but nothing works | The backend is not running, or not on port 8000 |
| VLM stage fails right away | The Claude service is not running, or `vlm/.env` is missing |
| "No trained weights found" | Train a model, or upload a `.pt` file on the Detection page |
| Tesseract fails but EasyOCR works | The Tesseract program is not installed, only the Python package |
| An OCR engine says "isn't installed" | Install it from the optional list in `requirements.txt` |
| Scores show 0 images scored | Nothing is annotated yet, or the image names do not match |
