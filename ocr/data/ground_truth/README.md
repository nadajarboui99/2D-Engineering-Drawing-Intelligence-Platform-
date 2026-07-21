# OCR ground truth

Put the answer key for OCR accuracy here — **one file per task**:

- `tables.json`
- `dimensions.json`

Each file is a single JSON object keyed by **image name** (the file name of the
validation image, without its extension). The value is the list of the true
text strings that appear in that image:

```json
{
  "drawing_001": ["45.5", "M6", "R10", "ISO 2768"],
  "drawing_002": ["120", "Ø8", "Steel 1045"]
}
```

## Which images?

The validation images that OCR runs on:

| Task       | Images folder                                                  |
|------------|----------------------------------------------------------------|
| tables     | `Table_dimensions_detection/data/tables/valid/`                |
| dimensions | `Table_dimensions_detection/data/dimensions/val/images/`       |

A ready-to-fill `*.template.json` with every image name already listed is in
this folder — copy it to `<task>.json` and fill in the text lists:

```bash
cp tables.template.json tables.json      # then edit tables.json
```

## One ground truth, both modes

The same file scores **both** OCR modes — "Full image" and "Cropped patches".
The answer key is *what text is truly in the drawing*; it doesn't depend on how
the text was extracted. You do **not** need separate ground truth per mode.

## How it's scored

Metrics appear on the OCR page automatically once `<task>.json` exists (no
re-run needed — click Run OCR again, or they compute from the last run):

- **CER** — character error rate (lower is better)
- **WER** — word error rate
- **Exact match** — fraction of true strings extracted character-perfect
- **Precision / Recall / F1** — did we find the strings that are there, without
  inventing extra ones

Matching is case-insensitive and whitespace-normalized; each true string is
compared against its best predicted match.
