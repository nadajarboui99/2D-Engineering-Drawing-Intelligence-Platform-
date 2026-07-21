# VLM ground truth

Put the answer key for VLM feature-extraction accuracy here — **one file per
task**:

- `tables.json`
- `dimensions.json`

Each file is a single JSON object keyed by **image name** (validation image
file name without extension). The value is the correct feature values for that
image (a subset of the fields in `vlm/configs/feature_schema.yaml`):

```json
{
  "drawing_001": {
    "length": 120.0,
    "width": 60.0,
    "bends_count": 2,
    "holes_count": 4,
    "hole_diameter": 8.0,
    "cut_perimeter": 360.0,
    "material": "Al 6061",
    "standard": "ISO 2768"
  },
  "drawing_002": { "length": 88.5, "holes_count": 2 }
}
```

Only the fields you fill in are scored. Leave a field out (or `null`) if it
doesn't apply to that drawing — partial annotation is fine.

## Which images?

| Task       | Images folder                                            |
|------------|----------------------------------------------------------|
| tables     | `Table_dimensions_detection/data/tables/valid/`          |
| dimensions | `Table_dimensions_detection/data/dimensions/val/images/` |

A ready-to-fill `*.template.json` listing every image with all schema fields set
to `null` is in this folder — copy it and fill in the values:

```bash
cp dimensions.template.json dimensions.json      # then edit dimensions.json
```

## How it's scored

Metrics appear on the VLM page once `<task>.json` exists (computed from the last
run's saved results — no re-run needed):

- **Field accuracy** — correct fields / total annotated fields, across all images
- **Exact match** — fraction of images where every annotated field is correct
- **Per-field accuracy** — a breakdown so you can see which features the model
  gets right

Numbers are compared with a small tolerance (within 2% or 0.5 absolute);
strings are compared case-insensitively and whitespace-normalized. Scores are
reported **per input mode** (whole image / image+OCR / crops+OCR) so you can
compare them.
