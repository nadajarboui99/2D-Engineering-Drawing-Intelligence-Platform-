"""
Prompt builder — generates the instruction prompt from feature_schema.yaml.
Same schema is reused across all 3 input modes and all VLMs.
"""

import yaml


def load_schema(schema_path: str = "configs/feature_schema.yaml") -> list:
    with open(schema_path) as f:
        data = yaml.safe_load(f)
    return data["features"]


def build_json_template(features: list) -> str:
    """Builds an empty JSON template string showing the expected output shape."""
    lines = ["{"]
    for i, feat in enumerate(features):
        comma = "," if i < len(features) - 1 else ""
        unit_note = f" ({feat['unit']})" if feat.get("unit") else ""
        default = "[]" if feat.get("type") == "list" else "null"
        lines.append(f'  "{feat["name"]}": {default}{comma}  // {feat["type"]}{unit_note} — {feat["description"]}')
    lines.append("}")
    return "\n".join(lines)


def build_prompt(features: list, mode: str, text_context: str = "",
                 custom_instruction: str = "") -> str:
    """
    mode: "whole_image" | "whole_image_ocr" | "cropped_ocr"

    custom_instruction: optional user-written intro (from the prompt editor). It
    REPLACES only the opening instruction — the JSON schema template and the
    mode-specific OCR context block are ALWAYS appended, so a custom prompt can
    never accidentally drop the schema or the OCR text the mode is supposed to use.
    """
    template = build_json_template(features)

    intro = (custom_instruction.strip() if custom_instruction and custom_instruction.strip()
             else "You are analyzing a 2D mechanical engineering drawing.\n"
                  "Extract the following manufacturing features from it.")

    # These rules are ALWAYS appended, whatever the (custom) intro says — so a
    # changed prompt can never drop the JSON-output contract or the schema.
    base_instruction = f"""{intro}

Return ONLY a single valid JSON object matching the schema below — no preamble,
no explanation, no markdown code fences. If a feature is not present or cannot be
determined, use null.

Expected JSON schema:
{template}
"""

    if mode == "whole_image":
        return base_instruction + "\nAnalyze the provided image directly."

    elif mode == "whole_image_ocr":
        return base_instruction + f"""
You are also given OCR-extracted text from this image to help you, but the OCR
may contain errors — use the image as the primary source of truth and the OCR
text as a hint.

OCR extracted text:
\"\"\"
{text_context}
\"\"\"
"""

    elif mode == "cropped_ocr":
        return base_instruction + f"""
You are an expert in reading mechanical engineering drawings and technical dimension annotations.

You are given:

* The FULL drawing image (the primary source of truth).
* OCR text read from regions a detector located on this drawing, each line prefixed
  with the bounding box [box x1, y1, x2, y2] (pixel coords) where the text was found.
  OCR may contain errors — always verify against the image.
* A JSON schema describing the features to extract.

Important rules:

1. The image is the primary source of truth. Use the OCR text and box locations only
   as a supporting signal for where to look and what the text likely says.
2. Mechanical drawing notation must be interpreted correctly:

   * Linear dimensions (e.g., 25, 150.5)
   * Diameters (⌀, Ø)
   * Radii (R)
   * Angles (°)
   * Thread annotations (M8, M10x1.5, etc.)
   * Chamfers
   * Tolerances (±, limit dimensions)
   * Depth symbols
   * Hole callouts and feature annotations
3. OCR may confuse characters such as 0/O, 1/I, 5/S, 8/B, and decimal points/commas.
   Always verify against the image.
4. Extract only values that are explicitly present in the drawing. Do not invent dimensions.
5. If a requested field cannot be determined with confidence, return null.

OCR text with box locations:
\"\"\"
{text_context}
\"\"\"
"""

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use: whole_image | whole_image_ocr | cropped_ocr")