"""
Grounding DINO configured for TABLE detection — same open-vocabulary model as
grounding_dino.py, just prompted to find tables / title blocks instead of
dimensions. Zero-shot, no training. Install: pip install transformers torch

Kept as its own class so the registry loader (which calls cls(weights=...)) picks
the table prompt automatically, with no loader changes.
"""
from grounding_dino import GroundingDINODetector

TABLE_PROMPT = "table . title block . data table . parts list . bill of materials ."


class GroundingDINOTableDetector(GroundingDINODetector):
    def __init__(self, weights: str = None,
                 model_name: str = "IDEA-Research/grounding-dino-base",
                 prompt: str = TABLE_PROMPT):
        super().__init__(weights=weights, model_name=model_name, prompt=prompt)
