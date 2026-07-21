from fastapi import APIRouter
from pydantic import BaseModel
from core.config_loader import read_text, write_text

router = APIRouter()

DEFAULTS = {
    "whole_image": "You are analyzing a 2D mechanical engineering drawing.\nExtract the manufacturing features and return ONLY a valid JSON object, no preamble.\nAnalyze the provided image directly.",
    "whole_image_ocr": "You are analyzing a 2D mechanical engineering drawing.\nYou are also given OCR-extracted text. Use the image as primary source, OCR as a hint.\nReturn ONLY a valid JSON object.",
    "cropped_ocr": "You are analyzing cropped patches from a 2D mechanical engineering drawing.\nCombine information from all patches. Use the crop images as primary source.\nReturn ONLY a valid JSON object.",
}


class PromptUpdate(BaseModel):
    mode: str
    text: str


@router.get("/{mode}")
def get_prompt(mode: str):
    text = read_text(f"prompts_{mode}")
    return {"mode": mode, "text": text or DEFAULTS.get(mode, "")}


@router.post("/")
def update_prompt(body: PromptUpdate):
    write_text(f"prompts_{body.mode}", body.text)
    return {"ok": True}


@router.post("/reset/{mode}")
def reset_prompt(mode: str):
    write_text(f"prompts_{mode}", DEFAULTS.get(mode, ""))
    return {"ok": True}