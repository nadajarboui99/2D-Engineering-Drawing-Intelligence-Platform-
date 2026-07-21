import sys
import os

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import List
from core.job_manager import create_job, run_job

router = APIRouter()


class PipelineConfig(BaseModel):
    stages:          List[str] = ["detection", "ocr", "vlm"]
    task:            str       = "tables"
    detection_model: str       = "yolov11"
    detection_size:  str       = "n"
    ocr_model:       str       = "easyocr"
    vlm_model:       str       = "claude"
    input_mode:      str       = "cropped_ocr"
    epochs:          int       = 100
    batch_size:      int       = 16
    imgsz:           int       = 640
    device:          str       = "mps"


@router.post("/run")
async def run_pipeline(cfg: PipelineConfig, background: BackgroundTasks):
    job = create_job(f"Pipeline · {' → '.join(cfg.stages)} · {cfg.task}")

    def _run(job, cfg):
        tasks = ["tables", "dimensions"] if cfg.task == "both" else [cfg.task]

        if "detection" in cfg.stages:
            from routers.detection import _run as _det_run
            from routers.detection import DetectionConfig
            for task in tasks:
                det_cfg = DetectionConfig(
                    task=task,
                    model_name=cfg.detection_model,
                    model_size=cfg.detection_size,
                    epochs=cfg.epochs,
                    batch_size=cfg.batch_size,
                    imgsz=cfg.imgsz,
                    device=cfg.device,
                )
                job.log(f"[Detection] Starting {task}...")
                _det_run(job=job, task=task, cfg=det_cfg)

        if "ocr" in cfg.stages:
            from routers.ocr import _run_ocr_for_task, OCRConfig
            ocr_cfg = OCRConfig(task=cfg.task, ocr_model=cfg.ocr_model, imgsz=cfg.imgsz)
            for task in tasks:
                job.log(f"[OCR] Starting {task}...")
                _run_ocr_for_task(task, ocr_cfg, job)

        if "vlm" in cfg.stages:
            from routers.vlm import _run_vlm_for_task, VLMConfig
            vlm_cfg = VLMConfig(
                vlm_model=cfg.vlm_model,
                modes=[cfg.input_mode],
                task=cfg.task,
                ocr_model=cfg.ocr_model,
            )
            for task in tasks:
                job.log(f"[VLM] Starting {task}...")
                _run_vlm_for_task(task, vlm_cfg, job)

        job.log("Pipeline complete.")
        return {"stages": cfg.stages, "task": cfg.task}

    background.add_task(run_job, job, _run, cfg=cfg)
    return {"job_id": job.job_id}