import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.api_keys import load_all_to_env
load_all_to_env()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import detection, ocr, vlm, pipeline, annotation, features, prompts, jobs, models, results, weights, keys, ocr_models

app = FastAPI(title="Eval Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection.router, prefix="/detection", tags=["detection"])
app.include_router(ocr.router,       prefix="/ocr",       tags=["ocr"])
app.include_router(vlm.router,       prefix="/vlm",       tags=["vlm"])
app.include_router(pipeline.router,   prefix="/pipeline",   tags=["pipeline"])
app.include_router(annotation.router, prefix="/annotation", tags=["annotation"])
app.include_router(features.router,  prefix="/features",  tags=["features"])
app.include_router(prompts.router,   prefix="/prompts",   tags=["prompts"])
app.include_router(jobs.router,      prefix="/jobs",      tags=["jobs"])
app.include_router(models.router,    prefix="/models",    tags=["models"])
app.include_router(results.router,   prefix="/results",   tags=["results"])
app.include_router(weights.router,    prefix="/weights",    tags=["weights"])
app.include_router(keys.router,       prefix="/keys",       tags=["keys"])
app.include_router(ocr_models.router, prefix="/ocr-models", tags=["ocr-models"])

@app.get("/")
def root():
    return {"status": "ok", "version": "1.0.0"}