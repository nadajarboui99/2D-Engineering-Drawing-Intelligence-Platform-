from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from core.config_loader import load, save

router = APIRouter()


class Feature(BaseModel):
    name:        str
    type:        str
    unit:        Optional[str] = None
    description: str


@router.get("/")
def get_features():
    schema = load("feature_schema")
    return schema.get("features", [])


@router.post("/")
def add_feature(feature: Feature):
    schema = load("feature_schema")
    schema.setdefault("features", []).append(feature.dict())
    save("feature_schema", schema)
    return {"ok": True}


@router.delete("/{name}")
def delete_feature(name: str):
    schema = load("feature_schema")
    schema["features"] = [f for f in schema.get("features", []) if f["name"] != name]
    save("feature_schema", schema)
    return {"ok": True}