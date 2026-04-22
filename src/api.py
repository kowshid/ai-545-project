"""FastAPI inference service for champion insurance model."""
from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils import load_model, load_params

app = FastAPI(title="Insurance Charge Predictor API", version="1.0.0")
PARAMS = load_params()
MODEL = load_model(PARAMS)


class PredictRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: Literal["female", "male"]
    bmi: float = Field(ge=10.0, le=70.0)
    children: int = Field(ge=0, le=10)
    smoker: Literal["no", "yes"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": "yes" if MODEL is not None else "no"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, float | str]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Champion model not available")

    frame = pd.DataFrame([payload.model_dump()])
    try:
        pred = float(MODEL.predict(frame)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    return {"prediction": pred, "currency": "USD"}
