"""FastAPI service that serves the trained churn pipeline.

Run locally with::

    uvicorn api.main:app --reload --port 8000

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    Prediction,
    PredictionResponse,
)
from churn.features import ALL_FEATURES
from churn.scoring import annual_revenue_at_risk, monthly_revenue_at_risk, risk_tier

MODEL_DIR = Path(os.environ.get("CHURN_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


class ModelState:
    """Everything loaded once at startup and shared by every request."""

    pipeline: Any = None
    metrics: dict[str, Any] = {}
    model_name: str = ""
    threshold: float = 0.5


state = ModelState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise RuntimeError(
            f"Model artifacts not found in {MODEL_DIR}. Run `python train.py` first "
            "(or point CHURN_MODEL_DIR at a directory containing churn_pipeline.joblib and metrics.json)."
        )
    state.pipeline = joblib.load(MODEL_PATH)
    state.metrics = json.loads(METRICS_PATH.read_text())
    state.model_name = str(state.metrics["best_model"])
    state.threshold = float(state.metrics["threshold"])
    yield
    state.pipeline = None


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Serves a scikit-learn churn model trained on the IBM Telco dataset. "
        "POST customer attributes to `/predict` and get back a churn probability, "
        "a risk tier and the monthly revenue at risk."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def score(customers: list[CustomerFeatures]) -> np.ndarray:
    """Run the full preprocessing + model pipeline on validated customers."""
    frame = pd.DataFrame([c.model_dump() for c in customers])[ALL_FEATURES]
    return state.pipeline.predict_proba(frame)[:, 1]


def to_prediction(probability: float, monthly_charges: float) -> Prediction:
    p = round(float(probability), 4)
    return Prediction(
        churn_probability=p,
        will_churn=p >= state.threshold,
        risk_tier=risk_tier(p),
        monthly_charges=round(float(monthly_charges), 2),
        monthly_revenue_at_risk=monthly_revenue_at_risk(p, monthly_charges),
        annual_revenue_at_risk=annual_revenue_at_risk(p, monthly_charges),
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=state.model_name,
        threshold=state.threshold,
        trained_at=str(state.metrics.get("trained_at", "")),
    )


@app.get("/model/info", tags=["meta"], summary="Training metrics, feature importance and business impact")
def model_info() -> dict[str, Any]:
    return state.metrics


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Churn probability and revenue at risk for a single customer."""
    probability = score([customer])[0]
    base = to_prediction(probability, customer.MonthlyCharges)
    return PredictionResponse(**base.model_dump(), threshold=state.threshold, model_name=state.model_name)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["predictions"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score many customers in one call. Results are returned in request order."""
    probabilities = score(request.customers)
    predictions = [
        to_prediction(p, c.MonthlyCharges) for p, c in zip(probabilities, request.customers)
    ]
    return BatchPredictionResponse(
        threshold=state.threshold,
        model_name=state.model_name,
        count=len(predictions),
        predictions=predictions,
    )
