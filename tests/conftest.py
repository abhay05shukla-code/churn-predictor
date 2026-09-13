"""Shared fixtures: an in-process API client and canonical customer payloads."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import METRICS_PATH, MODEL_PATH, app

MODEL_TRAINED = MODEL_PATH.exists() and METRICS_PATH.exists()
requires_model = pytest.mark.skipif(not MODEL_TRAINED, reason="run `python train.py` first")

HIGH_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 95.0,
}

LOW_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 70,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 60.0,
    "TotalCharges": 4200.0,
}


@pytest.fixture(scope="session")
def client():
    """TestClient used as a context manager so the lifespan loads the model."""
    if not MODEL_TRAINED:
        pytest.skip("run `python train.py` first")
    with TestClient(app) as c:
        yield c
