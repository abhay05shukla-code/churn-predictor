"""End-to-end tests against the FastAPI app with the trained model loaded."""

from __future__ import annotations

import pytest

from tests.conftest import HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER, requires_model

pytestmark = requires_model


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_name"]
    assert 0 < body["threshold"] < 1


def test_predict_returns_well_formed_prediction(client):
    r = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in {"Low", "Medium", "High"}
    assert body["monthly_charges"] == HIGH_RISK_CUSTOMER["MonthlyCharges"]
    assert body["monthly_revenue_at_risk"] == pytest.approx(
        body["churn_probability"] * body["monthly_charges"], abs=0.01
    )
    assert body["annual_revenue_at_risk"] == pytest.approx(body["monthly_revenue_at_risk"] * 12, abs=0.05)
    assert body["will_churn"] == (body["churn_probability"] >= body["threshold"])


def test_high_risk_profile_scores_above_low_risk_profile(client):
    high = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()["churn_probability"]
    low = client.post("/predict", json=LOW_RISK_CUSTOMER).json()["churn_probability"]
    assert high > 0.5 > low
    assert high - low > 0.4


def test_total_charges_is_optional(client):
    payload = {k: v for k, v in HIGH_RISK_CUSTOMER.items() if k != "TotalCharges"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text


def test_moving_to_two_year_contract_lowers_risk(client):
    base = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()["churn_probability"]
    moved = client.post("/predict", json={**HIGH_RISK_CUSTOMER, "Contract": "Two year"}).json()[
        "churn_probability"
    ]
    assert moved < base


@pytest.mark.parametrize(
    "bad_patch",
    [
        {"Contract": "Three year"},
        {"MonthlyCharges": -5},
        {"tenure": "twelve"},
        {"SeniorCitizen": 2},
        {"unknown_field": 1},
        {"PhoneService": "No", "MultipleLines": "Yes"},
        {"InternetService": "No", "TechSupport": "No"},
    ],
)
def test_invalid_payloads_are_rejected_with_422(client, bad_patch):
    r = client.post("/predict", json={**HIGH_RISK_CUSTOMER, **bad_patch})
    assert r.status_code == 422, r.text


def test_missing_required_field_is_rejected(client):
    payload = {k: v for k, v in HIGH_RISK_CUSTOMER.items() if k != "Contract"}
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_matches_single_predictions_in_order(client):
    r = client.post("/predict/batch", json={"customers": [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    single_high = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()
    single_low = client.post("/predict", json=LOW_RISK_CUSTOMER).json()
    assert body["predictions"][0]["churn_probability"] == single_high["churn_probability"]
    assert body["predictions"][1]["churn_probability"] == single_low["churn_probability"]


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_model_info_exposes_metrics(client):
    info = client.get("/model/info").json()
    assert info["best_model"] in info["candidates"]
    assert 0.7 <= info["holdout"]["roc_auc"] <= 1.0
    assert len(info["feature_importance"]) == len(info["features"])
    assert 0 < info["business_impact"]["revenue_capture_rate"] <= 1
