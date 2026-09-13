"""Tests for the shared feature schema and cleaning logic."""

from __future__ import annotations

import typing

import pandas as pd
import pytest

from api.schemas import CustomerFeatures
from churn.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORY_VALUES,
    DATA_PATH,
    TARGET,
    clean,
    load_raw,
)
from churn.scoring import monthly_revenue_at_risk, risk_tier


def test_clean_coerces_blank_total_charges_and_target():
    raw = pd.DataFrame(
        {
            "tenure": ["0", "12"],
            "MonthlyCharges": ["50.5", "80"],
            "TotalCharges": [" ", "960"],
            "SeniorCitizen": [0, 1],
            **{c: [CATEGORY_VALUES[c][0] + " ", CATEGORY_VALUES[c][-1]] for c in CATEGORICAL_FEATURES},
            TARGET: ["Yes", "No"],
        }
    )
    out = clean(raw)
    assert out["TotalCharges"].tolist() == [0.0, 960.0]
    assert out["tenure"].tolist() == [0, 12]
    assert out[TARGET].tolist() == [1, 0]
    assert out["gender"].iloc[0] == CATEGORY_VALUES["gender"][0]  # whitespace stripped
    assert list(out[ALL_FEATURES].columns) == ALL_FEATURES


def test_schema_literals_match_category_values():
    """The API's Literal types and the encoder's category lists must never drift apart."""
    hints = typing.get_type_hints(CustomerFeatures)
    for col, allowed in CATEGORY_VALUES.items():
        assert set(typing.get_args(hints[col])) == set(allowed), col


def test_schema_covers_every_model_feature():
    assert set(CustomerFeatures.model_fields) == set(ALL_FEATURES)


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not downloaded")
def test_dataset_values_are_all_known_categories():
    df = clean(load_raw())
    for col, allowed in CATEGORY_VALUES.items():
        unknown = set(df[col].unique()) - set(allowed)
        assert not unknown, f"{col} has values missing from CATEGORY_VALUES: {unknown}"
    assert df["TotalCharges"].isna().sum() == 0


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not downloaded")
def test_every_dataset_row_passes_api_validation():
    """The dashboard sends the whole CSV through /predict/batch, so every row must validate."""
    df = clean(load_raw())
    for row in df[ALL_FEATURES].to_dict("records"):
        CustomerFeatures.model_validate(row)


@pytest.mark.parametrize(
    "p, tier",
    [(0.0, "Low"), (0.29, "Low"), (0.30, "Medium"), (0.59, "Medium"), (0.60, "High"), (1.0, "High")],
)
def test_risk_tier_boundaries(p, tier):
    assert risk_tier(p) == tier


def test_revenue_at_risk_is_probability_times_bill():
    assert monthly_revenue_at_risk(0.5, 80.0) == 40.0
    assert monthly_revenue_at_risk(0.0, 80.0) == 0.0
