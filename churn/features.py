"""Single source of truth for the feature schema and data cleaning.

``train.py``, the FastAPI service and the tests all import from here, so the
model is always trained and served on exactly the same columns and categories.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"

ID_COLUMN = "customerID"
TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["SeniorCitizen"]  # already encoded as 0/1 in the raw file
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

_YES_NO = ["Yes", "No"]
_INTERNET_ADDON = ["Yes", "No", "No internet service"]

# Every allowed value for every categorical column. The OneHotEncoder is built
# from this mapping (so its output columns never depend on which values happen
# to appear in a training sample) and the API schema is tested against it.
CATEGORY_VALUES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": _INTERNET_ADDON,
    "OnlineBackup": _INTERNET_ADDON,
    "DeviceProtection": _INTERNET_ADDON,
    "TechSupport": _INTERNET_ADDON,
    "StreamingTV": _INTERNET_ADDON,
    "StreamingMovies": _INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

INTERNET_ADDON_FEATURES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the raw Telco CSV exactly as downloaded."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of ``df`` with consistent dtypes.

    * ``TotalCharges`` is stored as text in the raw file and is blank for
      brand-new customers (tenure 0). Coerce to float and treat blank as 0.
    * Strip stray whitespace from every categorical column.
    * Encode the target as 1 (churned) / 0 (stayed) when it is present.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce").fillna(0).astype(int)
    out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str).str.strip()
    if TARGET in out.columns:
        out[TARGET] = (
            out[TARGET].astype(str).str.strip().str.lower().isin({"yes", "1", "true"}).astype(int)
        )
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into model inputs and the churn label."""
    return df[ALL_FEATURES], df[TARGET]
