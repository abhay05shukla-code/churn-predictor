"""Pydantic request/response models for the churn API.

Field names deliberately match the Telco dataset columns so a customer row can
be sent to the API exactly as it appears in the CSV.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER: dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
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
    "MonthlyCharges": 89.5,
    "TotalCharges": 179.0,
}


class CustomerFeatures(BaseModel):
    """One customer's attributes. Unknown fields are rejected so typos surface as 422s."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 if the customer is 65 or older")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months the customer has stayed with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000, description="Current monthly bill in USD")
    TotalCharges: float | None = Field(
        default=None,
        ge=0,
        description="Lifetime charges in USD. Defaults to tenure x MonthlyCharges when omitted.",
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> "CustomerFeatures":
        if self.TotalCharges is None:
            self.TotalCharges = round(self.tenure * self.MonthlyCharges, 2)
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError("MultipleLines must be 'No phone service' when PhoneService is 'No'")
        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            raise ValueError("MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'")
        addons = {
            "OnlineSecurity": self.OnlineSecurity,
            "OnlineBackup": self.OnlineBackup,
            "DeviceProtection": self.DeviceProtection,
            "TechSupport": self.TechSupport,
            "StreamingTV": self.StreamingTV,
            "StreamingMovies": self.StreamingMovies,
        }
        if self.InternetService == "No":
            bad = [k for k, v in addons.items() if v != "No internet service"]
            if bad:
                raise ValueError(
                    f"{', '.join(bad)} must be 'No internet service' when InternetService is 'No'"
                )
        else:
            bad = [k for k, v in addons.items() if v == "No internet service"]
            if bad:
                raise ValueError(
                    f"{', '.join(bad)} cannot be 'No internet service' when InternetService is "
                    f"'{self.InternetService}'"
                )
        return self


class Prediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1, description="P(customer cancels)")
    will_churn: bool = Field(description="True when churn_probability >= the model's tuned threshold")
    risk_tier: Literal["Low", "Medium", "High"]
    monthly_charges: float
    monthly_revenue_at_risk: float = Field(description="churn_probability x monthly_charges")
    annual_revenue_at_risk: float = Field(description="12 x monthly_revenue_at_risk")


class PredictionResponse(Prediction):
    threshold: float
    model_name: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=20_000)


class BatchPredictionResponse(BaseModel):
    threshold: float
    model_name: str
    count: int
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_name: str
    threshold: float
    trained_at: str
