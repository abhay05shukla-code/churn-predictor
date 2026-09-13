"""Business-facing scoring helpers shared by the API and the training script."""

from __future__ import annotations

# (tier name, minimum probability). Checked top-down, so order matters.
RISK_TIERS: tuple[tuple[str, float], ...] = (
    ("High", 0.60),
    ("Medium", 0.30),
    ("Low", 0.00),
)


def risk_tier(probability: float) -> str:
    """Map a churn probability to a coarse tier for prioritisation."""
    for name, floor in RISK_TIERS:
        if probability >= floor:
            return name
    return RISK_TIERS[-1][0]


def monthly_revenue_at_risk(probability: float, monthly_charges: float) -> float:
    """Expected monthly revenue lost if nothing is done: P(churn) x monthly bill."""
    return round(float(probability) * float(monthly_charges), 2)


def annual_revenue_at_risk(probability: float, monthly_charges: float) -> float:
    """Expected annualised revenue lost (12 x the monthly figure)."""
    return round(float(probability) * float(monthly_charges) * 12, 2)
