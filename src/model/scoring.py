from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.model.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES


MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def risk_band(churn_probability: float) -> str:
    if churn_probability >= 0.60:
        return "high"
    if churn_probability >= 0.32:
        return "medium"
    return "low"


def recommended_action_category(customer: pd.Series | dict[str, Any], band: str) -> str:
    unresolved = float(customer["unresolved_ticket_count"])
    usage = float(customer["product_usage_score"])
    delay = float(customer["payment_delay_days"])
    onboarding = bool(customer["onboarding_completed"])

    if unresolved >= 3:
        return "escalate unresolved tickets"
    if usage < 45:
        return "investigate product usage decline"
    if not onboarding:
        return "offer onboarding support"
    if delay > 21:
        return "offer commercial renewal support"
    if band == "high":
        return "schedule executive check-in"
    return "send customer-success email"


def top_risk_drivers(customer: pd.Series | dict[str, Any], limit: int = 4) -> list[str]:
    drivers: list[tuple[float, str]] = [
        (max(0, 55 - float(customer["product_usage_score"])), "Low product usage"),
        (float(customer["unresolved_ticket_count"]) * 12, "Unresolved support tickets"),
        (max(0, float(customer["payment_delay_days"]) - 14), "Payment delay"),
        (max(0, float(customer["last_login_days"]) - 21), "Recent login inactivity"),
        (max(0, 10 - float(customer["nps_score"])) / 2, "Weak customer sentiment"),
        (0 if bool(customer["onboarding_completed"]) else 24, "Onboarding incomplete"),
    ]
    ranked = [driver for score, driver in sorted(drivers, reverse=True) if score > 0]
    return ranked[:limit] or ["No major structured risk driver detected"]


def score_customer(
    customer: pd.Series | dict[str, Any],
    model_path: str | Path = "data/processed/churn_model.joblib",
) -> dict[str, Any]:
    model = joblib.load(model_path)
    row = pd.DataFrame([dict(customer)])[MODEL_FEATURES]
    churn_probability = float(model.predict_proba(row)[0, 1])
    band = risk_band(churn_probability)
    mrr = float(customer["monthly_recurring_revenue"])
    return {
        "customer_id": customer["customer_id"],
        "churn_probability": round(churn_probability, 4),
        "risk_band": band,
        "revenue_at_risk": round(churn_probability * mrr * 12, 2),
        "top_risk_drivers": top_risk_drivers(customer),
        "recommended_action_category": recommended_action_category(customer, band),
    }


def score_customers(customers: pd.DataFrame, model_path: str | Path = "data/processed/churn_model.joblib") -> pd.DataFrame:
    scores = [score_customer(row, model_path) for _, row in customers.iterrows()]
    return pd.DataFrame(scores)

