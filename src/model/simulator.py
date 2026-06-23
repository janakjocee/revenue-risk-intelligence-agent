from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.recommendations import recommend_actions
from src.model.scoring import score_customer


SIMULATABLE_FIELDS = {
    "unresolved_ticket_count",
    "product_usage_score",
    "payment_delay_days",
    "nps_score",
    "account_health_score",
    "contract_type",
}


def simulate_customer_scenario(
    customer: dict[str, Any],
    adjustments: dict[str, Any],
    model_path: str | Path = "data/processed/churn_model.joblib",
) -> dict[str, Any]:
    scenario = dict(customer)
    applied_adjustments = {key: value for key, value in adjustments.items() if key in SIMULATABLE_FIELDS}
    scenario.update(applied_adjustments)
    score = score_customer(scenario, model_path=model_path)
    return {
        "customer_id": customer["customer_id"],
        "baseline": score_customer(customer, model_path=model_path),
        "scenario": score,
        "recommended_actions": recommend_actions(scenario, score),
        "applied_adjustments": applied_adjustments,
    }

