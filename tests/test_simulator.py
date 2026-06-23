import pandas as pd

from src.model.simulator import simulate_customer_scenario


def test_simulator_returns_baseline_and_scenario():
    customer = pd.read_csv("data/processed/customer_features.csv").iloc[0].to_dict()
    result = simulate_customer_scenario(
        customer,
        {
            "unresolved_ticket_count": 0,
            "product_usage_score": 90,
            "payment_delay_days": 0,
            "nps_score": 60,
            "account_health_score": 90,
        },
    )
    assert result["baseline"]["risk_band"] in {"low", "medium", "high"}
    assert result["scenario"]["risk_band"] in {"low", "medium", "high"}
    assert result["applied_adjustments"]["product_usage_score"] == 90

