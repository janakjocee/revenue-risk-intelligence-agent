import pandas as pd

from src.model.scoring import risk_band, score_customer


def test_risk_band_thresholds():
    assert risk_band(0.20) == "low"
    assert risk_band(0.40) == "medium"
    assert risk_band(0.70) == "high"


def test_score_customer_output_shape():
    customer = pd.read_csv("data/processed/customer_features.csv").iloc[0].to_dict()
    score = score_customer(customer)
    assert {"churn_probability", "risk_band", "revenue_at_risk", "top_risk_drivers"}.issubset(score)
    assert 0 <= score["churn_probability"] <= 1
    assert score["risk_band"] in {"low", "medium", "high"}

