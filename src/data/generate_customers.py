from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


INDUSTRIES = [
    "FinTech",
    "HealthTech",
    "Retail",
    "Manufacturing",
    "Professional Services",
    "SaaS",
    "Education",
    "Logistics",
]

CONTRACT_TYPES = ["monthly", "annual", "multi-year"]
COMPANY_PREFIXES = ["Northstar", "Bright", "Summit", "Atlas", "Civic", "Harbour", "Vertex", "Pioneer"]
COMPANY_SUFFIXES = ["Analytics", "Systems", "Group", "Labs", "Works", "Cloud", "Partners", "Digital"]


@dataclass(frozen=True)
class CustomerDataConfig:
    rows: int = 500
    seed: int = 42


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def generate_customer_data(config: CustomerDataConfig = CustomerDataConfig()) -> pd.DataFrame:
    """Generate synthetic account data with an interpretable churn signal."""
    rng = np.random.default_rng(config.seed)
    rows = config.rows

    industry = rng.choice(INDUSTRIES, rows)
    contract_type = rng.choice(CONTRACT_TYPES, rows, p=[0.28, 0.52, 0.20])
    tenure_months = rng.integers(2, 73, rows)
    contract_value = rng.lognormal(mean=10.2, sigma=0.7, size=rows).round(0)
    monthly_recurring_revenue = (contract_value / 12).round(2)
    product_usage_score = np.clip(rng.normal(67, 18, rows), 0, 100).round(1)
    support_ticket_count = rng.poisson(5, rows)
    unresolved_ticket_count = np.minimum(support_ticket_count, rng.poisson(1.7, rows))
    payment_delay_days = np.clip(rng.gamma(shape=1.8, scale=6, size=rows), 0, 60).round(0)
    nps_score = np.clip(rng.normal(28, 34, rows), -100, 100).round(0)
    last_login_days = np.clip(rng.gamma(shape=2.0, scale=8, size=rows), 0, 90).round(0)
    onboarding_completed = rng.choice([True, False], rows, p=[0.82, 0.18])

    health_score = (
        0.48 * product_usage_score
        + 0.20 * (100 - np.clip(last_login_days * 1.7, 0, 100))
        + 0.18 * ((nps_score + 100) / 2)
        + 0.14 * (100 - np.clip(unresolved_ticket_count * 18, 0, 100))
    )
    health_score = np.where(onboarding_completed, health_score + 4, health_score - 12)
    account_health_score = np.clip(health_score, 0, 100).round(1)

    churn_logit = (
        -2.7
        + 0.034 * (100 - product_usage_score)
        + 0.38 * unresolved_ticket_count
        + 0.030 * payment_delay_days
        + 0.020 * last_login_days
        - 0.013 * nps_score
        - 0.018 * tenure_months
        + np.where(contract_type == "monthly", 0.38, 0)
        + np.where(onboarding_completed, -0.35, 0.45)
    )
    churn_probability = _sigmoid(churn_logit)
    churn = rng.binomial(1, churn_probability)

    company_names = [
        f"{rng.choice(COMPANY_PREFIXES)} {rng.choice(COMPANY_SUFFIXES)} {1000 + idx}"
        for idx in range(rows)
    ]

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{idx:04d}" for idx in range(1, rows + 1)],
            "company_name": company_names,
            "industry": industry,
            "contract_value": contract_value,
            "monthly_recurring_revenue": monthly_recurring_revenue,
            "tenure_months": tenure_months,
            "product_usage_score": product_usage_score,
            "support_ticket_count": support_ticket_count,
            "unresolved_ticket_count": unresolved_ticket_count,
            "payment_delay_days": payment_delay_days,
            "nps_score": nps_score,
            "contract_type": contract_type,
            "last_login_days": last_login_days,
            "onboarding_completed": onboarding_completed,
            "account_health_score": account_health_score,
            "churn": churn,
        }
    )


def save_customer_data(df: pd.DataFrame, data_dir: str | Path = "data") -> tuple[Path, Path]:
    data_path = Path(data_dir)
    raw_path = data_path / "raw" / "synthetic_customers.csv"
    processed_path = data_path / "processed" / "customer_features.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    df.to_csv(processed_path, index=False)
    return raw_path, processed_path

