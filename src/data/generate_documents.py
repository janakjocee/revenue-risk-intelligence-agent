from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


DOCUMENT_TYPES = [
    "support_ticket_summary",
    "customer_success_call_note",
    "pricing_policy_note",
    "retention_playbook",
    "contract_note",
    "onboarding_note",
    "product_usage_summary",
    "renewal_risk_note",
]


@dataclass(frozen=True)
class DocumentConfig:
    seed: int = 42
    docs_per_customer: int = 4


def infer_risk_theme(customer: pd.Series) -> str:
    if customer["unresolved_ticket_count"] >= 3:
        return "support_escalation"
    if customer["product_usage_score"] < 45 or customer["last_login_days"] > 30:
        return "usage_decline"
    if customer["payment_delay_days"] > 21:
        return "commercial_risk"
    if not bool(customer["onboarding_completed"]):
        return "onboarding_gap"
    if customer["nps_score"] < 0:
        return "relationship_health"
    return "stable_growth"


def _document_text(customer: pd.Series, document_type: str, risk_theme: str) -> str:
    company = customer["company_name"]
    usage = customer["product_usage_score"]
    tickets = customer["unresolved_ticket_count"]
    delay = customer["payment_delay_days"]
    nps = customer["nps_score"]

    templates = {
        "support_ticket_summary": (
            f"Support summary for {company}: {tickets} unresolved tickets remain open. "
            f"The main theme is {risk_theme}. Customer sentiment should be monitored before renewal."
        ),
        "customer_success_call_note": (
            f"Customer-success call note for {company}: stakeholder feedback shows NPS {nps}. "
            f"The team discussed adoption blockers, executive alignment, and renewal confidence."
        ),
        "pricing_policy_note": (
            f"Pricing note for {company}: contract type is {customer['contract_type']} with monthly revenue "
            f"of {customer['monthly_recurring_revenue']:.2f}. Discount approval requires evidence of retention risk."
        ),
        "retention_playbook": (
            f"Retention playbook for {risk_theme}: recommended motions include executive check-in, "
            f"support review, adoption coaching, and a commercially justified renewal incentive."
        ),
        "contract_note": (
            f"Contract note for {company}: annual contract value is {customer['contract_value']:.0f}. "
            f"Payment delay is {delay:.0f} days and tenure is {customer['tenure_months']} months."
        ),
        "onboarding_note": (
            f"Onboarding note for {company}: onboarding completed is {customer['onboarding_completed']}. "
            f"Low completion can reduce time-to-value and increase churn risk."
        ),
        "product_usage_summary": (
            f"Product usage summary for {company}: usage score is {usage:.1f} and last login was "
            f"{customer['last_login_days']:.0f} days ago. Adoption trend is linked to renewal health."
        ),
        "renewal_risk_note": (
            f"Renewal risk note for {company}: account health score is {customer['account_health_score']:.1f}. "
            f"Risk theme is {risk_theme}; the account team should validate evidence before committing action."
        ),
    }
    return templates[document_type]


def generate_documents(customers: pd.DataFrame, config: DocumentConfig = DocumentConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    base_date = date(2026, 6, 1)
    records: list[dict[str, object]] = []

    for _, customer in customers.iterrows():
        risk_theme = infer_risk_theme(customer)
        selected_types = list(rng.choice(DOCUMENT_TYPES, size=config.docs_per_customer, replace=False))
        if "renewal_risk_note" not in selected_types:
            selected_types[0] = "renewal_risk_note"

        for doc_type in selected_types:
            doc_date = base_date - timedelta(days=int(rng.integers(1, 120)))
            document_id = f"DOC-{customer['customer_id']}-{len(records) + 1:05d}"
            records.append(
                {
                    "document_id": document_id,
                    "customer_id": customer["customer_id"],
                    "document_type": doc_type,
                    "date": doc_date.isoformat(),
                    "source": "synthetic_demo_generator",
                    "risk_theme": risk_theme,
                    "text": _document_text(customer, doc_type, risk_theme),
                }
            )

    return pd.DataFrame(records)


def save_documents(documents: pd.DataFrame, data_dir: str | Path = "data") -> Path:
    output_path = Path(data_dir) / "synthetic_docs" / "customer_documents.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents.to_csv(output_path, index=False)
    return output_path

