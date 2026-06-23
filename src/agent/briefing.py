from __future__ import annotations

from pathlib import Path
from typing import Any


def render_account_brief(customer: dict[str, Any], agent_result: dict[str, Any]) -> str:
    score = agent_result["risk_score"]
    evidence_lines = "\n".join(
        f"- `{item['document_id']}` ({item['document_type']}, score {item['score']}): {item['text']}"
        for item in agent_result.get("cited_evidence", [])
    )
    action_lines = "\n".join(
        f"- **{item['action']}**: {item['why']} Owner: {item['owner']}."
        for item in agent_result.get("recommended_actions", [])
    )
    drivers = "\n".join(f"- {driver}" for driver in score.get("top_risk_drivers", []))
    return f"""# Account Brief: {customer['company_name']}

Synthetic demo data. This brief is generated for customer-success decision support.

## Customer Profile

- Customer ID: `{customer['customer_id']}`
- Industry: {customer['industry']}
- Contract type: {customer['contract_type']}
- Monthly recurring revenue: GBP {customer['monthly_recurring_revenue']:,.2f}
- Tenure: {customer['tenure_months']} months
- Account health score: {customer['account_health_score']}

## Risk Summary

- Churn probability: {score['churn_probability']:.1%}
- Risk band: {score['risk_band'].upper()}
- Revenue at risk: GBP {score['revenue_at_risk']:,.0f}

## Top Risk Drivers

{drivers}

## Recommended Actions

{action_lines}

## Retrieved Evidence

{evidence_lines}

## Email Draft

```text
{agent_result.get('email_draft') or 'Email draft was not requested.'}
```

## Caveats

{agent_result.get('caveats')}
"""


def save_account_brief(
    customer: dict[str, Any],
    agent_result: dict[str, Any],
    output_dir: str | Path = "data/processed/account_briefs",
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / f"{customer['customer_id']}_account_brief.md"
    output_path.write_text(render_account_brief(customer, agent_result), encoding="utf-8")
    return output_path

