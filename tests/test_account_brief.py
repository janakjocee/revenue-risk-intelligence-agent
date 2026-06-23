from src.agent.briefing import render_account_brief


def test_render_account_brief_contains_business_sections():
    customer = {
        "customer_id": "CUST-TEST",
        "company_name": "Example Co",
        "industry": "SaaS",
        "contract_type": "annual",
        "monthly_recurring_revenue": 1000.0,
        "tenure_months": 12,
        "account_health_score": 72.5,
    }
    result = {
        "risk_score": {
            "churn_probability": 0.42,
            "risk_band": "medium",
            "revenue_at_risk": 5040,
            "top_risk_drivers": ["Low product usage"],
        },
        "recommended_actions": [{"action": "Investigate product usage decline", "why": "Usage is low.", "owner": "CSM"}],
        "cited_evidence": [{"document_id": "DOC-1", "document_type": "usage", "score": 0.8, "text": "Usage declined."}],
        "email_draft": "Subject: Next steps",
        "caveats": "Synthetic evidence.",
    }
    brief = render_account_brief(customer, result)
    assert "# Account Brief: Example Co" in brief
    assert "## Risk Summary" in brief
    assert "## Retrieved Evidence" in brief
    assert "Subject: Next steps" in brief

