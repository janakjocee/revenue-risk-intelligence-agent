from src.data.generate_customers import CustomerDataConfig, generate_customer_data
from src.data.generate_documents import DocumentConfig, generate_documents


def test_customer_generation_has_required_fields():
    df = generate_customer_data(CustomerDataConfig(rows=25, seed=7))
    expected = {
        "customer_id",
        "company_name",
        "contract_value",
        "monthly_recurring_revenue",
        "product_usage_score",
        "unresolved_ticket_count",
        "account_health_score",
        "churn",
    }
    assert expected.issubset(df.columns)
    assert len(df) == 25
    assert df["churn"].isin([0, 1]).all()


def test_document_generation_has_metadata():
    customers = generate_customer_data(CustomerDataConfig(rows=5, seed=7))
    docs = generate_documents(customers, DocumentConfig(seed=7, docs_per_customer=3))
    assert len(docs) == 15
    assert {"document_id", "customer_id", "document_type", "date", "source", "risk_theme", "text"}.issubset(docs.columns)

