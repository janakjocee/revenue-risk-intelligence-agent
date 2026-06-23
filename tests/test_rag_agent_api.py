import pandas as pd
from fastapi.testclient import TestClient

from src.agent.workflow import run_agent
from src.api.main import app
from src.rag.retriever import load_retriever


def test_retrieval_returns_evidence_for_customer():
    retriever = load_retriever()
    results = retriever.retrieve("renewal risk support usage", customer_id="CUST-0001", top_k=3)
    assert results
    assert all(item.customer_id == "CUST-0001" for item in results)


def test_agent_returns_answer_and_evidence():
    customer = pd.read_csv("data/processed/customer_features.csv").iloc[0].to_dict()
    retriever = load_retriever()
    result = run_agent(customer, "What is the renewal risk?", retriever, include_email=True)
    assert result["answer"]
    assert result["cited_evidence"]
    assert result["email_draft"]


def test_api_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

