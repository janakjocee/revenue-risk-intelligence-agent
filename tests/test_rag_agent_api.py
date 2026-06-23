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


def test_agent_returns_answer_and_evidence(tmp_path):
    customer = pd.read_csv("data/processed/customer_features.csv").iloc[0].to_dict()
    retriever = load_retriever()
    result = run_agent(
        customer,
        "What is the renewal risk?",
        retriever,
        include_email=True,
        log_path=str(tmp_path / "agent_runs.jsonl"),
    )
    assert result["answer"]
    assert result["run_id"]
    assert result["cited_evidence"]
    assert result["email_draft"]


def test_api_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_feedback_endpoint(monkeypatch):
    from src.api import main as api_main

    monkeypatch.setattr(api_main, "append_feedback", lambda payload: {"timestamp": "test", **payload})
    client = TestClient(app)
    response = client.post(
        "/feedback",
        json={
            "run_id": "test-run",
            "customer_id": "CUST-0001",
            "rating": 5,
            "feedback_reason": "Clear and useful",
            "question": "Why is this customer at risk?",
            "risk_band": "medium",
        },
    )
    assert response.status_code == 200
    assert response.json()["rating"] == 5


def test_api_what_if_endpoint():
    client = TestClient(app)
    response = client.post(
        "/what-if",
        json={"customer_id": "CUST-0001", "product_usage_score": 85, "unresolved_ticket_count": 0},
    )
    assert response.status_code == 200
    assert "baseline" in response.json()
    assert "scenario" in response.json()
