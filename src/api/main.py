from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent.workflow import run_agent
from src.model.scoring import score_customer
from src.observability.logger import read_agent_logs
from src.rag.retriever import load_retriever


DATA_PATH = Path("data/processed/customer_features.csv")
SCORES_PATH = Path("data/processed/customer_risk_scores.csv")
MODEL_PATH = Path("data/processed/churn_model.joblib")
RETRIEVER_PATH = Path("data/processed/tfidf_retriever.joblib")
METRICS_PATH = Path("data/processed/model_metrics.json")

app = FastAPI(
    title="Revenue Risk Intelligence Agent",
    description="Synthetic customer-success revenue risk decision-support API.",
    version="0.1.0",
)


class ScoreRequest(BaseModel):
    customer_id: str


class AskRequest(BaseModel):
    customer_id: str
    question: str
    include_email: bool = False


def load_customers() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def get_customer(customer_id: str) -> dict[str, Any]:
    customers = load_customers()
    match = customers.loc[customers["customer_id"].eq(customer_id)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown customer_id: {customer_id}")
    return match.iloc[0].to_dict()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "revenue-risk-intelligence-agent"}


@app.get("/customers")
def customers() -> list[dict[str, Any]]:
    df = load_customers()
    columns = ["customer_id", "company_name", "industry", "monthly_recurring_revenue", "account_health_score"]
    if SCORES_PATH.exists():
        scores = pd.read_csv(SCORES_PATH)
        df = df.merge(scores[["customer_id", "risk_band", "churn_probability", "revenue_at_risk"]], on="customer_id", how="left")
        columns += ["risk_band", "churn_probability", "revenue_at_risk"]
    return df[columns].to_dict(orient="records")


@app.get("/customers/{customer_id}")
def customer_detail(customer_id: str) -> dict[str, Any]:
    return get_customer(customer_id)


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    customer = get_customer(request.customer_id)
    return score_customer(customer, MODEL_PATH)


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    customer = get_customer(request.customer_id)
    retriever = load_retriever(RETRIEVER_PATH)
    return run_agent(
        customer,
        question=request.question,
        retriever=retriever,
        model_path=str(MODEL_PATH),
        include_email=request.include_email,
    )


@app.get("/observability/logs")
def observability_logs() -> list[dict[str, Any]]:
    return read_agent_logs()


@app.get("/evaluation/summary")
def evaluation_summary() -> dict[str, Any]:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    logs = read_agent_logs()
    avg_latency = round(sum(item["latency_ms"] for item in logs) / len(logs), 2) if logs else None
    return {
        "model_metrics": metrics,
        "agent_runs": len(logs),
        "average_latency_ms": avg_latency,
        "retrieval_backend": "local_tfidf",
        "data_scope": "synthetic demo data",
    }

