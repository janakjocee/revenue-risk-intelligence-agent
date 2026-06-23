from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.briefing import render_account_brief, save_account_brief
from src.agent.workflow import run_agent
from src.model.scoring import score_customer
from src.model.simulator import simulate_customer_scenario
from src.observability.logger import append_feedback, feedback_summary, read_agent_logs
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
    question: str = Field(min_length=3, max_length=1200)
    include_email: bool = False


class FeedbackRequest(BaseModel):
    run_id: str
    customer_id: str
    rating: int = Field(ge=1, le=5)
    feedback_reason: str = Field(default="", max_length=1200)
    question: str = Field(default="", max_length=1200)
    risk_band: str = Field(default="", max_length=20)


class AccountBriefRequest(BaseModel):
    customer_id: str
    question: str = Field(
        default="Prepare an account brief with renewal risk, evidence, recommendations, and email draft.",
        min_length=3,
        max_length=1200,
    )


class WhatIfRequest(BaseModel):
    customer_id: str
    unresolved_ticket_count: Optional[int] = Field(default=None, ge=0, le=50)
    product_usage_score: Optional[float] = Field(default=None, ge=0, le=100)
    payment_delay_days: Optional[float] = Field(default=None, ge=0, le=120)
    nps_score: Optional[float] = Field(default=None, ge=-100, le=100)
    account_health_score: Optional[float] = Field(default=None, ge=0, le=100)
    contract_type: Optional[str] = Field(default=None, pattern="^(monthly|annual|multi-year)$")


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


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, Any]:
    return append_feedback(request.model_dump())


@app.post("/account-brief")
def account_brief(request: AccountBriefRequest) -> dict[str, Any]:
    customer = get_customer(request.customer_id)
    retriever = load_retriever(RETRIEVER_PATH)
    result = run_agent(
        customer,
        question=request.question,
        retriever=retriever,
        model_path=str(MODEL_PATH),
        include_email=True,
    )
    path = save_account_brief(customer, result)
    return {"customer_id": request.customer_id, "brief_path": str(path), "markdown": render_account_brief(customer, result)}


@app.post("/what-if")
def what_if(request: WhatIfRequest) -> dict[str, Any]:
    customer = get_customer(request.customer_id)
    adjustments = {key: value for key, value in request.model_dump().items() if key != "customer_id" and value is not None}
    return simulate_customer_scenario(customer, adjustments, model_path=MODEL_PATH)


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
        "feedback": feedback_summary(),
    }
