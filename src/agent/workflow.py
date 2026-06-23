from __future__ import annotations

from time import perf_counter
from typing import Any

from src.agent.llm import BaseLLMProvider, get_llm_provider
from src.agent.recommendations import recommend_actions
from src.evaluation.groundedness import evaluate_groundedness
from src.model.scoring import score_customer
from src.observability.logger import log_agent_run
from src.rag.retriever import LocalTfidfRetriever


def _email_draft(customer: dict[str, Any], actions: list[dict[str, str]]) -> str:
    primary_action = actions[0]["action"].lower()
    return (
        f"Subject: Aligning on next steps for {customer['company_name']}\n\n"
        f"Hi {customer['company_name']} team,\n\n"
        "I wanted to follow up with a practical next step based on our latest account review. "
        f"I suggest we {primary_action} so we can remove blockers before the renewal conversation. "
        "Would you be open to a short working session this week?\n\n"
        "Best,\nCustomer Success"
    )


def run_agent(
    customer: dict[str, Any],
    question: str,
    retriever: LocalTfidfRetriever,
    model_path: str = "data/processed/churn_model.joblib",
    include_email: bool = False,
    log_path: str = "data/processed/agent_runs.jsonl",
    llm_provider: BaseLLMProvider | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    score = score_customer(customer, model_path=model_path)
    retrieval_query = f"{question} {' '.join(score['top_risk_drivers'])} {score['recommended_action_category']}"
    evidence = [doc.as_dict() for doc in retriever.retrieve(retrieval_query, customer_id=customer["customer_id"], top_k=5)]
    actions = recommend_actions(customer, score)

    if evidence and max(item["score"] for item in evidence) > 0:
        evidence_summary = " ".join(item["text"] for item in evidence[:2])
        caveat = "Answer is grounded in retrieved synthetic account evidence."
    else:
        evidence_summary = "No strong matching evidence was retrieved for this customer."
        caveat = "Evidence is weak; validate with the account team before acting."

    explanation = (
        f"{customer['company_name']} is classified as {score['risk_band']} risk with "
        f"{score['churn_probability']:.1%} churn probability and "
        f"GBP {score['revenue_at_risk']:,.0f} estimated revenue at risk. "
        f"Main drivers: {', '.join(score['top_risk_drivers'])}."
    )
    answer = (
        f"{explanation} Recommended next step: {actions[0]['action']}. "
        f"Supporting evidence: {evidence_summary}"
    )
    provider = llm_provider or get_llm_provider()
    provider_note = provider.complete(
        "Summarise the account risk using only the structured score and supplied evidence. "
        f"Question: {question}\nRisk explanation: {explanation}\nEvidence: {evidence_summary}"
    )

    result = {
        "customer_id": customer["customer_id"],
        "question": question,
        "answer": answer,
        "risk_score": score,
        "risk_explanation": explanation,
        "recommended_actions": actions,
        "cited_evidence": evidence,
        "caveats": caveat,
        "email_draft": _email_draft(customer, actions) if include_email else None,
        "llm_provider_note": provider_note,
    }
    result["groundedness_evaluation"] = evaluate_groundedness(result)

    latency_ms = round((perf_counter() - started) * 1000, 2)
    log_agent_run(
        {
            "customer_id": customer["customer_id"],
            "user_question": question,
            "retrieved_document_ids": [item["document_id"] for item in evidence],
            "retrieved_document_types": [item["document_type"] for item in evidence],
            "risk_band": score["risk_band"],
            "latency_ms": latency_ms,
            "response_length": len(answer),
            "feedback": None,
            "llm_provider": provider.__class__.__name__,
        },
        log_path=log_path,
    )
    result["latency_ms"] = latency_ms
    return result
