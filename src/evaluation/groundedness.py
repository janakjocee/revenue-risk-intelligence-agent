from __future__ import annotations

import re
from typing import Any


UNSUPPORTED_PATTERNS = [
    "guaranteed",
    "definitely churn",
    "legal",
    "compliance breach",
    "bankrupt",
    "fraud",
]


def evaluate_groundedness(agent_result: dict[str, Any]) -> dict[str, Any]:
    answer = str(agent_result.get("answer", ""))
    caveats = str(agent_result.get("caveats", ""))
    evidence = agent_result.get("cited_evidence", []) or []
    evidence_text = " ".join(str(item.get("text", "")) for item in evidence).lower()
    answer_lower = answer.lower()

    cited_document_ids = [str(item.get("document_id", "")) for item in evidence if item.get("document_id")]
    cites_evidence = bool(cited_document_ids) and bool(evidence_text)
    driver_terms = [term.lower() for term in agent_result.get("risk_score", {}).get("top_risk_drivers", [])]
    supported_driver_count = sum(
        1
        for term in driver_terms
        if any(token in evidence_text for token in re.findall(r"[a-zA-Z]+", term))
    )
    unsupported_mentions = [pattern for pattern in UNSUPPORTED_PATTERNS if pattern in answer_lower]
    weak_evidence = not evidence or max(float(item.get("score", 0)) for item in evidence) <= 0

    score_parts = [
        1.0 if cites_evidence else 0.0,
        1.0 if supported_driver_count > 0 else 0.0,
        1.0 if not unsupported_mentions else 0.0,
        1.0 if (not weak_evidence or "weak" in caveats.lower() or "validate" in caveats.lower()) else 0.0,
    ]
    return {
        "cites_retrieved_evidence": cites_evidence,
        "supported_driver_count": supported_driver_count,
        "unsupported_mentions": unsupported_mentions,
        "includes_weak_evidence_caveat": not weak_evidence or "weak" in caveats.lower() or "validate" in caveats.lower(),
        "groundedness_score": round(sum(score_parts) / len(score_parts), 3),
        "evidence_document_ids": cited_document_ids,
    }

