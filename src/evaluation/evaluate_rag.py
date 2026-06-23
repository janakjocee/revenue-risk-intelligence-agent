from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from src.rag.retriever import LocalTfidfRetriever


EVAL_QUERIES = [
    ("support_escalation", "unresolved support tickets renewal risk"),
    ("usage_decline", "usage decline last login adoption risk"),
    ("commercial_risk", "payment delay pricing renewal discount"),
    ("onboarding_gap", "onboarding incomplete time to value"),
    ("relationship_health", "low NPS customer sentiment"),
]


def evaluate_retrieval(
    documents: pd.DataFrame,
    output_path: str | Path = "data/processed/rag_evaluation_summary.csv",
) -> pd.DataFrame:
    retriever = LocalTfidfRetriever.from_documents(documents)
    records: list[dict[str, object]] = []

    for expected_theme, query in EVAL_QUERIES:
        started = perf_counter()
        results = retriever.retrieve(query, top_k=10)
        latency_ms = round((perf_counter() - started) * 1000, 2)
        retrieved_themes = [item.risk_theme for item in results]
        hits = sum(theme == expected_theme for theme in retrieved_themes)
        records.append(
            {
                "query": query,
                "expected_theme": expected_theme,
                "top_k": 10,
                "theme_hits": hits,
                "precision_at_10": round(hits / 10, 3),
                "latency_ms": latency_ms,
                "top_document_ids": ", ".join(item.document_id for item in results[:3]),
            }
        )

    summary = pd.DataFrame(records)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary

