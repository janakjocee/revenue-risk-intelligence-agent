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
        theme_docs = documents.loc[documents["risk_theme"].eq(expected_theme)]
        if theme_docs.empty:
            continue
        customer_id = str(theme_docs.iloc[0]["customer_id"])
        started = perf_counter()
        results = retriever.retrieve(query, customer_id=customer_id, top_k=10)
        latency_ms = round((perf_counter() - started) * 1000, 2)
        retrieved_themes = [item.risk_theme for item in results]
        hits = sum(theme == expected_theme for theme in retrieved_themes)
        records.append(
            {
                "query": query,
                "customer_id": customer_id,
                "expected_theme": expected_theme,
                "top_k": len(results),
                "theme_hits": hits,
                "precision_at_k": round(hits / len(results), 3) if results else 0,
                "latency_ms": latency_ms,
                "top_document_ids": ", ".join(item.document_id for item in results[:3]),
            }
        )

    summary = pd.DataFrame(records)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary
