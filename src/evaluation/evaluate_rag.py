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


def _coverage(text: str, expected_points: str) -> float:
    points = [point.strip().lower() for point in expected_points.split(";") if point.strip()]
    if not points:
        return 0.0
    lowered = text.lower()
    hits = sum(point in lowered for point in points)
    return round(hits / len(points), 3)


def evaluate_retrieval_dataset(
    documents: pd.DataFrame,
    questions: pd.DataFrame,
    output_path: str | Path = "data/processed/rag_evaluation_summary.csv",
    top_k: int = 5,
) -> pd.DataFrame:
    retriever = LocalTfidfRetriever.from_documents(documents)
    records: list[dict[str, object]] = []

    for _, question in questions.iterrows():
        expected_theme = str(question["expected_risk_theme"])
        expected_doc_type = str(question["expected_document_type"])
        customer_id = str(question["customer_id"])
        query = str(question["question"])
        started = perf_counter()
        results = retriever.retrieve(query, customer_id=customer_id, top_k=top_k)
        latency_ms = round((perf_counter() - started) * 1000, 2)
        retrieved_themes = [item.risk_theme for item in results]
        retrieved_types = [item.document_type for item in results]
        theme_hits = sum(theme == expected_theme for theme in retrieved_themes)
        type_hits = sum(doc_type == expected_doc_type for doc_type in retrieved_types)
        evidence_text = " ".join(item.text for item in results)
        evidence_coverage = _coverage(evidence_text, str(question["expected_answer_points"]))
        records.append(
            {
                "question_id": question["question_id"],
                "query": query,
                "customer_id": customer_id,
                "expected_theme": expected_theme,
                "expected_document_type": expected_doc_type,
                "top_k": top_k,
                "retrieved_k": len(results),
                "theme_hits": theme_hits,
                "document_type_hits": type_hits,
                "precision_at_k": round(theme_hits / len(results), 3) if results else 0,
                "recall_at_k": 1.0 if theme_hits > 0 else 0.0,
                "expected_theme_match": theme_hits > 0,
                "expected_document_type_match": type_hits > 0,
                "latency_ms": latency_ms,
                "groundedness_heuristic": evidence_coverage,
                "evidence_coverage_score": evidence_coverage,
                "top_document_ids": ", ".join(item.document_id for item in results[:3]),
            }
        )

    summary = pd.DataFrame(records)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary


def evaluate_retrieval(
    documents: pd.DataFrame,
    output_path: str | Path = "data/processed/rag_evaluation_summary.csv",
) -> pd.DataFrame:
    rows = []
    for idx, (theme, query) in enumerate(EVAL_QUERIES, start=1):
        theme_docs = documents.loc[documents["risk_theme"].eq(theme)]
        if theme_docs.empty:
            continue
        rows.append(
            {
                "question_id": f"LEGACY-{idx}",
                "customer_id": str(theme_docs.iloc[0]["customer_id"]),
                "question": query,
                "expected_risk_theme": theme,
                "expected_document_type": "renewal_risk_note",
                "expected_answer_points": theme.replace("_", " "),
            }
        )
    return evaluate_retrieval_dataset(documents, pd.DataFrame(rows), output_path)
