from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedDocument:
    document_id: str
    customer_id: str
    document_type: str
    date: str
    source: str
    risk_theme: str
    text: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class LocalTfidfRetriever:
    def __init__(self, documents: pd.DataFrame, vectorizer: TfidfVectorizer, matrix: Any) -> None:
        self.documents = documents.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def from_documents(cls, documents: pd.DataFrame) -> "LocalTfidfRetriever":
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(documents["text"].fillna(""))
        return cls(documents, vectorizer, matrix)

    def retrieve(
        self,
        query: str,
        customer_id: str | None = None,
        top_k: int = 5,
        document_type: str | None = None,
    ) -> list[RetrievedDocument]:
        mask = pd.Series([True] * len(self.documents))
        if customer_id:
            mask &= self.documents["customer_id"].eq(customer_id)
        if document_type:
            mask &= self.documents["document_type"].eq(document_type)

        candidate_indices = self.documents.index[mask].tolist()
        if not candidate_indices:
            candidate_indices = self.documents.index.tolist()

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.matrix[candidate_indices]).flatten()
        ranked = sorted(zip(candidate_indices, similarities), key=lambda item: item[1], reverse=True)[:top_k]

        results: list[RetrievedDocument] = []
        for idx, score in ranked:
            row = self.documents.iloc[idx]
            results.append(
                RetrievedDocument(
                    document_id=str(row["document_id"]),
                    customer_id=str(row["customer_id"]),
                    document_type=str(row["document_type"]),
                    date=str(row["date"]),
                    source=str(row["source"]),
                    risk_theme=str(row["risk_theme"]),
                    text=str(row["text"]),
                    score=round(float(score), 4),
                )
            )
        return results


def build_retriever(documents_path: str | Path = "data/synthetic_docs/customer_documents.csv") -> LocalTfidfRetriever:
    documents = pd.read_csv(documents_path)
    return LocalTfidfRetriever.from_documents(documents)


def save_retriever(
    documents_path: str | Path = "data/synthetic_docs/customer_documents.csv",
    output_path: str | Path = "data/processed/tfidf_retriever.joblib",
) -> Path:
    retriever = build_retriever(documents_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(retriever, path)
    return path


def load_retriever(path: str | Path = "data/processed/tfidf_retriever.joblib") -> LocalTfidfRetriever:
    return joblib.load(path)

