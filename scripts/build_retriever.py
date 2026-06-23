from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.rag.retriever import save_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local TF-IDF retriever.")
    parser.add_argument("--documents-path", default="data/synthetic_docs/customer_documents.csv")
    parser.add_argument("--output-path", default="data/processed/tfidf_retriever.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = save_retriever(args.documents_path, args.output_path)
    print(f"Saved local retriever: {path}")


if __name__ == "__main__":
    main()

