from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate_rag import evaluate_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local retrieval quality.")
    parser.add_argument("--documents-path", default="data/synthetic_docs/customer_documents.csv")
    parser.add_argument("--output-path", default="data/processed/rag_evaluation_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = pd.read_csv(args.documents_path)
    summary = evaluate_retrieval(documents, args.output_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

