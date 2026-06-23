from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate_rag import evaluate_retrieval_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local retrieval quality.")
    parser.add_argument("--documents-path", default="data/synthetic_docs/customer_documents.csv")
    parser.add_argument("--questions-path", default="data/evaluation/rag_eval_questions.csv")
    parser.add_argument("--output-path", default="data/processed/rag_evaluation_summary.csv")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = pd.read_csv(args.documents_path)
    questions = pd.read_csv(args.questions_path)
    summary = evaluate_retrieval_dataset(documents, questions, args.output_path, top_k=args.top_k)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
