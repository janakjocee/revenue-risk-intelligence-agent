from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.generate_documents import DocumentConfig, generate_documents, save_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic RAG documents.")
    parser.add_argument("--customers-path", default="data/processed/customer_features.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--docs-per-customer", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    customers = pd.read_csv(args.customers_path)
    docs = generate_documents(customers, DocumentConfig(seed=args.seed, docs_per_customer=args.docs_per_customer))
    output_path = save_documents(docs, args.data_dir)
    print(f"Generated {len(docs)} synthetic documents")
    print(f"Documents: {output_path}")


if __name__ == "__main__":
    main()

