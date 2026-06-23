from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model.scoring import score_customers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score synthetic customers for revenue risk.")
    parser.add_argument("--customers-path", default="data/processed/customer_features.csv")
    parser.add_argument("--model-path", default="data/processed/churn_model.joblib")
    parser.add_argument("--output-path", default="data/processed/customer_risk_scores.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    customers = pd.read_csv(args.customers_path)
    scores = score_customers(customers, args.model_path)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_path, index=False)
    print(f"Scored {len(scores)} customers")
    print(f"Risk scores: {args.output_path}")


if __name__ == "__main__":
    main()

