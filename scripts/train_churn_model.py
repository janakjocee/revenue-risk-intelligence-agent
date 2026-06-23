from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model.train import train_churn_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the synthetic churn model.")
    parser.add_argument("--customers-path", default="data/processed/customer_features.csv")
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    customers = pd.read_csv(args.customers_path)
    result = train_churn_model(customers, args.output_dir)
    print("Model training complete")
    print(result["metrics"])


if __name__ == "__main__":
    main()

