from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.generate_customers import CustomerDataConfig, generate_customer_data, save_customer_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic customer-success data.")
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CustomerDataConfig(rows=args.rows, seed=args.seed)
    df = generate_customer_data(config)
    raw_path, processed_path = save_customer_data(df, args.data_dir)
    print(f"Generated {len(df)} synthetic customer rows")
    print(f"Raw data: {raw_path}")
    print(f"Processed data: {processed_path}")


if __name__ == "__main__":
    main()
