from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def shannon_entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log2(values)))


def knn_neighborhood_stability(neighborhood_a: list[int], neighborhood_b: list[int]) -> float:
    a = set(neighborhood_a)
    b = set(neighborhood_b)
    if not a and not b:
        return 1.0
    union = len(a | b) or 1
    return float(len(a & b) / union)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute cluster stability diagnostics for uncertainty-aware graph analysis")
    parser.add_argument("--labels", type=Path, help="Path to a table containing repeated human labels")
    parser.add_argument("--output", type=Path, help="Optional output path for stability metrics")
    args = parser.parse_args()

    if args.labels is None:
        print("Placeholder script: add label data and implement stability aggregation here.")
        return

    frame = pd.read_parquet(args.labels) if args.labels.suffix == ".parquet" else pd.read_csv(args.labels)
    print(f"Loaded {len(frame)} rows for future stability analysis.")


if __name__ == "__main__":
    main()