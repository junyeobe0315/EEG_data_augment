from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize results.csv")
    parser.add_argument("--results", type=str, default="results/results.csv")
    parser.add_argument("--out", type=str, default="results/summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    if df.empty:
        raise RuntimeError("results.csv is empty.")

    group_cols = ["method", "classifier", "r", "alpha_ratio", "qc_on", "generator"]
    metrics = ["acc", "kappa", "macro_f1"]

    summary = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
