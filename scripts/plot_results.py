from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """Plot metric curves from results.csv.

    Inputs:
    - results.csv path and metric name.

    Outputs:
    - Saves PNG plots per classifier in artifacts/figures.

    Internal logic:
    - Aggregates mean metric by r for each method and classifier, then plots lines.
    """
    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument("--results", type=str, default="results/results.csv")
    parser.add_argument("--out_dir", type=str, default="artifacts/figures")
    parser.add_argument("--metric", type=str, default="kappa")
    args = parser.parse_args()

    df = pd.read_csv(args.results)  # full results table
    if df.empty:
        raise RuntimeError("results.csv is empty.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric
    for classifier in sorted(df["classifier"].unique()):  # plot each classifier separately
        sub = df[df["classifier"] == classifier]
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in sorted(sub["method"].unique()):
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            grouped = msub.groupby("r")[metric].mean().reset_index()
            ax.plot(grouped["r"], grouped[metric], marker="o", label=method)
        ax.set_title(f"{metric} vs r ({classifier})")
        ax.set_xlabel("r (train fraction)")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{classifier}_{metric}_vs_r.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
