from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _plot_metric_vs_r(df: pd.DataFrame, out_dir: Path, metric: str) -> None:
    """Plot metric vs low-data ratio r for each classifier and method."""
    for classifier in sorted(df["classifier"].unique()):
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


def _plot_metric_vs_alpha(gen: pd.DataFrame, out_dir: Path, metric: str) -> None:
    """Plot metric vs alpha_ratio for GenAug runs, split by classifier and QC flag."""
    if gen.empty:
        return
    for classifier in sorted(gen["classifier"].unique()):
        sub = gen[gen["classifier"] == classifier]
        for qc_on in sorted(sub["qc_on"].dropna().unique()):
            qsub = sub[sub["qc_on"] == qc_on]
            if qsub.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            for r_val in sorted(qsub["r"].dropna().unique()):
                rsub = qsub[qsub["r"] == r_val]
                grouped = rsub.groupby("alpha_ratio")[metric].mean().reset_index()
                if grouped.empty:
                    continue
                ax.plot(grouped["alpha_ratio"], grouped[metric], marker="o", label=f"r={r_val:g}")
            ax.set_title(f"{metric} vs alpha ({classifier}, qc={bool(qc_on)})")
            ax.set_xlabel("alpha_ratio (synthetic:real)")
            ax.set_ylabel(metric)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{classifier}_{metric}_vs_alpha_qc{int(bool(qc_on))}.png", dpi=150)
            plt.close(fig)


def _plot_pass_rate_vs_alpha(gen: pd.DataFrame, out_dir: Path) -> None:
    """Plot QC pass_rate vs alpha_ratio for GenAug runs."""
    if gen.empty or "pass_rate" not in gen.columns:
        return
    for classifier in sorted(gen["classifier"].unique()):
        sub = gen[gen["classifier"] == classifier]
        for qc_on in sorted(sub["qc_on"].dropna().unique()):
            qsub = sub[sub["qc_on"] == qc_on]
            if qsub.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            for r_val in sorted(qsub["r"].dropna().unique()):
                rsub = qsub[qsub["r"] == r_val]
                grouped = rsub.groupby("alpha_ratio")["pass_rate"].mean().reset_index()
                if grouped.empty:
                    continue
                ax.plot(grouped["alpha_ratio"], grouped["pass_rate"], marker="o", label=f"r={r_val:g}")
            ax.set_title(f"pass_rate vs alpha ({classifier}, qc={bool(qc_on)})")
            ax.set_xlabel("alpha_ratio (synthetic:real)")
            ax.set_ylabel("pass_rate")
            ax.set_ylim(0.0, 1.0)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{classifier}_pass_rate_vs_alpha_qc{int(bool(qc_on))}.png", dpi=150)
            plt.close(fig)


def main() -> None:
    """Plot metric curves from results.csv.

    Inputs:
    - results.csv path and metric name.

    Outputs:
    - Saves PNG plots in artifacts/figures:
      - metric vs r
      - metric vs alpha_ratio (GenAug)
      - pass_rate vs alpha_ratio (GenAug)

    Internal logic:
    - Aggregates means over runs and writes analysis-friendly figures.
    """
    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument("--results", type=str, default="results/results.csv")
    parser.add_argument("--out_dir", type=str, default="artifacts/figures")
    parser.add_argument("--metric", type=str, default="kappa")
    parser.add_argument("--config_pack", type=str, default="all")
    args = parser.parse_args()

    df = pd.read_csv(args.results)  # full results table
    if df.empty:
        raise RuntimeError("results.csv is empty.")
    if args.config_pack != "all" and "config_pack" in df.columns:
        df = df[df["config_pack"] == str(args.config_pack)]
    if df.empty:
        raise RuntimeError("No rows remain after filtering.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric
    if metric not in df.columns:
        raise KeyError(f"Metric {metric} not found in results.csv")

    _plot_metric_vs_r(df, out_dir, metric)
    gen = df[df["method"] == "GenAug"].copy()
    _plot_metric_vs_alpha(gen, out_dir, metric)
    _plot_pass_rate_vs_alpha(gen, out_dir)


if __name__ == "__main__":
    main()
