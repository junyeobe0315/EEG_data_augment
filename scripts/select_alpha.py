from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml


def main() -> None:
    """Select alpha* per r by maximizing a validation metric.

    Inputs:
    - results.csv path and metric name (e.g., val_kappa).
    - Optional qc_on flag to filter rows.

    Outputs:
    - Writes alpha_star.json mapping r -> best alpha_ratio.

    Internal logic:
    - Filters GenAug rows for screening classifier and picks alpha with best mean metric.
    """
    parser = argparse.ArgumentParser(description="Select alpha* from validation results")
    parser.add_argument("--results", type=str, default="results/results.csv")
    parser.add_argument("--metric", type=str, default="val_kappa")
    parser.add_argument("--config_pack", type=str, default="base")
    parser.add_argument("--qc_on", action="store_true")
    parser.add_argument("--out", type=str, default="./artifacts/alpha_star.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    exp_cfg = load_yaml("configs/experiment_grid.yaml", overrides=args.override)  # grid config
    screening_classifier = exp_cfg.get("stage", {}).get("screening_classifier", "eegnet")  # EEGNet proxy

    df = pd.read_csv(args.results)  # results table
    df = df[(df["method"] == "GenAug") & (df["classifier"] == screening_classifier)]
    if "config_pack" in df.columns:
        df = df[df["config_pack"] == str(args.config_pack)]
    if args.qc_on:
        df = df[df["qc_on"] == True]

    metric = args.metric
    if metric not in df.columns:
        raise KeyError(f"Metric {metric} not found in results.csv")

    alpha_star = {}
    for r_val in sorted(df["r"].unique()):
        sub = df[df["r"] == r_val]
        if sub.empty:
            continue
        grouped = sub.groupby("alpha_ratio")[metric].mean()
        best_alpha = float(grouped.idxmax())
        alpha_star[str(r_val)] = best_alpha

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(alpha_star, f, indent=2)


if __name__ == "__main__":
    main()
