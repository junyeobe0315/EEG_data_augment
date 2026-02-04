from __future__ import annotations

from pathlib import Path

import pandas as pd


def aggregate_metrics(metrics_dir: str | Path, out_csv: str | Path) -> pd.DataFrame:
    metrics_dir = Path(metrics_dir)
    files = sorted(metrics_dir.glob("clf_*.csv"))
    if not files:
        raise RuntimeError(f"No metric files found under {metrics_dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    candidate_cols = ["protocol", "subject", "p", "clf_model", "gen_model", "mode", "synth_ratio", "qc_on"]
    group_cols = [c for c in candidate_cols if c in df.columns]

    agg = (
        df.groupby(group_cols, dropna=False)[["acc", "kappa", "f1_macro"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    return agg
