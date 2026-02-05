#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._script_utils import project_root
except ImportError:  # pragma: no cover - direct script execution
    from _script_utils import project_root

import pandas as pd

ROOT = project_root(__file__)

from src.dataio import load_processed_index
from src.eval import evaluate_saved_classifier
from src.utils import load_split_any, load_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Final-only E_test evaluation from saved classifier checkpoints")
    ap.add_argument("--input-csv", type=str, default="results/metrics/clf_cross_session.csv")
    ap.add_argument("--output-csv", type=str, default="results/metrics/clf_cross_session_test.csv")
    ap.add_argument("--force", action="store_true", help="Re-evaluate rows even if test metrics already exist")
    args = ap.parse_args()

    in_csv = ROOT / args.input_csv
    out_csv = ROOT / args.output_csv
    if not in_csv.exists():
        raise FileNotFoundError(f"Input metric csv not found: {in_csv}")

    clf_cfg = load_yaml(ROOT / "configs/clf.yaml")
    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    index_df = load_processed_index(ROOT / data_cfg["index_path"])

    df = pd.read_csv(in_csv)
    rows = []

    for _, r in df.iterrows():
        row = r.to_dict()
        run_dir = Path(str(row.get("run_dir", "")))
        if not run_dir.exists():
            print(f"[warn] Missing run_dir, skip: {run_dir}")
            continue

        if (not args.force) and str(row.get("evaluated_on", "")).lower() == "test":
            rows.append(row)
            continue

        split = load_split_any(str(row.get("split_file", "")), str(row.get("split", "")), root=ROOT)

        # Rebuild classifier config with per-row model key.
        run_cfg = {
            **clf_cfg,
            "model": {**clf_cfg.get("model", {}), "type": str(row.get("clf_model", clf_cfg.get("model", {}).get("type", "eegnet")))},
        }

        metrics = evaluate_saved_classifier(
            run_dir=run_dir,
            split=split,
            index_df=index_df,
            clf_cfg=run_cfg,
            device=str(clf_cfg.get("train", {}).get("device", "auto")),
        )

        row.update(
            {
                "acc": float(metrics.get("acc", float("nan"))),
                "bal_acc": float(metrics.get("bal_acc", float("nan"))),
                "kappa": float(metrics.get("kappa", float("nan"))),
                "f1_macro": float(metrics.get("f1_macro", float("nan"))),
                "test_acc": float(metrics.get("acc", float("nan"))),
                "test_bal_acc": float(metrics.get("bal_acc", float("nan"))),
                "test_kappa": float(metrics.get("kappa", float("nan"))),
                "test_f1_macro": float(metrics.get("f1_macro", float("nan"))),
                "evaluated_on": "test",
            }
        )
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved final test metrics -> {out_csv}")


if __name__ == "__main__":
    from src.cli_deprecated import exit_deprecated
    exit_deprecated("final-test")
