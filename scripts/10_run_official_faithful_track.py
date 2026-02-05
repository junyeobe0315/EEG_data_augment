#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _script_utils import project_root

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = project_root(__file__)

from src.dataio import load_processed_index
from src.train_official_faithful import train_faithful_classifier
from src.utils import ensure_dir, load_yaml, set_seed


def _make_split(index_df: pd.DataFrame, subject: int, val_ratio: float, split_seed: int, stratify: bool) -> dict:
    sub = index_df[index_df["subject"] == int(subject)].copy()
    t_df = sub[sub["session"] == "T"].copy()
    e_df = sub[sub["session"] == "E"].copy()
    ids = t_df["sample_id"].tolist()
    tr, va = train_test_split(
        ids,
        test_size=float(val_ratio),
        random_state=int(split_seed),
        stratify=t_df["label"].tolist() if bool(stratify) else None,
    )
    return {
        "protocol": "official_faithful_subject_specific",
        "subject": int(subject),
        "train_ids": list(tr),
        "val_ids": list(va),
        "test_ids": e_df["sample_id"].tolist(),
    }


def _decide_keep(compare_df: pd.DataFrame) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for _, r in compare_df.iterrows():
        out[str(r["model"])] = bool(r["pass_2pct"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run official-faithful PyTorch track (EEGNet/ATCNet).")
    ap.add_argument("--config", type=str, default="configs/official_faithful.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    cfg = load_yaml(ROOT / args.config)
    if not bool(cfg.get("enabled", True)):
        raise RuntimeError("official_faithful.yaml is disabled.")

    set_seed(int(args.seed))
    idx = load_processed_index(cfg["track_index"])

    rows = []
    for model_key in cfg["models"]:
        model_cfg = cfg["model_cfg"][model_key]
        train_cfg = cfg["train_cfg"][model_key]
        for subject in cfg["subjects"]:
            split = _make_split(
                index_df=idx,
                subject=int(subject),
                val_ratio=float(cfg["split"]["val_ratio"]),
                split_seed=int(cfg["split"]["split_seed"]),
                stratify=bool(cfg["split"]["stratify"]),
            )
            run_dir = ROOT / "runs/clf" / f"official_faithful__model-{model_key}__subj-{int(subject):02d}__seed-{int(args.seed)}"
            m = train_faithful_classifier(
                split=split,
                index_df=idx,
                model_key=model_key,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                preprocess_cfg=cfg["preprocess"],
                out_dir=run_dir,
            )
            row = {
                "seed": int(args.seed),
                "subject": int(subject),
                "model": model_key,
                "epochs": int(train_cfg["epochs"]),
                "batch_size": int(train_cfg["batch_size"]),
                **m,
            }
            rows.append(row)
            print(f"done model={model_key} subject={int(subject):02d} acc={m['acc']:.4f} kappa={m['kappa']:.4f}", flush=True)

    if not rows:
        raise RuntimeError("No runs executed.")

    suffix = f"_seed{int(args.seed)}" if not args.tag else f"_seed{int(args.seed)}_{args.tag}"
    metrics_out = ROOT / "results/metrics" / f"official_faithful{suffix}.csv"
    ensure_dir(metrics_out.parent)
    df = pd.DataFrame(rows)
    df.to_csv(metrics_out, index=False)
    print(f"saved metrics -> {metrics_out}")

    summary = df.groupby("model")[["acc", "kappa", "f1_macro"]].agg(["mean", "std"]).reset_index()
    summary_out = ROOT / "results/tables" / f"official_faithful{suffix}_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"saved summary -> {summary_out}")

    comp_rows = []
    for model_key in cfg["models"]:
        target = cfg["paper_targets"][model_key]
        ours = float(df[df["model"] == model_key]["acc"].mean() * 100.0)
        ref = float(target["value"])
        delta = ours - ref
        comp_rows.append(
            {
                "model": model_key,
                "metric": str(target["metric"]),
                "ours": ours,
                "paper": ref,
                "abs_diff": abs(delta),
                "delta": delta,
                "pass_2pct": abs(delta) <= float(cfg["acceptance"]["max_abs_diff_pct"]),
                "source": str(target.get("source", "")),
            }
        )
    comp = pd.DataFrame(comp_rows)
    comp_out = ROOT / "results/tables" / f"official_faithful{suffix}_compare.csv"
    comp.to_csv(comp_out, index=False)
    print(f"saved compare -> {comp_out}")

    keep = _decide_keep(comp)
    decision_rows = []
    for model_key in cfg["models"]:
        decision_rows.append(
            {
                "model": model_key,
                "keep_for_main": bool(keep.get(model_key, False)),
                "reason": "within_2pct_of_paper" if bool(keep.get(model_key, False)) else "over_2pct_gap_vs_paper",
            }
        )
    decision_df = pd.DataFrame(decision_rows)
    decision_out = ROOT / "results/tables" / f"official_faithful{suffix}_decision.csv"
    decision_df.to_csv(decision_out, index=False)
    print(f"saved decision -> {decision_out}")

    print(json.dumps({"kept_models": [m for m, ok in keep.items() if ok], "dropped_models": [m for m, ok in keep.items() if not ok]}))


if __name__ == "__main__":
    main()
