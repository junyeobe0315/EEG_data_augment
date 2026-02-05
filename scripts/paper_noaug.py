#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

try:
    from scripts._script_utils import project_root
except ImportError:  # pragma: no cover - direct script execution
    from _script_utils import project_root

import pandas as pd

ROOT = project_root(__file__)

from src.config_utils import apply_paper_preset
from src.dataio import load_processed_index
from src.models_clf import normalize_classifier_type
from src.train_clf import train_classifier
from src.utils import ensure_dir, load_json, load_yaml, make_exp_id, p_tag, set_seed, split_file_path


def main() -> None:
    ap = argparse.ArgumentParser(description="No-augmentation baseline using paper hyperparameter presets.")
    ap.add_argument("--subject", type=int, default=1, help="Subject id in [1..9]")
    ap.add_argument("--seed", type=int, default=0, help="Split seed")
    ap.add_argument("--p", type=float, default=1.0, help="Low-data fraction tag (e.g., 1.0, 0.1)")
    ap.add_argument(
        "--models",
        type=str,
        default="eegnet,svm,eeg_conformer,atcnet,ctnet",
        help="Comma-separated classifier list",
    )
    ap.add_argument("--epoch-cap", type=int, default=0, help="Optional cap for fast smoke run (0 = no cap).")
    ap.add_argument("--force", action="store_true", help="Re-train even if outputs already exist.")
    args = ap.parse_args()

    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    pp_cfg = load_yaml(ROOT / "configs/preprocess.yaml")
    clf_cfg = load_yaml(ROOT / "configs/clf.yaml")

    split_path = split_file_path(ROOT, subject=args.subject, seed=args.seed, p=args.p)
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    split = load_json(split_path)
    index_df = load_processed_index(data_cfg["index_path"])
    set_seed(int(args.seed))

    model_list = [normalize_classifier_type(x.strip()) for x in args.models.split(",") if x.strip()]
    rows = []
    for model_type in model_list:
        cfg = apply_paper_preset(clf_cfg, model_type=model_type, epoch_cap=args.epoch_cap if args.epoch_cap > 0 else None)
        cfg["augmentation"]["modes"] = ["none"]
        run_id = make_exp_id(
            "paper_none",
            subject=int(args.subject),
            seed=int(args.seed),
            p=float(args.p),
            clf=model_type,
        )
        run_dir = ROOT / "runs/clf" / run_id
        metrics_path = run_dir / "metrics.json"
        if (not args.force) and metrics_path.exists():
            metrics = load_json(metrics_path)
            print(f"[skip] {run_dir.name} (metrics.json exists)")
        else:
            metrics = train_classifier(split, index_df, cfg, pp_cfg, run_dir, mode="none", ratio=0.0, evaluate_test=True)
        rows.append(
            {
                "subject": int(args.subject),
                "seed": int(args.seed),
                "p": float(args.p),
                "clf_model": model_type,
                "epochs_used": int(cfg["train"]["epochs"]),
                "batch_size_used": int(cfg["train"].get("batch_size", 0)),
                "lr_used": float(cfg["train"]["lr"]),
                **metrics,
            }
        )
        print(
            f"done clf={model_type} acc={metrics['acc']:.4f} kappa={metrics['kappa']:.4f} "
            f"(epochs={cfg['train']['epochs']}, batch={cfg['train'].get('batch_size', 0)})"
        )

    out_dir = ensure_dir(ROOT / "results/metrics")
    tag = p_tag(args.p)
    out_csv = out_dir / f"paper_noaug_subject_{args.subject:02d}_seed_{args.seed}_p_{tag}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    from src.cli_deprecated import exit_deprecated
    exit_deprecated("paper-noaug")
