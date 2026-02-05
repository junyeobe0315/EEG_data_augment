#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from _script_utils import project_root

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = project_root(__file__)

from src.config_utils import apply_paper_preset
from src.dataio import load_processed_index
from src.models_clf import normalize_classifier_type
from src.train_clf import train_classifier
from src.utils import ensure_dir, load_json, load_yaml, make_exp_id, set_seed


def _split_subject_index(
    index_df: pd.DataFrame,
    subject: int,
    split_seed: int,
    val_ratio: float,
    stratify: bool,
) -> dict:
    sub = index_df[index_df["subject"] == int(subject)].copy()
    t_df = sub[sub["session"] == "T"].copy()
    e_df = sub[sub["session"] == "E"].copy()
    if len(t_df) == 0 or len(e_df) == 0:
        raise RuntimeError(f"Missing T/E for subject={subject}")

    train_ids_all = t_df["sample_id"].tolist()
    if float(val_ratio) > 0.0:
        tr_ids, va_ids = train_test_split(
            train_ids_all,
            test_size=float(val_ratio),
            random_state=int(split_seed),
            stratify=t_df["label"].tolist() if bool(stratify) else None,
        )
    else:
        tr_ids = train_ids_all
        # EEG-Conformer 공개 코드는 test를 보면서 best를 고르는 형태여서, 동일 트랙에서는 val=test를 허용한다.
        va_ids = e_df["sample_id"].tolist()
    te_ids = e_df["sample_id"].tolist()
    return {
        "protocol": "paper_subject_specific",
        "subject": int(subject),
        "seed": int(split_seed),
        "train_ids": list(tr_ids),
        "val_ids": list(va_ids),
        "test_ids": list(te_ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run paper-faithful track (separate from main protocol).")
    ap.add_argument("--models", type=str, default="eegnet,svm,eeg_conformer,atcnet,ctnet")
    ap.add_argument("--subjects", type=str, default="1,2,3,4,5,6,7,8,9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch-cap", type=int, default=0, help="Fast debug cap. 0 means full paper epochs.")
    ap.add_argument("--tag", type=str, default="", help="Optional suffix tag for output filenames.")
    ap.add_argument("--force", action="store_true", help="Re-train even if outputs already exist.")
    args = ap.parse_args()

    clf_cfg = load_yaml(ROOT / "configs/clf.yaml")
    pp_cfg = load_yaml(ROOT / "configs/preprocess.yaml")
    paper_cfg = load_yaml(ROOT / "configs/paper_track.yaml")
    max_diff = float(paper_cfg.get("acceptance", {}).get("max_abs_diff_pct", 3.0))

    model_list = [normalize_classifier_type(x.strip()) for x in args.models.split(",") if x.strip()]
    subjects = [int(x.strip()) for x in args.subjects.split(",") if x.strip()]
    seed = int(args.seed)
    set_seed(seed)

    rows = []
    for model_type in model_list:
        track_name = paper_cfg["model_track_map"][model_type]
        index_path = ROOT / "data/paper_track" / track_name / "index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing paper-track index: {index_path}. Run `python main.py prepare-paper-track-data` first.")
        index_df = load_processed_index(index_path)

        val_ratio = float(paper_cfg["model_val_ratio"].get(model_type, 0.2))
        split_seed = int(paper_cfg.get("model_split_seed", {}).get(model_type, seed))
        split_stratify = bool(paper_cfg.get("model_split_stratify", {}).get(model_type, True))
        aug_mode = str(paper_cfg["model_aug_mode"].get(model_type, "none"))
        norm_mode = str(paper_cfg.get("model_norm_mode", {}).get(model_type, pp_cfg["normalization"].get("mode", "channel_global")))
        cfg = apply_paper_preset(
            clf_cfg,
            model_type=model_type,
            epoch_cap=int(args.epoch_cap),
            include_scheduler=True,
            disable_step_control=True,
        )
        cfg["augmentation"]["modes"] = [aug_mode]
        pp_cfg_model = copy.deepcopy(pp_cfg)
        pp_cfg_model["normalization"]["mode"] = norm_mode

        for subject in subjects:
            split = _split_subject_index(
                index_df=index_df,
                subject=subject,
                split_seed=split_seed,
                val_ratio=val_ratio,
                stratify=split_stratify,
            )
            run_id = make_exp_id(
                "paper_track",
                subject=subject,
                seed=seed,
                clf=model_type,
                track=track_name,
                aug=aug_mode,
                val=val_ratio,
            )
            run_dir = ROOT / "runs/clf" / run_id
            metrics_path = run_dir / "metrics.json"
            if (not args.force) and metrics_path.exists():
                metrics = load_json(metrics_path)
                print(f"[skip] {run_dir.name} (metrics.json exists)")
            else:
                metrics = train_classifier(
                    split=split,
                    index_df=index_df,
                    clf_cfg=cfg,
                    preprocess_cfg=pp_cfg_model,
                    out_dir=run_dir,
                    mode=aug_mode,
                    ratio=0.0,
                    evaluate_test=True,
                )
            rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "model": model_type,
                    "track": track_name,
                    "val_ratio": val_ratio,
                    "aug_mode": aug_mode,
                    "epochs": int(cfg["train"]["epochs"]),
                    "batch_size": int(cfg["train"].get("batch_size", 0)),
                    "lr": float(cfg["train"]["lr"]),
                    **metrics,
                }
            )
            print(
                f"done model={model_type} subject={subject:02d} "
                f"acc={metrics['acc']:.4f} kappa={metrics['kappa']:.4f}"
            , flush=True)

    if not rows:
        raise RuntimeError("No runs executed.")

    df = pd.DataFrame(rows)
    suffix = f"_seed{seed}" if not args.tag else f"_seed{seed}_{args.tag}"
    metrics_out = ROOT / "results/metrics" / f"paper_track{suffix}.csv"
    ensure_dir(metrics_out.parent)
    df.to_csv(metrics_out, index=False)
    print(f"saved metrics -> {metrics_out}")

    summary = df.groupby("model")[["acc", "kappa", "f1_macro"]].agg(["mean", "std"]).reset_index()
    summary_out = ROOT / "results/tables" / f"paper_track{suffix}_summary.csv"
    ensure_dir(summary_out.parent)
    summary.to_csv(summary_out, index=False)
    print(f"saved summary -> {summary_out}")

    comp_rows = []
    mean_df = df.groupby("model")[["acc", "kappa"]].mean().reset_index()
    for _, r in mean_df.iterrows():
        m = str(r["model"])
        tgt = paper_cfg["paper_targets"].get(m)
        if tgt is None:
            continue
        metric = str(tgt["metric"])
        ref = float(tgt["value"])
        if metric == "acc":
            ours = float(r["acc"] * 100.0)
            delta = ours - ref
        elif metric == "kappa":
            ours = float(r["kappa"])
            delta = ours - ref
        else:
            continue
        comp_rows.append(
            {
                "model": m,
                "metric": metric,
                "ours": ours,
                "paper": ref,
                "abs_diff": abs(delta),
                "delta": delta,
                "pass_3pct": abs(delta) <= max_diff,
                "source": str(tgt.get("source", "")),
            }
        )
    comp = pd.DataFrame(comp_rows)
    comp_out = ROOT / "results/tables" / f"paper_track{suffix}_compare.csv"
    comp.to_csv(comp_out, index=False)
    print(f"saved compare -> {comp_out}")
    if len(comp) > 0:
        passed = int(comp["pass_3pct"].sum())
        print(f"acceptance: {passed}/{len(comp)} models within ±{max_diff}")


if __name__ == "__main__":
    main()
