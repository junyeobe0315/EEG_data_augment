#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.aggregate import aggregate_metrics
from src.dataio import load_processed_index, load_samples_by_ids
from src.distribution import FrozenEEGNetEmbedder, classwise_distance_summary
from src.utils import load_yaml


def _load_split_obj(split_file: str | Path, split_name: str) -> dict:
    p = Path(split_file)
    if not p.exists():
        p = ROOT / "data/splits" / f"{split_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Split json not found: {split_file} / {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_baseline_ckpts(df: pd.DataFrame, baseline_model: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    base = df[(df["clf_model"] == baseline_model) & (df["condition"] == "C0_no_aug")].copy()
    for split_name, sub in base.groupby("split"):
        for _, r in sub.iterrows():
            ckpt = Path(r["run_dir"]) / "ckpt.pt"
            if ckpt.exists():
                out[str(split_name)] = ckpt
                break
    return out


def _attach_distribution_distance(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df

    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    clf_cfg = load_yaml(ROOT / "configs/clf.yaml")
    sweep_cfg = load_yaml(ROOT / "configs/sweep.yaml")

    emb_cfg = sweep_cfg.get("analysis", {}).get("embedding", {})
    dist_cfg = sweep_cfg.get("analysis", {}).get("distance", {})
    baseline_model = str(emb_cfg.get("baseline_model", "eegnet_tf_faithful"))

    baseline_ckpts = _resolve_baseline_ckpts(df, baseline_model=baseline_model)
    if not baseline_ckpts:
        print("[warn] No baseline EEGNet checkpoint found for distance analysis. Skipping.")
        return df

    index_df = load_processed_index(data_cfg["index_path"])
    num_classes = int(len(data_cfg.get("class_names", [0, 1, 2, 3])))
    n_proj = int(dist_cfg.get("n_projections", 64))
    mmd_gamma = dist_cfg.get("mmd_gamma", "median_heuristic")

    out = df.copy()
    out["embedding_method"] = "none"
    out["dist_swd"] = np.nan
    out["dist_mmd"] = np.nan

    # Cache per split for speed.
    embedder_cache: dict[str, FrozenEEGNetEmbedder] = {}
    real_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    emb_device = str(emb_cfg.get("device", "auto"))
    if emb_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = emb_device

    for i, row in out.iterrows():
        ratio = float(row.get("ratio", 0.0))
        n_train_aug = int(row.get("n_train_aug", 0))
        aug_npz = Path(str(row.get("aug_npz", "")))
        split_name = str(row.get("split", ""))

        if ratio <= 0 or n_train_aug <= 0 or not aug_npz.exists():
            continue
        if split_name not in baseline_ckpts:
            continue

        if split_name not in embedder_cache:
            try:
                embedder_cache[split_name] = FrozenEEGNetEmbedder(
                    ckpt_path=baseline_ckpts[split_name],
                    clf_cfg=clf_cfg["model"],
                    device=device,
                )
            except Exception as e:
                print(f"[warn] Failed to build embedder for {split_name}: {e}")
                continue

        embedder = embedder_cache[split_name]

        if split_name not in real_cache:
            try:
                split_obj = _load_split_obj(str(row.get("split_file", "")), split_name=split_name)
                x_real, y_real = load_samples_by_ids(index_df, split_obj["train_ids"])
                x_real = embedder.normalize(x_real)
                real_emb = embedder.transform(x_real)
                real_cache[split_name] = (real_emb, y_real.astype(np.int64))
            except Exception as e:
                print(f"[warn] Failed to cache real embedding for {split_name}: {e}")
                continue

        try:
            aug_arr = np.load(aug_npz)
            x_aug = aug_arr["X"].astype(np.float32)
            y_aug = aug_arr["y"].astype(np.int64)

            x_aug = embedder.normalize(x_aug)
            aug_emb = embedder.transform(x_aug)
            real_emb, y_real = real_cache[split_name]

            d = classwise_distance_summary(
                real_emb=real_emb,
                real_y=y_real,
                aug_emb=aug_emb,
                aug_y=y_aug,
                num_classes=num_classes,
                n_projections=n_proj,
                mmd_gamma=mmd_gamma,
                seed=int(row.get("seed", 0)),
            )
            for k, v in d.items():
                out.loc[i, k] = v
            out.loc[i, "embedding_method"] = str(emb_cfg.get("method", "eegnet_frozen"))
        except Exception as e:
            print(f"[warn] Distance analysis failed for row={i}, split={split_name}: {e}")

    return out


def main() -> None:
    split_cfg = load_yaml(ROOT / "configs/split.yaml")
    sweep_cfg = load_yaml(ROOT / "configs/sweep.yaml")

    metrics_csv = ROOT / "results/metrics" / f"clf_{split_cfg['protocol']}.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_csv}. Run scripts/04_train_clf.py first.")

    df = pd.read_csv(metrics_csv)
    df = _attach_distribution_distance(df)
    df.to_csv(metrics_csv, index=False)

    ratio_ref = float(
        sweep_cfg.get("analysis", {}).get("plots", {}).get("ratio_ref_for_r_curve", 1.0)
    )
    out_csv = ROOT / "results/tables/main_results.csv"
    aggregate_metrics(ROOT / "results/metrics", out_csv, ratio_ref_for_r_curve=ratio_ref)
    print(f"Saved aggregate tables -> {out_csv.parent}")
    print(f"Saved figures -> {(ROOT / 'results/figures')}")


if __name__ == "__main__":
    main()
