from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import ensure_dir


def _stratified_subsample(indices: np.ndarray, labels: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Stratified subsample of indices by class.

    Inputs:
    - indices: array of indices to subsample.
    - labels: class labels aligned with indices.
    - frac: fraction to keep per class.
    - seed: RNG seed for reproducibility.

    Outputs:
    - subsampled indices array preserving original order.

    Internal logic:
    - Samples per-class indices with a fixed RNG and then reorders to original order.
    """
    if frac >= 0.999:
        return indices.copy()
    rng = np.random.default_rng(seed)
    keep = []
    for cls in sorted(np.unique(labels).tolist()):
        mask = labels == cls
        cls_idx = indices[mask]
        n = len(cls_idx)
        k = max(1, int(round(n * frac)))
        choose = rng.choice(n, size=min(k, n), replace=False)
        keep.extend(cls_idx[choose].tolist())
    keep_set = set(keep)
    return np.array([i for i in indices if i in keep_set], dtype=np.int64)


def build_splits(index_df: pd.DataFrame, dataset_name: str, split_cfg: dict, out_root: str | Path) -> None:
    """Build and persist cross-session train/val/test splits.

    Inputs:
    - index_df: DataFrame with session and label columns.
    - dataset_name: dataset identifier string.
    - split_cfg: config dict with protocol, seeds, val_ratio, low_data_fracs.
    - out_root: output root directory for split files.

    Outputs:
    - Writes .npy index files under artifacts/splits/.

    Internal logic:
    - Train/val split within session T, test is full session E.
    - Low-data subsamples are stratified within T_train_full.
    """
    protocol = str(split_cfg.get("protocol", "cross_session"))
    if protocol != "cross_session":
        raise ValueError("Only cross_session protocol is supported in v2.")

    low_fracs = [float(x) for x in split_cfg.get("low_data_fracs", [1.0])]  # r list
    seeds = [int(s) for s in split_cfg.get("seeds", [0])]  # split seeds
    val_ratio = float(split_cfg.get("val_ratio", 0.2))  # T val ratio

    for subject in sorted(index_df["subject"].unique().tolist()):
        sub = index_df[index_df["subject"] == subject].copy()
        t_df = sub[sub["session"] == "T"].copy()  # training session
        e_df = sub[sub["session"] == "E"].copy()  # eval session
        if len(t_df) == 0 or len(e_df) == 0:
            raise RuntimeError(f"Subject {subject}: both T and E sessions are required.")

        for seed in seeds:
            tr_idx, va_idx = train_test_split(
                t_df.index.values,
                test_size=val_ratio,
                random_state=int(seed),
                stratify=t_df["label"].tolist(),
            )
            te_idx = e_df.index.values  # use full E session for test

            base_dir = ensure_dir(Path(out_root) / dataset_name / f"subject_{subject:02d}" / f"seed{seed}")
            np.save(base_dir / "T_train_full_idx.npy", np.asarray(tr_idx, dtype=np.int64))
            np.save(base_dir / "T_val_idx.npy", np.asarray(va_idx, dtype=np.int64))
            np.save(base_dir / "E_test_idx.npy", np.asarray(te_idx, dtype=np.int64))

            tr_labels = index_df.loc[tr_idx, "label"].values
            for frac in low_fracs:
                sub_idx = _stratified_subsample(tr_idx, tr_labels, frac=frac, seed=int(seed))
                sub_dir = ensure_dir(base_dir / f"r{str(frac).replace('.', 'p')}")
                np.save(sub_dir / "T_train_sub_idx.npy", np.asarray(sub_idx, dtype=np.int64))
