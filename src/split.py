from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_json, save_json


def _stratified_split(ids: List[str], y: List[int], test_size: float, seed: int):
    return train_test_split(ids, test_size=test_size, random_state=seed, stratify=y)


def make_split(
    index_df: pd.DataFrame,
    protocol: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    loso_subject: int | None = None,
) -> Dict[str, List[str]]:
    ids = index_df["sample_id"].tolist()

    if protocol == "cross_session":
        train_pool = index_df[index_df["session"] == "T"]
        test_df = index_df[index_df["session"] == "E"]

        tr_ids, va_ids = _stratified_split(
            train_pool["sample_id"].tolist(),
            train_pool["label"].tolist(),
            test_size=val_ratio,
            seed=seed,
        )
        te_ids = test_df["sample_id"].tolist()

    elif protocol == "within_subject":
        tr_ids, va_ids, te_ids = [], [], []
        for subject, grp in index_df.groupby("subject"):
            sub_ids = grp["sample_id"].tolist()
            sub_y = grp["label"].tolist()

            train_ids, test_ids = _stratified_split(sub_ids, sub_y, test_size=test_ratio, seed=seed)
            train_grp = grp[grp["sample_id"].isin(train_ids)]
            tr_sub, va_sub = _stratified_split(
                train_grp["sample_id"].tolist(),
                train_grp["label"].tolist(),
                test_size=val_ratio,
                seed=seed,
            )
            tr_ids.extend(tr_sub)
            va_ids.extend(va_sub)
            te_ids.extend(test_ids)

    elif protocol == "loso":
        if loso_subject is None:
            raise ValueError("loso_subject is required for LOSO protocol")

        test_df = index_df[index_df["subject"] == loso_subject]
        train_pool = index_df[index_df["subject"] != loso_subject]

        tr_ids, va_ids = _stratified_split(
            train_pool["sample_id"].tolist(),
            train_pool["label"].tolist(),
            test_size=val_ratio,
            seed=seed,
        )
        te_ids = test_df["sample_id"].tolist()
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    split = {
        "protocol": protocol,
        "seed": int(seed),
        "train_ids": list(tr_ids),
        "val_ids": list(va_ids),
        "test_ids": list(te_ids),
    }
    if protocol == "loso":
        split["loso_subject"] = int(loso_subject)

    overlap = set(split["train_ids"]) & set(split["val_ids"]) | set(split["train_ids"]) & set(split["test_ids"]) | set(split["val_ids"]) & set(split["test_ids"])
    if overlap:
        raise RuntimeError("Split leakage detected: train/val/test overlap exists")

    return split


def save_split(path: str | Path, split: Dict[str, List[str]]) -> None:
    save_json(path, split)


def load_split(path: str | Path) -> Dict[str, List[str]]:
    return load_json(path)
