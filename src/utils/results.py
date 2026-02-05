from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_KEY_FIELDS = [
    "subject",
    "seed",
    "r",
    "classifier",
    "method",
    "generator",
    "qc_on",
    "alpha_ratio",
]

RESULT_COLUMNS = [
    # Metadata
    "run_id",
    "timestamp",
    "git_commit",
    "config_hash",
    "dataset",
    "subject",
    "seed",
    "r",
    "method",
    "classifier",
    "generator",
    "alpha_ratio",
    "qc_on",
    # Metrics
    "acc",
    "kappa",
    "macro_f1",
    "val_acc",
    "val_kappa",
    "val_macro_f1",
    # Diagnostics
    "pass_rate",
    "oversample_factor",
    "dist_mmd",
    "dist_swd",
    "dist_mmd_class_0",
    "dist_mmd_class_1",
    "dist_mmd_class_2",
    "dist_mmd_class_3",
    "dist_swd_class_0",
    "dist_swd_class_1",
    "dist_swd_class_2",
    "dist_swd_class_3",
    "ratio_effective",
    "alpha_mix_effective",
    "runtime_sec",
    "device",
]


def make_run_id(values: dict[str, Any]) -> str:
    key = "|".join(str(values.get(k, "")) for k in PRIMARY_KEY_FIELDS)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def load_results(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def has_primary_key(df: pd.DataFrame, row: dict[str, Any]) -> bool:
    if df.empty:
        return False
    mask = None
    for k in PRIMARY_KEY_FIELDS:
        if k not in df.columns:
            return False
        v = row.get(k)
        cur = df[k] == v
        mask = cur if mask is None else (mask & cur)
    return bool(mask.any()) if mask is not None else False


def _normalize_row(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    out = {}
    for col in columns:
        val = row.get(col, np.nan)
        out[col] = val
    return out


def append_result(path: str | Path, row: dict[str, Any]) -> bool:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        df = pd.read_csv(p)
        if has_primary_key(df, row):
            return False

        columns = list(df.columns)
        for k in row.keys():
            if k not in columns:
                columns.append(k)
                df[k] = np.nan
        if not columns:
            columns = RESULT_COLUMNS

        normalized = _normalize_row(row, columns)
        df = pd.concat([df, pd.DataFrame([normalized])], ignore_index=True)
        df = df[columns]
        df.to_csv(p, index=False)
        return True

    columns = RESULT_COLUMNS.copy()
    for k in row.keys():
        if k not in columns:
            columns.append(k)

    normalized = _normalize_row(row, columns)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(normalized)
    return True
