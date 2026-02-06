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
    "bal_acc",
    "kappa",
    "macro_f1",
    "val_acc",
    "val_bal_acc",
    "val_kappa",
    "val_macro_f1",
    # Diagnostics
    "pass_rate",
    "oversample_factor",
    "distance",
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
    "gpu_name",
    "gpu_mem_mb",
]


def make_run_id(values: dict[str, Any]) -> str:
    """Create a deterministic run_id from primary key fields.

    Inputs:
    - values: dict with keys in PRIMARY_KEY_FIELDS.

    Outputs:
    - short SHA-256 hex string.

    Internal logic:
    - Concatenates primary key values in order and hashes the string.
    """
    key = "|".join(str(values.get(k, "")) for k in PRIMARY_KEY_FIELDS)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def load_results(path: str | Path) -> pd.DataFrame:
    """Load results.csv if it exists.

    Inputs:
    - path: results file path.

    Outputs:
    - DataFrame (empty if file missing).

    Internal logic:
    - Returns empty DataFrame when file is absent to simplify callers.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def has_primary_key(df: pd.DataFrame, row: dict[str, Any]) -> bool:
    """Check if a row with the same primary key already exists.

    Inputs:
    - df: existing results DataFrame.
    - row: candidate result row.

    Outputs:
    - True if primary key matches an existing row.

    Internal logic:
    - Builds a boolean mask over PK columns and checks any match.
    """
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
    """Normalize a row to match a target column list.

    Inputs:
    - row: raw row dict.
    - columns: desired column ordering.

    Outputs:
    - dict with all columns filled (missing -> NaN).

    Internal logic:
    - Fills missing columns with NaN to maintain a fixed schema.
    """
    out = {}
    for col in columns:
        val = row.get(col, np.nan)
        out[col] = val
    return out


def append_result(path: str | Path, row: dict[str, Any]) -> bool:
    """Append a row to results.csv with primary-key skipping.

    Inputs:
    - path: results file path.
    - row: result row dict.

    Outputs:
    - True if row appended, False if skipped due to PK collision.

    Internal logic:
    - Loads/extends columns, normalizes the row, then appends or creates file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    columns = RESULT_COLUMNS.copy()
    if p.exists():
        df = pd.read_csv(p)
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns]
        if has_primary_key(df, row):
            return False

        normalized = _normalize_row(row, columns)
        df = pd.concat([df, pd.DataFrame([normalized])], ignore_index=True)
        df = df[columns]
        df.to_csv(p, index=False)
        return True

    normalized = _normalize_row(row, columns)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(normalized)
    return True
