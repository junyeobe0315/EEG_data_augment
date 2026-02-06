from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TUNING_PK_FIELDS = ["target", "phase", "trial_id"]

TUNING_COLUMNS = [
    "timestamp",
    "git_commit",
    "trial_id",
    "target",
    "phase",
    "status",
    "reason",
    "config_hash",
    "run_id",
    "seed",
    "objective_metric",
    "objective_value",
    "tie_break_value",
    "n_eval",
    "runtime_sec",
    "params_json",
]


def _normalize_row(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    out = {}
    for col in columns:
        out[col] = row.get(col, np.nan)
    return out


def _has_pk(df: pd.DataFrame, row: dict[str, Any], pk_fields: list[str]) -> bool:
    if df.empty:
        return False
    mask = None
    for k in pk_fields:
        if k not in df.columns:
            return False
        cur = df[k] == row.get(k)
        mask = cur if mask is None else (mask & cur)
    return bool(mask.any()) if mask is not None else False


def append_tuning_trial(path: str | Path, row: dict[str, Any], pk_fields: list[str] | None = None) -> bool:
    """Append one tuning trial row with resume-safe primary-key skip."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pk = pk_fields or TUNING_PK_FIELDS
    columns = TUNING_COLUMNS.copy()

    if p.exists():
        df = pd.read_csv(p)
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns]
        if _has_pk(df, row, pk):
            return False
        normalized = _normalize_row(row, columns)
        df = pd.concat([df, pd.DataFrame([normalized])], ignore_index=True)
        df.to_csv(p, index=False)
        return True

    normalized = _normalize_row(row, columns)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(normalized)
    return True
