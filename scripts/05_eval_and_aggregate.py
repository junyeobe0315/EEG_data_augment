#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.aggregate import aggregate_metrics
from src.utils import ensure_dir


def main() -> None:
    tables_dir = ensure_dir(ROOT / "results/tables")
    out_csv = tables_dir / "main_results.csv"
    aggregate_metrics(ROOT / "results/metrics", out_csv)
    print(f"Saved aggregate table -> {out_csv}")


if __name__ == "__main__":
    main()
