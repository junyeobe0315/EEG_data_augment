from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_index
from src.data.splits import build_splits
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/val/test splits for BCI2a")
    parser.add_argument("--dataset_cfg", default="configs/dataset_bci2a.yaml")
    parser.add_argument("--split_cfg", default="configs/split.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    dataset_cfg = load_yaml(args.dataset_cfg, overrides=args.override)
    split_cfg = load_yaml(args.split_cfg, overrides=args.override)

    index_df = load_index(dataset_cfg["index_path"])
    build_splits(index_df, dataset_cfg["name"], split_cfg, out_root="./artifacts/splits")


if __name__ == "__main__":
    main()
