from __future__ import annotations

import argparse

from src.data.cache import prepare_preprocessed_cache
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache preprocessed BCI2a data")
    parser.add_argument("--dataset_cfg", default="configs/dataset_bci2a.yaml")
    parser.add_argument("--preprocess_cfg", default="configs/preprocess.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    dataset_cfg = load_yaml(args.dataset_cfg, overrides=args.override)
    preprocess_cfg = load_yaml(args.preprocess_cfg, overrides=args.override)

    prepare_preprocessed_cache(dataset_cfg, preprocess_cfg)


if __name__ == "__main__":
    main()
