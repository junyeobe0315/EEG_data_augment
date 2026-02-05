from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cache import prepare_preprocessed_cache
from src.utils.config import load_yaml


def main() -> None:
    """Cache preprocessed data and write index.csv for later runs.

    Inputs:
    - CLI args: dataset_cfg path, preprocess_cfg path, optional overrides.

    Outputs:
    - Writes compressed .npz files per subject/session to processed_dir.
    - Writes index.csv with (sample_id, subject, session, label, file, row).

    Internal logic:
    - Loads YAML configs (with overrides), then runs preprocessing + caching.
    """
    parser = argparse.ArgumentParser(description="Cache preprocessed BCI2a data")
    parser.add_argument("--dataset_cfg", default="configs/dataset_bci2a.yaml")
    parser.add_argument("--preprocess_cfg", default="configs/preprocess.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    dataset_cfg = load_yaml(args.dataset_cfg, overrides=args.override)  # dataset paths + channel info
    preprocess_cfg = load_yaml(args.preprocess_cfg, overrides=args.override)  # filter/resample settings

    prepare_preprocessed_cache(dataset_cfg, preprocess_cfg)


if __name__ == "__main__":
    main()
