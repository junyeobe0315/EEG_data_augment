from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.bci2a_loader import load_bci2a_trials, load_true_labels, parse_subject_session
from src.utils.io import ensure_dir


def prepare_preprocessed_cache(dataset_cfg: dict, preprocess_cfg: dict) -> Path:
    """Prepare cached .npz files and index.csv for the dataset.

    Inputs:
    - dataset_cfg: dict with raw_dir, processed_dir, subjects, channels, etc.
    - preprocess_cfg: dict with filter/resample settings.

    Outputs:
    - Path to the generated index.csv.

    Internal logic:
    - Iterates over GDF files, extracts trials, writes per-session .npz,
      and aggregates metadata into a single index CSV.
    """
    raw_dir = Path(dataset_cfg["raw_dir"])
    if not raw_dir.exists() and Path("./BCICIV_2a_gdf").exists():
        raw_dir = Path("./BCICIV_2a_gdf")

    true_labels_dir = Path(dataset_cfg.get("true_labels_dir", "./data/raw/true_labels"))
    require_eval_labels = bool(dataset_cfg.get("require_eval_labels", True))

    processed_dir = ensure_dir(dataset_cfg["processed_dir"])  # output cache dir

    rows = []
    for gdf in sorted(raw_dir.glob("A*.gdf")):
        subject, session = parse_subject_session(gdf)
        if subject not in set(dataset_cfg["subjects"]):
            continue

        eval_labels = None
        if session == "E":
            try:
                eval_labels = load_true_labels(true_labels_dir, subject=subject, session="E")
            except FileNotFoundError:
                if require_eval_labels:
                    raise
                continue

        out = load_bci2a_trials(
            gdf,
            channels=dataset_cfg["channels"],
            class_map=dataset_cfg["class_map"],
            tmin=float(dataset_cfg["window"]["tmin"]),
            tmax=float(dataset_cfg["window"]["tmax"]),
            sfreq_expected=int(dataset_cfg["sfreq"]),
            bandpass_cfg=preprocess_cfg["bandpass"],
            notch_cfg=preprocess_cfg["notch"],
            resample_cfg=preprocess_cfg["resample"],
            eval_labels=eval_labels,
        )

        out_file = processed_dir / f"S{subject:02d}_{session}.npz"
        np.savez_compressed(out_file, X=out["X"], y=out["y"], sample_id=out["sample_id"])

        for i, sid in enumerate(out["sample_id"]):
            rows.append(
                {
                    "sample_id": str(sid),
                    "subject": subject,
                    "session": session,
                    "label": int(out["y"][i]),
                    "file": str(out_file.resolve()),
                    "row": i,
                }
            )

    if not rows:
        raise RuntimeError(f"No data prepared from raw dir: {raw_dir}")

    idx = pd.DataFrame(rows)
    index_path = Path(dataset_cfg["index_path"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(index_path, index=False)
    return index_path
