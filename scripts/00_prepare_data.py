#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from _script_utils import project_root

import pandas as pd

ROOT = project_root(__file__)

from src.dataio import load_bci2a_trials, load_true_labels, parse_subject_session
from src.utils import ensure_dir, load_yaml


def main() -> None:
    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    pp_cfg = load_yaml(ROOT / "configs/preprocess.yaml")

    raw_dir = Path(data_cfg["raw_dir"])
    true_labels_dir = Path(data_cfg.get("true_labels_dir", "./data/raw/true_labels"))
    require_eval_labels = bool(data_cfg.get("require_eval_labels", True))
    processed_dir = ensure_dir(data_cfg["processed_dir"])

    rows = []
    for gdf in sorted(raw_dir.glob("A*.gdf")):
        subject, session = parse_subject_session(gdf)
        if subject not in set(data_cfg["subjects"]):
            continue

        eval_labels = None
        if session == "E":
            try:
                eval_labels = load_true_labels(true_labels_dir, subject=subject, session="E")
            except FileNotFoundError:
                if require_eval_labels:
                    raise
                print(f"[skip] {gdf.name}: true labels not found under {true_labels_dir}")
                continue

        try:
            out = load_bci2a_trials(
                gdf,
                channels=data_cfg["channels"],
                class_map=data_cfg["class_map"],
                tmin=float(data_cfg["window"]["tmin"]),
                tmax=float(data_cfg["window"]["tmax"]),
                sfreq_expected=int(data_cfg["sfreq"]),
                bandpass_cfg=pp_cfg["bandpass"],
                notch_cfg=pp_cfg["notch"],
                resample_cfg=pp_cfg["resample"],
                eval_labels=eval_labels,
            )
        except RuntimeError as e:
            if "No MI trials were extracted" in str(e):
                print(f"[skip] {gdf.name}: no labeled MI trials in this file")
                continue
            raise

        out_file = processed_dir / f"S{subject:02d}_{session}.npz"
        file_abs = str(out_file.resolve())
        import numpy as np

        np.savez_compressed(out_file, X=out["X"], y=out["y"], sample_id=out["sample_id"])

        for i, sid in enumerate(out["sample_id"]):
            rows.append(
                {
                    "sample_id": str(sid),
                    "subject": subject,
                    "session": session,
                    "label": int(out["y"][i]),
                    "file": file_abs,
                    "row": i,
                }
            )

    if not rows:
        raise RuntimeError(f"No data prepared from raw dir: {raw_dir}")

    idx = pd.DataFrame(rows)
    idx.to_csv(data_cfg["index_path"], index=False)
    print(f"Prepared {len(idx)} windows -> {data_cfg['index_path']}")


if __name__ == "__main__":
    main()
