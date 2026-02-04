from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import scipy.io


def parse_subject_session(file_path: str | Path) -> Tuple[int, str]:
    name = Path(file_path).name
    m = re.match(r"A(\d{2})([TE])\.gdf", name)
    if m is None:
        raise ValueError(f"Unexpected BCI2a filename: {name}")
    return int(m.group(1)), m.group(2)


def _desc_to_code(desc: str) -> str:
    desc = str(desc).strip()
    if desc.isdigit():
        return desc
    digits = "".join(ch for ch in desc if ch.isdigit())
    return digits if digits else desc


def load_true_labels(labels_dir: str | Path, subject: int, session: str = "E") -> np.ndarray:
    """
    Load competition-released true labels (AxxE.mat), 1..4 -> 0..3.
    """
    labels_dir = Path(labels_dir)
    file_mat = labels_dir / f"A{subject:02d}{session}.mat"
    if not file_mat.exists():
        raise FileNotFoundError(f"True label file not found: {file_mat}")

    data = scipy.io.loadmat(file_mat)
    key = None
    for cand in ("classlabel", "labels", "y", "label"):
        if cand in data:
            key = cand
            break
    if key is None:
        raise KeyError(f"No label key found in {file_mat}. Keys={list(data.keys())}")

    y = np.asarray(data[key]).reshape(-1).astype(np.int64)
    if y.min() >= 1 and y.max() <= 4:
        y = y - 1
    return y


def load_bci2a_trials(
    gdf_path: str | Path,
    channels: List[str],
    class_map: Dict[str, int],
    tmin: float,
    tmax: float,
    sfreq_expected: int,
    bandpass_cfg: Dict,
    notch_cfg: Dict,
    resample_cfg: Dict,
    eval_labels: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")

    if channels:
        picks = [ch for ch in channels if ch in raw.ch_names]
        if len(picks) == 0:
            picks = raw.ch_names[:22]
        raw.pick(picks)

    if bandpass_cfg.get("enabled", False):
        raw.filter(
            l_freq=float(bandpass_cfg.get("l_freq", 8.0)),
            h_freq=float(bandpass_cfg.get("h_freq", 30.0)),
            verbose="ERROR",
        )

    if notch_cfg.get("enabled", False):
        raw.notch_filter(float(notch_cfg.get("freq", 50.0)), verbose="ERROR")

    if resample_cfg.get("enabled", False):
        raw.resample(float(resample_cfg.get("sfreq", sfreq_expected)), verbose="ERROR")

    sfreq = int(round(float(raw.info["sfreq"])))
    if sfreq != int(sfreq_expected):
        raise ValueError(f"Sampling rate mismatch: got {sfreq}, expected {sfreq_expected}")

    data = raw.get_data()

    x_list = []
    y_list = []
    sample_ids = []

    subject, session = parse_subject_session(gdf_path)
    n_start = int(round(tmin * sfreq))
    n_stop = int(round(tmax * sfreq))

    if eval_labels is not None:
        # Evaluation session in BCI2a uses marker 783 with hidden labels.
        cue_desc = "783"
        cue_onsets = [int(round(float(ann["onset"]) * sfreq)) for ann in raw.annotations if _desc_to_code(ann["description"]) == cue_desc]
        if len(cue_onsets) == 0:
            # Fallback: use trial start markers if 783 is absent.
            cue_onsets = [int(round(float(ann["onset"]) * sfreq)) for ann in raw.annotations if _desc_to_code(ann["description"]) == "768"]

        if len(cue_onsets) != len(eval_labels):
            raise RuntimeError(
                f"Eval label count mismatch for {gdf_path}: cues={len(cue_onsets)}, labels={len(eval_labels)}"
            )

        for i, (onset, y) in enumerate(zip(cue_onsets, eval_labels)):
            s0 = onset + n_start
            s1 = onset + n_stop
            if s0 < 0 or s1 > data.shape[1] or s1 <= s0:
                continue

            x = data[:, s0:s1].astype(np.float32)
            sid = f"S{subject:02d}_{session}_trial{i:04d}"
            x_list.append(x)
            y_list.append(int(y))
            sample_ids.append(sid)
    else:
        for i, ann in enumerate(raw.annotations):
            code = _desc_to_code(ann["description"])
            if code not in class_map:
                continue

            onset = int(round(float(ann["onset"]) * sfreq))
            s0 = onset + n_start
            s1 = onset + n_stop
            if s0 < 0 or s1 > data.shape[1] or s1 <= s0:
                continue

            x = data[:, s0:s1].astype(np.float32)
            y = int(class_map[code])
            sid = f"S{subject:02d}_{session}_trial{i:04d}"

            x_list.append(x)
            y_list.append(y)
            sample_ids.append(sid)

    if not x_list:
        raise RuntimeError(f"No MI trials were extracted from {gdf_path}")

    return {
        "X": np.stack(x_list, axis=0),
        "y": np.asarray(y_list, dtype=np.int64),
        "sample_id": np.asarray(sample_ids),
    }


def load_processed_index(index_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(index_path)


def load_samples_by_ids(index_df: pd.DataFrame, sample_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    # Load only needed files and keep the exact split order.
    lookup = index_df.set_index("sample_id")
    rows = lookup.loc[sample_ids].reset_index()

    cache: Dict[str, Dict[str, np.ndarray]] = {}
    x_list: List[np.ndarray] = []
    y_list: List[int] = []

    for _, r in rows.iterrows():
        file_path = r["file"]
        if file_path not in cache:
            arr = np.load(file_path)
            cache[file_path] = {"X": arr["X"], "y": arr["y"]}
        rr = int(r["row"])
        x_list.append(cache[file_path]["X"][rr].astype(np.float32))
        y_list.append(int(cache[file_path]["y"][rr]))

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return x, y
