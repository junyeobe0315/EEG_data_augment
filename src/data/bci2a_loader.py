from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import scipy.io


def parse_subject_session(file_path: str | Path) -> Tuple[int, str]:
    """Parse subject ID and session letter from a BCI2a GDF filename.

    Inputs:
    - file_path: path like "A01T.gdf" or "A01E.gdf".

    Outputs:
    - (subject_id, session_letter) where session_letter is "T" or "E".

    Internal logic:
    - Uses a regex on the filename to extract digits and session.
    """
    name = Path(file_path).name
    m = re.match(r"A(\d{2})([TE])\.gdf", name)
    if m is None:
        raise ValueError(f"Unexpected BCI2a filename: {name}")
    return int(m.group(1)), m.group(2)


def _desc_to_code(desc: str) -> str:
    """Normalize annotation descriptions to numeric event codes.

    Inputs:
    - desc: raw annotation description (string or numeric-like).

    Outputs:
    - A numeric code as string if digits exist, otherwise original cleaned string.

    Internal logic:
    - Strips whitespace and extracts digits when possible.
    """
    desc = str(desc).strip()
    if desc.isdigit():
        return desc
    digits = "".join(ch for ch in desc if ch.isdigit())
    return digits if digits else desc


def load_true_labels(labels_dir: str | Path, subject: int, session: str = "E") -> np.ndarray:
    """Load ground-truth labels for evaluation session from a .mat file.

    Inputs:
    - labels_dir: directory containing true label .mat files.
    - subject: subject integer (1–9).
    - session: typically "E".

    Outputs:
    - y: ndarray of shape [n_trials], 0-based class labels.

    Internal logic:
    - Reads .mat file, finds the label key, and converts to 0-based labels.
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
    """Load BCI2a trials, apply preprocessing, and return X/y/sample_id arrays.

    Inputs:
    - gdf_path: path to a .gdf file.
    - channels: list of EEG channel names to keep.
    - class_map: mapping of event codes to class indices.
    - tmin/tmax: epoch window in seconds relative to cue onset.
    - sfreq_expected: expected sampling rate after any resample.
    - bandpass_cfg/notch_cfg/resample_cfg: preprocessing settings.
    - eval_labels: optional labels for evaluation session (E).

    Outputs:
    - dict with:
      - "X": ndarray [N, C, T] float32
      - "y": ndarray [N] int64
      - "sample_id": ndarray [N] string identifiers

    Internal logic:
    - Reads raw GDF, applies filters, picks channels, extracts epochs,
      and aligns labels depending on session.
    """
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

    data = raw.get_data()  # raw EEG data [C, T_total]
    x_list: list[np.ndarray] = []  # per-trial EEG windows
    y_list: list[int] = []  # per-trial labels
    sample_ids: list[str] = []  # per-trial identifiers

    subject, session = parse_subject_session(gdf_path)
    n_start = int(round(tmin * sfreq))  # window start in samples
    n_stop = int(round(tmax * sfreq))   # window end in samples

    if eval_labels is not None:
        cue_desc = "783"  # cue onset for eval files
        cue_onsets = [int(round(float(ann["onset"]) * sfreq)) for ann in raw.annotations if _desc_to_code(ann["description"]) == cue_desc]  # sample indices
        if len(cue_onsets) == 0:
            cue_onsets = [int(round(float(ann["onset"]) * sfreq)) for ann in raw.annotations if _desc_to_code(ann["description"]) == "768"]  # fallback cue code

        if len(cue_onsets) != len(eval_labels):
            raise RuntimeError(
                f"Eval label count mismatch for {gdf_path}: cues={len(cue_onsets)}, labels={len(eval_labels)}"
            )

        for i, (onset, y) in enumerate(zip(cue_onsets, eval_labels)):
            s0 = onset + n_start  # window start index
            s1 = onset + n_stop   # window end index
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
