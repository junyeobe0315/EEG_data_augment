#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import cheby2, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dataio import load_true_labels, parse_subject_session
from src.utils import ensure_dir, load_yaml


def _digits_only(s: str) -> str:
    s = str(s).strip()
    if s.isdigit():
        return s
    d = "".join(ch for ch in s if ch.isdigit())
    return d if d else s


def _event_id_map(event_dict: dict) -> dict[str, int]:
    out = {}
    for name, idx in event_dict.items():
        out[_digits_only(name)] = int(idx)
    return out


def _pick_eeg22(raw, channel_list: list[str]) -> None:
    picks = [ch for ch in channel_list if ch in raw.ch_names]
    if len(picks) != len(channel_list):
        picks = raw.ch_names[:22]
    raw.pick(picks)


def _extract_trial_start_windows(
    raw,
    session: str,
    class_map: dict[str, int],
    tmin_s: float,
    tmax_s: float,
    eval_labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    sfreq = float(raw.info["sfreq"])
    events, event_dict = mne.events_from_annotations(raw, verbose="ERROR")
    emap = _event_id_map(event_dict)
    start_id = emap.get("768")
    if start_id is None:
        raise RuntimeError("Event 768 (trial-start) not found.")

    starts = events[events[:, 2] == start_id][:, 0].astype(int)
    if len(starts) == 0:
        raise RuntimeError("No trial-start events found.")

    s0_off = int(round(tmin_s * sfreq))
    s1_off = int(round(tmax_s * sfreq))
    data = raw.get_data()

    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    if session == "T":
        class_ids = {emap[k]: int(v) for k, v in class_map.items() if k in emap}
        cls_events = [(int(s), int(eid)) for s, _, eid in events if int(eid) in class_ids]
        cls_events.sort(key=lambda x: x[0])
        p = 0
        for i, st in enumerate(starts):
            nxt = starts[i + 1] if (i + 1) < len(starts) else int(data.shape[1] + 1)
            while p < len(cls_events) and cls_events[p][0] <= st:
                p += 1
            q = p
            label = None
            while q < len(cls_events) and cls_events[q][0] < nxt:
                label = class_ids[cls_events[q][1]]
                break
            if label is None:
                continue
            a = st + s0_off
            b = st + s1_off
            if a < 0 or b > data.shape[1] or b <= a:
                continue
            x_list.append(data[:, a:b].astype(np.float32))
            y_list.append(int(label))
    else:
        if eval_labels is None:
            raise RuntimeError("Evaluation labels are required for session E.")
        if len(starts) != len(eval_labels):
            raise RuntimeError(f"E-session mismatch: starts={len(starts)} labels={len(eval_labels)}")
        for st, y in zip(starts, eval_labels):
            a = st + s0_off
            b = st + s1_off
            if a < 0 or b > data.shape[1] or b <= a:
                continue
            x_list.append(data[:, a:b].astype(np.float32))
            y_list.append(int(y))

    if not x_list:
        raise RuntimeError("No windows extracted in trial_start mode.")
    return np.stack(x_list, axis=0), np.asarray(y_list, dtype=np.int64)


def _extract_cue_windows(
    raw,
    session: str,
    class_map: dict[str, int],
    tmin_s: float,
    tmax_s: float,
    eval_labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    sfreq = float(raw.info["sfreq"])
    events, event_dict = mne.events_from_annotations(raw, verbose="ERROR")
    emap = _event_id_map(event_dict)
    data = raw.get_data()
    s0_off = int(round(tmin_s * sfreq))
    s1_off = int(round(tmax_s * sfreq))

    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    if session == "T":
        for k, y in class_map.items():
            if k not in emap:
                continue
            cue_id = emap[k]
            for st in events[events[:, 2] == cue_id][:, 0].astype(int):
                a = st + s0_off
                b = st + s1_off
                if a < 0 or b > data.shape[1] or b <= a:
                    continue
                x_list.append(data[:, a:b].astype(np.float32))
                y_list.append(int(y))
    else:
        unk_id = emap.get("783")
        if unk_id is None:
            raise RuntimeError("Event 783 (unknown cue) not found for E session.")
        starts = events[events[:, 2] == unk_id][:, 0].astype(int)
        if eval_labels is None:
            raise RuntimeError("Evaluation labels are required for cue mode E session.")
        if len(starts) != len(eval_labels):
            raise RuntimeError(f"E-session mismatch: cues={len(starts)} labels={len(eval_labels)}")
        for st, y in zip(starts, eval_labels):
            a = st + s0_off
            b = st + s1_off
            if a < 0 or b > data.shape[1] or b <= a:
                continue
            x_list.append(data[:, a:b].astype(np.float32))
            y_list.append(int(y))

    if not x_list:
        raise RuntimeError("No windows extracted in cue_onset mode.")
    return np.stack(x_list, axis=0), np.asarray(y_list, dtype=np.int64)


def _apply_track_filter(x: np.ndarray, sfreq: int, filt_cfg: dict) -> np.ndarray:
    ftype = str(filt_cfg.get("type", "none")).lower()
    if ftype == "none":
        return x.astype(np.float32)
    if ftype != "cheby2_bandpass":
        raise ValueError(f"Unsupported filter type: {ftype}")
    sos = cheby2(
        N=int(filt_cfg.get("order", 6)),
        rs=float(filt_cfg.get("rs", 60.0)),
        Wn=[float(filt_cfg.get("l_freq", 4.0)), float(filt_cfg.get("h_freq", 40.0))],
        btype="bandpass",
        fs=float(sfreq),
        output="sos",
    )
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def _extract_by_track(raw, session: str, class_map: dict[str, int], track_cfg: dict, eval_labels: np.ndarray | None):
    mode = str(track_cfg["window_mode"])
    if mode == "trial_start":
        x, y = _extract_trial_start_windows(
            raw=raw,
            session=session,
            class_map=class_map,
            tmin_s=float(track_cfg["tmin_sec"]),
            tmax_s=float(track_cfg["tmax_sec"]),
            eval_labels=eval_labels,
        )
    elif mode == "cue_onset":
        x, y = _extract_cue_windows(
            raw=raw,
            session=session,
            class_map=class_map,
            tmin_s=float(track_cfg["tmin_sec"]),
            tmax_s=float(track_cfg["tmax_sec"]),
            eval_labels=eval_labels,
        )
    else:
        raise ValueError(f"Unsupported window_mode: {mode}")
    return x, y


def main() -> None:
    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    paper_cfg = load_yaml(ROOT / "configs/paper_track.yaml")

    raw_dir = Path(data_cfg["raw_dir"])
    labels_dir = Path(data_cfg.get("true_labels_dir", "./data/raw/true_labels"))
    sfreq = int(data_cfg["sfreq"])
    class_map = {str(k): int(v) for k, v in data_cfg["class_map"].items()}
    channels = list(data_cfg["channels"])
    subjects = set(int(s) for s in data_cfg["subjects"])

    all_rows = []
    for track_name, track_cfg in paper_cfg["tracks"].items():
        track_dir = ensure_dir(ROOT / "data/paper_track" / track_name)
        rows = []
        for gdf in sorted(raw_dir.glob("A*.gdf")):
            subject, session = parse_subject_session(gdf)
            if subject not in subjects:
                continue
            raw = mne.io.read_raw_gdf(str(gdf), preload=True, verbose="ERROR")
            _pick_eeg22(raw, channels)
            if int(round(float(raw.info["sfreq"]))) != sfreq:
                raw.resample(float(sfreq), verbose="ERROR")

            eval_labels = None
            if session == "E":
                eval_labels = load_true_labels(labels_dir, subject=subject, session="E")

            x, y = _extract_by_track(raw, session=session, class_map=class_map, track_cfg=track_cfg, eval_labels=eval_labels)
            x = _apply_track_filter(x, sfreq=sfreq, filt_cfg=track_cfg.get("filter", {"type": "none"}))

            sample_ids = np.asarray(
                [f"PT_{track_name}_S{subject:02d}_{session}_{i:04d}" for i in range(len(y))],
                dtype=object,
            )
            out_file = track_dir / f"S{subject:02d}_{session}.npz"
            np.savez_compressed(out_file, X=x, y=y, sample_id=sample_ids)

            file_abs = str(out_file.resolve())
            for i, sid in enumerate(sample_ids):
                rows.append(
                    {
                        "track": track_name,
                        "sample_id": str(sid),
                        "subject": int(subject),
                        "session": session,
                        "label": int(y[i]),
                        "file": file_abs,
                        "row": int(i),
                    }
                )
        if not rows:
            raise RuntimeError(f"No rows extracted for track={track_name}")
        idx = pd.DataFrame(rows)
        idx_path = track_dir / "index.csv"
        idx.to_csv(idx_path, index=False)
        print(f"[track={track_name}] saved {len(idx)} samples -> {idx_path}")
        all_rows.extend(rows)

    all_idx = pd.DataFrame(all_rows)
    all_path = ROOT / "data/paper_track/index_all.csv"
    ensure_dir(all_path.parent)
    all_idx.to_csv(all_path, index=False)
    print(f"[all] saved {len(all_idx)} rows -> {all_path}")


if __name__ == "__main__":
    main()

