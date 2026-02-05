from __future__ import annotations

from typing import Dict

import numpy as np

from src.qc.psd_qc import fit_psd_stats, psd_zscore
from src.qc.cov_qc import fit_cov_stats, cov_distance


def fit_qc(real_x: np.ndarray, real_y: np.ndarray, sfreq: int, cfg: Dict) -> dict:
    classes = np.unique(real_y).tolist()
    state = {"psd": {}, "cov": {}, "classes": classes}

    if cfg.get("psd", {}).get("enabled", True):
        fmin, fmax = cfg.get("psd", {}).get("band", [8.0, 30.0])
        for cls in classes:
            ref = real_x[real_y == cls]
            mu, sd = fit_psd_stats(ref, sfreq, float(fmin), float(fmax))
            state["psd"][int(cls)] = {"mu": mu, "sd": sd, "fmin": float(fmin), "fmax": float(fmax)}

    if cfg.get("covariance", {}).get("enabled", True):
        q = float(cfg.get("covariance", {}).get("dist_threshold_quantile", 0.95))
        for cls in classes:
            ref = real_x[real_y == cls]
            center, thr = fit_cov_stats(ref, q=q)
            state["cov"][int(cls)] = {"center": center, "thr": thr}

    return state


def filter_qc(synth_x: np.ndarray, synth_y: np.ndarray, sfreq: int, cfg: Dict, state: dict) -> tuple[np.ndarray, np.ndarray]:
    n = int(synth_x.shape[0])
    mask = np.ones(n, dtype=bool)
    score = np.zeros(n, dtype=np.float64)

    classes = state.get("classes", np.unique(synth_y).tolist())
    for cls in classes:
        cls = int(cls)
        idx = synth_y == cls
        if not np.any(idx):
            continue

        if cfg.get("psd", {}).get("enabled", True) and cls in state.get("psd", {}):
            st = state["psd"][cls]
            z = psd_zscore(synth_x[idx], st["mu"], st["sd"], sfreq, st["fmin"], st["fmax"])
            mask[idx] &= z <= float(cfg["psd"].get("z_threshold", 2.5))
            score[idx] += z

        if cfg.get("covariance", {}).get("enabled", True) and cls in state.get("cov", {}):
            st = state["cov"][cls]
            dist = cov_distance(synth_x[idx], st["center"])
            mask[idx] &= dist <= float(st["thr"])
            score[idx] += dist / max(float(st["thr"]), 1e-6)

    # Optional relaxation to keep minimum ratio
    target_keep_ratio = float(cfg.get("target_keep_ratio", 0.0))
    if n > 0 and target_keep_ratio > 0.0:
        for cls in classes:
            cls = int(cls)
            idx = synth_y == cls
            if not np.any(idx):
                continue
            keep_ratio = float(mask[idx].mean())
            if keep_ratio >= target_keep_ratio:
                continue
            k = min(int(np.sum(idx)), max(1, int(np.ceil(np.sum(idx) * target_keep_ratio))))
            order = np.argsort(score[idx])
            relaxed = np.zeros(int(np.sum(idx)), dtype=bool)
            relaxed[order[:k]] = True
            mask[idx] = relaxed

    return mask, score
