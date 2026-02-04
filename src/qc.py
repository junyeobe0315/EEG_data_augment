from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import welch


def _band_power(x: np.ndarray, sfreq: int, fmin: float, fmax: float) -> np.ndarray:
    # x: [N, C, T]
    freqs, psd = welch(x, fs=sfreq, axis=-1, nperseg=min(256, x.shape[-1]))
    m = (freqs >= fmin) & (freqs <= fmax)
    return psd[..., m].mean(axis=-1).mean(axis=-1)


def psd_mask(real_x: np.ndarray, synth_x: np.ndarray, sfreq: int, fmin: float, fmax: float, z_thr: float) -> np.ndarray:
    rp = np.log(_band_power(real_x, sfreq, fmin, fmax) + 1e-8)
    sp = np.log(_band_power(synth_x, sfreq, fmin, fmax) + 1e-8)
    mu, sd = rp.mean(), rp.std() + 1e-6
    z = np.abs((sp - mu) / sd)
    return z <= z_thr


def _cov_feat(x: np.ndarray) -> np.ndarray:
    feats = []
    for i in range(x.shape[0]):
        c = np.cov(x[i])
        feats.append(c.reshape(-1))
    return np.asarray(feats)


def covariance_mask(real_x: np.ndarray, synth_x: np.ndarray, q: float = 0.95) -> np.ndarray:
    r = _cov_feat(real_x)
    s = _cov_feat(synth_x)
    center = r.mean(axis=0, keepdims=True)
    rdist = np.linalg.norm(r - center, axis=1)
    sdist = np.linalg.norm(s - center, axis=1)
    thr = np.quantile(rdist, q)
    return sdist <= thr


def run_qc(real_x: np.ndarray, synth: Dict[str, np.ndarray], sfreq: int, cfg: Dict) -> tuple[Dict[str, np.ndarray], Dict]:
    mask = np.ones(synth["X"].shape[0], dtype=bool)

    if cfg["psd"].get("enabled", True):
        fmin, fmax = cfg["psd"].get("band", [8.0, 30.0])
        mask &= psd_mask(real_x, synth["X"], sfreq, float(fmin), float(fmax), float(cfg["psd"].get("z_threshold", 2.5)))

    if cfg["covariance"].get("enabled", True):
        mask &= covariance_mask(real_x, synth["X"], q=float(cfg["covariance"].get("dist_threshold_quantile", 0.95)))

    kept = {
        "X": synth["X"][mask],
        "y": synth["y"][mask],
        "sample_id": synth["sample_id"][mask],
    }

    report = {
        "n_before": int(synth["X"].shape[0]),
        "n_after": int(kept["X"].shape[0]),
        "keep_ratio": float(mask.mean()),
    }

    for cls in np.unique(synth["y"]):
        c = int(cls)
        cm = synth["y"] == c
        report[f"class_{c}_keep_ratio"] = float(mask[cm].mean()) if cm.any() else 0.0

    return kept, report
