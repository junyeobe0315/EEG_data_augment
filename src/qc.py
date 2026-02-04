from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import welch


def _band_power(x: np.ndarray, sfreq: int, fmin: float, fmax: float) -> np.ndarray:
    # x: [N, C, T]
    freqs, psd = welch(x, fs=sfreq, axis=-1, nperseg=min(256, x.shape[-1]))
    m = (freqs >= fmin) & (freqs <= fmax)
    return psd[..., m].mean(axis=-1).mean(axis=-1)


def psd_zscore(real_x: np.ndarray, synth_x: np.ndarray, sfreq: int, fmin: float, fmax: float) -> np.ndarray:
    rp = np.log(_band_power(real_x, sfreq, fmin, fmax) + 1e-8)
    sp = np.log(_band_power(synth_x, sfreq, fmin, fmax) + 1e-8)
    mu, sd = rp.mean(), rp.std() + 1e-6
    return np.abs((sp - mu) / sd)


def psd_mask(real_x: np.ndarray, synth_x: np.ndarray, sfreq: int, fmin: float, fmax: float, z_thr: float) -> np.ndarray:
    z = psd_zscore(real_x, synth_x, sfreq, fmin, fmax)
    return z <= z_thr


def _cov_feat(x: np.ndarray) -> np.ndarray:
    feats = []
    for i in range(x.shape[0]):
        c = np.cov(x[i])
        feats.append(c.reshape(-1))
    return np.asarray(feats)


def covariance_mask(real_x: np.ndarray, synth_x: np.ndarray, q: float = 0.95) -> np.ndarray:
    sdist, thr = covariance_distance(real_x, synth_x, q=q)
    return sdist <= thr


def covariance_distance(real_x: np.ndarray, synth_x: np.ndarray, q: float = 0.95) -> tuple[np.ndarray, float]:
    r = _cov_feat(real_x)
    s = _cov_feat(synth_x)
    center = r.mean(axis=0, keepdims=True)
    rdist = np.linalg.norm(r - center, axis=1)
    sdist = np.linalg.norm(s - center, axis=1)
    thr = np.quantile(rdist, q)
    return sdist, float(thr)


def run_qc(
    real_x: np.ndarray,
    synth: Dict[str, np.ndarray],
    sfreq: int,
    cfg: Dict,
    real_y: np.ndarray | None = None,
) -> tuple[Dict[str, np.ndarray], Dict]:
    n = int(synth["X"].shape[0])
    mask = np.ones(n, dtype=bool)
    score = np.zeros(n, dtype=np.float64)

    # Fit QC statistics on train-only real samples.
    # If class labels are provided, apply per-class QC by default.
    classwise = real_y is not None
    classes = np.unique(synth["y"]).tolist()

    if classwise:
        real_y = np.asarray(real_y).astype(np.int64)
        for cls in classes:
            cls = int(cls)
            synth_idx = synth["y"] == cls
            if not np.any(synth_idx):
                continue

            real_idx = real_y == cls
            real_ref = real_x[real_idx] if np.any(real_idx) else real_x
            if len(real_ref) == 0:
                continue

            cls_mask = np.ones(np.sum(synth_idx), dtype=bool)
            if cfg["psd"].get("enabled", True):
                fmin, fmax = cfg["psd"].get("band", [8.0, 30.0])
                z = psd_zscore(
                    real_ref,
                    synth["X"][synth_idx],
                    sfreq,
                    float(fmin),
                    float(fmax),
                )
                cls_mask &= z <= float(cfg["psd"].get("z_threshold", 2.5))
                score[synth_idx] += z

            if cfg["covariance"].get("enabled", True):
                sdist, thr = covariance_distance(
                    real_ref,
                    synth["X"][synth_idx],
                    q=float(cfg["covariance"].get("dist_threshold_quantile", 0.95)),
                )
                cls_mask &= sdist <= thr
                score[synth_idx] += sdist / max(thr, 1e-6)

            mask[synth_idx] &= cls_mask
    else:
        if cfg["psd"].get("enabled", True):
            fmin, fmax = cfg["psd"].get("band", [8.0, 30.0])
            z = psd_zscore(real_x, synth["X"], sfreq, float(fmin), float(fmax))
            mask &= z <= float(cfg["psd"].get("z_threshold", 2.5))
            score += z

        if cfg["covariance"].get("enabled", True):
            sdist, thr = covariance_distance(real_x, synth["X"], q=float(cfg["covariance"].get("dist_threshold_quantile", 0.95)))
            mask &= sdist <= thr
            score += sdist / max(thr, 1e-6)

    target_keep_ratio = float(cfg.get("target_keep_ratio", 0.0))
    fallback_applied = False
    if n > 0 and target_keep_ratio > 0.0 and float(mask.mean()) < target_keep_ratio:
        k = min(n, max(1, int(np.ceil(n * target_keep_ratio))))
        order = np.argsort(score)
        relaxed = np.zeros(n, dtype=bool)
        relaxed[order[:k]] = True
        mask = relaxed
        fallback_applied = True

    kept = {
        "X": synth["X"][mask],
        "y": synth["y"][mask],
        "sample_id": synth["sample_id"][mask],
    }

    report = {
        "n_before": int(synth["X"].shape[0]),
        "n_after": int(kept["X"].shape[0]),
        "keep_ratio": float(mask.mean()),
        "target_keep_ratio": target_keep_ratio,
        "fallback_applied": bool(fallback_applied),
    }

    for cls in np.unique(synth["y"]):
        c = int(cls)
        cm = synth["y"] == c
        report[f"class_{c}_keep_ratio"] = float(mask[cm].mean()) if cm.any() else 0.0

    return kept, report
