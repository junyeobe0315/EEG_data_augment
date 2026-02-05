from __future__ import annotations

import numpy as np


def _cov_feat(x: np.ndarray) -> np.ndarray:
    feats = []
    for i in range(x.shape[0]):
        c = np.cov(x[i])
        feats.append(c.reshape(-1))
    return np.asarray(feats)


def fit_cov_stats(real_x: np.ndarray, q: float = 0.95) -> tuple[np.ndarray, float]:
    r = _cov_feat(real_x)
    center = r.mean(axis=0, keepdims=True)
    rdist = np.linalg.norm(r - center, axis=1)
    thr = np.quantile(rdist, q)
    return center, float(thr)


def cov_distance(synth_x: np.ndarray, center: np.ndarray) -> np.ndarray:
    s = _cov_feat(synth_x)
    return np.linalg.norm(s - center, axis=1)
