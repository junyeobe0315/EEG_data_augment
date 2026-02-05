from __future__ import annotations

import numpy as np


def _cov_feat(x: np.ndarray) -> np.ndarray:
    """Compute flattened covariance features for each trial.

    Inputs:
    - x: ndarray [N, C, T]

    Outputs:
    - ndarray [N, C*C] flattened covariance per trial.

    Internal logic:
    - Computes per-trial covariance matrices and flattens them.
    """
    feats = []
    for i in range(x.shape[0]):
        c = np.cov(x[i])
        feats.append(c.reshape(-1))
    return np.asarray(feats)


def fit_cov_stats(real_x: np.ndarray, q: float = 0.95) -> tuple[np.ndarray, float]:
    """Fit covariance center and distance threshold for QC.

    Inputs:
    - real_x: ndarray [N, C, T]
    - q: quantile for threshold.

    Outputs:
    - center: ndarray [1, C*C] mean covariance feature
    - thr: float distance threshold at quantile q

    Internal logic:
    - Computes covariance features, their mean, and a quantile threshold.
    """
    r = _cov_feat(real_x)
    center = r.mean(axis=0, keepdims=True)
    rdist = np.linalg.norm(r - center, axis=1)
    thr = np.quantile(rdist, q)
    return center, float(thr)


def cov_distance(synth_x: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Compute distance of synthetic samples to covariance center.

    Inputs:
    - synth_x: ndarray [N, C, T]
    - center: ndarray [1, C*C] reference covariance feature

    Outputs:
    - ndarray [N] of L2 distances.

    Internal logic:
    - Computes covariance features for synth_x and L2 distance to center.
    """
    s = _cov_feat(synth_x)
    return np.linalg.norm(s - center, axis=1)
