from __future__ import annotations

import math
import numpy as np


def _median_heuristic_gamma(x: np.ndarray, y: np.ndarray) -> float:
    """Compute RBF gamma using the median heuristic.

    Inputs:
    - x, y: arrays [N, D] and [M, D].

    Outputs:
    - gamma: float for RBF kernel.

    Internal logic:
    - Computes pairwise squared distances on a subsample and uses median as sigma^2.
    """
    z = np.concatenate([x, y], axis=0)
    if len(z) > 1024:
        idx = np.random.choice(len(z), size=1024, replace=False)
        z = z[idx]
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    sigma2 = max(float(med), 1e-6)
    return 1.0 / (2.0 * sigma2)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF kernel matrix.

    Inputs:
    - x: [N, D]
    - y: [M, D]
    - gamma: kernel parameter.

    Outputs:
    - kernel matrix [N, M].

    Internal logic:
    - Computes squared L2 distances and applies exp(-gamma * d2).
    """
    d2 = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return np.exp(-gamma * d2)


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | str = "median_heuristic") -> float:
    """Compute MMD with an RBF kernel.

    Inputs:
    - x: [N, D]
    - y: [M, D]
    - gamma: float or "median_heuristic".

    Outputs:
    - MMD value (float).

    Internal logic:
    - Uses unbiased MMD estimator with optional median-heuristic gamma.
    """
    if len(x) < 2 or len(y) < 2:
        return math.nan
    if isinstance(gamma, str):
        gamma = _median_heuristic_gamma(x, y)
    kxx = _rbf_kernel(x, x, float(gamma))
    kyy = _rbf_kernel(y, y, float(gamma))
    kxy = _rbf_kernel(x, y, float(gamma))
    n = len(x)
    m = len(y)
    term_x = (np.sum(kxx) - np.trace(kxx)) / (n * (n - 1))
    term_y = (np.sum(kyy) - np.trace(kyy)) / (m * (m - 1))
    term_xy = (2.0 * np.sum(kxy)) / (n * m)
    return float(term_x + term_y - term_xy)


def sliced_wasserstein_distance(x: np.ndarray, y: np.ndarray, n_projections: int = 64, seed: int = 0) -> float:
    """Approximate Sliced Wasserstein distance.

    Inputs:
    - x: [N, D]
    - y: [M, D]
    - n_projections: number of random projections.
    - seed: RNG seed.

    Outputs:
    - SWD value (float).

    Internal logic:
    - Projects onto random directions, sorts projections, and averages L1 distances.
    """
    if len(x) == 0 or len(y) == 0:
        return math.nan
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    dirs = rng.normal(size=(n_projections, d)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8

    xp = x @ dirs.T
    yp = y @ dirs.T

    k = min(len(x), len(y))
    if k <= 1:
        return math.nan

    if len(x) != k:
        idx = rng.choice(len(x), size=k, replace=False)
        xp = xp[idx]
    if len(y) != k:
        idy = rng.choice(len(y), size=k, replace=False)
        yp = yp[idy]

    xp = np.sort(xp, axis=0)
    yp = np.sort(yp, axis=0)
    return float(np.mean(np.abs(xp - yp)))


def classwise_distance_summary(
    real_emb: np.ndarray,
    real_y: np.ndarray,
    aug_emb: np.ndarray,
    aug_y: np.ndarray,
    num_classes: int,
    n_projections: int = 64,
    mmd_gamma: float | str = "median_heuristic",
    seed: int = 0,
    class_priors: np.ndarray | None = None,
) -> dict:
    """Compute class-wise and aggregated distance summaries.

    Inputs:
    - real_emb/real_y: real embeddings [N, D] and labels [N].
    - aug_emb/aug_y: synthetic embeddings [M, D] and labels [M].
    - num_classes: number of classes.
    - n_projections, mmd_gamma, seed: distance settings.
    - class_priors: optional priors for weighted aggregation.

    Outputs:
    - dict with per-class distances and weighted averages.

    Internal logic:
    - Computes per-class MMD/SWD and aggregates using class priors from real data.
    """
    rows = {}
    swd_vals = []
    mmd_vals = []
    swd_weights = []
    mmd_weights = []

    if class_priors is None:
        counts = np.bincount(real_y.astype(np.int64), minlength=num_classes)
        total = max(1, int(counts.sum()))
        class_priors = counts / total

    for cls in range(num_classes):
        rx = real_emb[real_y == cls]
        qx = aug_emb[aug_y == cls]

        swd = sliced_wasserstein_distance(rx, qx, n_projections=n_projections, seed=seed + cls)
        mmd = mmd_rbf(rx, qx, gamma=mmd_gamma)

        rows[f"dist_swd_class_{cls}"] = swd
        rows[f"dist_mmd_class_{cls}"] = mmd
        if not math.isnan(swd):
            swd_vals.append(swd)
            swd_weights.append(float(class_priors[cls]))
        if not math.isnan(mmd):
            mmd_vals.append(mmd)
            mmd_weights.append(float(class_priors[cls]))

    if swd_vals:
        rows["dist_swd"] = float(np.sum(np.asarray(swd_vals) * np.asarray(swd_weights)))
    else:
        rows["dist_swd"] = math.nan
    if mmd_vals:
        rows["dist_mmd"] = float(np.sum(np.asarray(mmd_vals) * np.asarray(mmd_weights)))
    else:
        rows["dist_mmd"] = math.nan
    return rows
