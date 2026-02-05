from __future__ import annotations

import numpy as np


def mixup_batch(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = max(float(alpha), 1e-3)
    n = x.shape[0]
    lam = np.random.beta(alpha, alpha, size=n).astype(np.float32)
    idx = np.random.permutation(n)
    x2 = x[idx]
    y2 = y[idx]

    lam_x = lam[:, None, None]
    x_mix = lam_x * x + (1.0 - lam_x) * x2

    y_one = np.eye(num_classes, dtype=np.float32)[y]
    y_two = np.eye(num_classes, dtype=np.float32)[y2]
    y_mix = lam[:, None] * y_one + (1.0 - lam)[:, None] * y_two

    return x_mix.astype(np.float32), y_mix.astype(np.float32)
