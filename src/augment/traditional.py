from __future__ import annotations

import numpy as np


def apply_traditional_augment(
    x: np.ndarray,
    noise_std: float = 0.01,
    max_time_shift: int = 20,
    channel_dropout_prob: float = 0.1,
) -> np.ndarray:
    out = x.copy().astype(np.float32)

    if noise_std > 0:
        out += np.random.randn(*out.shape).astype(np.float32) * float(noise_std)

    if max_time_shift > 0:
        shifts = np.random.randint(-max_time_shift, max_time_shift + 1, size=out.shape[0])
        for i, s in enumerate(shifts):
            out[i] = np.roll(out[i], shift=int(s), axis=-1)

    if channel_dropout_prob > 0:
        drop_mask = np.random.rand(out.shape[0], out.shape[1]) < channel_dropout_prob
        out[drop_mask, :] = 0.0

    return out
