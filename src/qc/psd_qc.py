from __future__ import annotations

import numpy as np
from scipy.signal import welch


def _band_power(x: np.ndarray, sfreq: int, fmin: float, fmax: float) -> np.ndarray:
    freqs, psd = welch(x, fs=sfreq, axis=-1, nperseg=min(256, x.shape[-1]))
    m = (freqs >= fmin) & (freqs <= fmax)
    return psd[..., m].mean(axis=-1).mean(axis=-1)


def fit_psd_stats(real_x: np.ndarray, sfreq: int, fmin: float, fmax: float) -> tuple[float, float]:
    rp = np.log(_band_power(real_x, sfreq, fmin, fmax) + 1e-8)
    mu = float(rp.mean())
    sd = float(rp.std() + 1e-6)
    return mu, sd


def psd_zscore(synth_x: np.ndarray, mu: float, sd: float, sfreq: int, fmin: float, fmax: float) -> np.ndarray:
    sp = np.log(_band_power(synth_x, sfreq, fmin, fmax) + 1e-8)
    return np.abs((sp - mu) / sd)
