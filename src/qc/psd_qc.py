from __future__ import annotations

import numpy as np
from scipy.signal import welch


def _band_power(x: np.ndarray, sfreq: int, fmin: float, fmax: float) -> np.ndarray:
    """Compute mean band power for each trial.

    Inputs:
    - x: ndarray [N, C, T]
    - sfreq: sampling rate
    - fmin/fmax: band limits

    Outputs:
    - ndarray [N] of mean band power values.

    Internal logic:
    - Uses Welch PSD, selects frequency band, and averages over channels/time.
    """
    freqs, psd = welch(x, fs=sfreq, axis=-1, nperseg=min(256, x.shape[-1]))
    m = (freqs >= fmin) & (freqs <= fmax)
    return psd[..., m].mean(axis=-1).mean(axis=-1)


def fit_psd_stats(real_x: np.ndarray, sfreq: int, fmin: float, fmax: float) -> tuple[float, float]:
    """Fit PSD mean/std on real data for QC.

    Inputs:
    - real_x: ndarray [N, C, T]
    - sfreq: sampling rate
    - fmin/fmax: band limits

    Outputs:
    - (mu, sd) of log band power.

    Internal logic:
    - Computes log band power then returns mean and std with epsilon.
    """
    rp = np.log(_band_power(real_x, sfreq, fmin, fmax) + 1e-8)
    mu = float(rp.mean())
    sd = float(rp.std() + 1e-6)
    return mu, sd


def psd_zscore(synth_x: np.ndarray, mu: float, sd: float, sfreq: int, fmin: float, fmax: float) -> np.ndarray:
    """Compute absolute z-score of band power for synthetic samples.

    Inputs:
    - synth_x: ndarray [N, C, T]
    - mu/sd: reference stats from real data
    - sfreq, fmin, fmax: PSD settings

    Outputs:
    - ndarray [N] of absolute z-scores.

    Internal logic:
    - Computes log band power for synth_x and normalizes by real stats.
    """
    sp = np.log(_band_power(synth_x, sfreq, fmin, fmax) + 1e-8)
    return np.abs((sp - mu) / sd)
