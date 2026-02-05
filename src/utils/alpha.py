from __future__ import annotations


def alpha_ratio_to_mix(alpha_ratio: float) -> float:
    """Convert synthetic:real ratio to mixture weight.

    alpha_ratio = N_synth / N_real
    alpha_mix = N_synth / (N_real + N_synth)
    """
    alpha_ratio = float(alpha_ratio)
    if alpha_ratio <= 0:
        return 0.0
    return alpha_ratio / (1.0 + alpha_ratio)
