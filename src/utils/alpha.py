from __future__ import annotations


def alpha_ratio_to_mix(alpha_ratio: float) -> float:
    """Convert synthetic:real ratio to mixture weight.

    Inputs:
    - alpha_ratio: float, N_synth / N_real.

    Outputs:
    - alpha_mix: float, N_synth / (N_real + N_synth).

    Internal logic:
    - Normalizes ratio to mixture weight and clamps non-positive values to 0.
    """
    # alpha_ratio is the input; output is normalized mixture weight.
    alpha_ratio = float(alpha_ratio)
    if alpha_ratio <= 0:
        return 0.0
    return alpha_ratio / (1.0 + alpha_ratio)
