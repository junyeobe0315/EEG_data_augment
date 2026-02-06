from __future__ import annotations


def normalize_generator_type(model_type: str) -> str:
    """Normalize generator type strings to canonical keys.

    Inputs:
    - model_type: raw generator name.

    Outputs:
    - canonical generator key.

    Internal logic:
    - Lowercases and maps known aliases to supported generator identifiers.
    """
    key = model_type.strip().lower().replace("-", "_").replace(" ", "_")
    alias = {
        "cwgan_gp": "cwgan_gp",
        "cwgangp": "cwgan_gp",
        "cvae": "cvae",
        "conditional_vae": "cvae",
        "ddpm": "ddpm",
        "conditional_ddpm": "ddpm",
    }
    if key not in alias:
        raise ValueError(f"Unsupported generator type: {model_type}")
    return alias[key]
