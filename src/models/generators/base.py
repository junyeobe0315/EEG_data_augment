from __future__ import annotations


def normalize_generator_type(model_type: str) -> str:
    key = model_type.strip().lower().replace("-", "_").replace(" ", "_")
    alias = {
        "cwgan_gp": "cwgan_gp",
        "cwgangp": "cwgan_gp",
        "ddpm": "ddpm",
        "conditional_ddpm": "ddpm",
    }
    if key not in alias:
        raise ValueError(f"Unsupported generator type: {model_type}")
    return alias[key]
