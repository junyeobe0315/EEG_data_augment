from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.generators.base import normalize_generator_type
from src.models.generators.cwgan_gp import CWGANGenerator, CWGANCritic
from src.models.generators.ddpm import ConditionalDDPM1D


def build_generator(
    model_type: str,
    in_channels: int,
    time_steps: int,
    num_classes: int,
    model_cfg: dict,
) -> torch.nn.Module:
    mtype = normalize_generator_type(model_type)
    if mtype == "cwgan_gp":
        return CWGANGenerator(
            in_channels=in_channels,
            time_steps=time_steps,
            num_classes=num_classes,
            latent_dim=int(model_cfg.get("latent_dim", 128)),
            base_channels=int(model_cfg.get("base_channels", 64)),
            cond_dim=int(model_cfg.get("cond_dim", 16)),
        )
    if mtype == "ddpm":
        return ConditionalDDPM1D(
            in_channels=in_channels,
            time_steps=time_steps,
            num_classes=num_classes,
            base_channels=int(model_cfg.get("base_channels", 64)),
            time_dim=int(model_cfg.get("time_dim", 128)),
            diffusion_steps=int(model_cfg.get("diffusion_steps", 200)),
            beta_start=float(model_cfg.get("beta_start", 1.0e-4)),
            beta_end=float(model_cfg.get("beta_end", 0.02)),
        )
    raise ValueError(f"Unsupported generator type: {model_type}")


def build_critic(
    model_type: str,
    in_channels: int,
    time_steps: int,
    num_classes: int,
    model_cfg: dict,
) -> torch.nn.Module | None:
    mtype = normalize_generator_type(model_type)
    if mtype != "cwgan_gp":
        return None
    return CWGANCritic(
        in_channels=in_channels,
        time_steps=time_steps,
        num_classes=num_classes,
        base_channels=int(model_cfg.get("base_channels", 64)),
        cond_dim=int(model_cfg.get("cond_dim", 16)),
    )


def load_generator_checkpoint(ckpt_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    ckpt = torch.load(Path(ckpt_path), map_location=device)
    return ckpt

