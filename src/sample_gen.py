from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.models_gen import (
    CVAE1D,
    CWGANGenerator,
    ConditionalDDPM1D,
    EEGGANNetGenerator,
    normalize_generator_type,
)
from src.preprocess import ZScoreNormalizer
from src.utils import resolve_device


def load_generator(ckpt_path: str | Path, device: str | torch.device = "cpu") -> Tuple[Any, ZScoreNormalizer, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = normalize_generator_type(str(ckpt.get("model_type", ckpt["gen_cfg"]["model"].get("type", "cvae"))))
    kwargs = dict(ckpt.get("model_kwargs", {}))
    if not kwargs:
        shape = ckpt.get("shape", {})
        model_cfg = ckpt.get("gen_cfg", {}).get("model", {})
        kwargs = {
            "in_channels": int(shape.get("c")),
            "time_steps": int(shape.get("t")),
            "num_classes": int(ckpt.get("num_classes")),
        }
        if model_type == "cvae":
            sec = model_cfg.get("cvae", model_cfg)
            kwargs["latent_dim"] = int(sec.get("latent_dim", 64))
            kwargs["hidden_dim"] = int(sec.get("hidden_dim", 128))
            kwargs["cond_dim"] = int(sec.get("cond_dim", 16))
        if model_type in {"eeggan_net", "cwgan_gp"}:
            sec = model_cfg.get(model_type, model_cfg)
            kwargs["latent_dim"] = int(sec.get("latent_dim", 128))
            kwargs["base_channels"] = int(sec.get("base_channels", sec.get("hidden_dim", 64)))
            kwargs["cond_dim"] = int(sec.get("cond_dim", 16))
        if model_type == "conditional_ddpm":
            sec = model_cfg.get("conditional_ddpm", model_cfg)
            kwargs["base_channels"] = int(sec.get("base_channels", sec.get("hidden_dim", 64)))
            kwargs["time_dim"] = int(sec.get("time_dim", 128))
            kwargs["diffusion_steps"] = int(sec.get("diffusion_steps", 200))
            kwargs["beta_start"] = float(sec.get("beta_start", 1e-4))
            kwargs["beta_end"] = float(sec.get("beta_end", 0.02))

    if model_type == "cvae":
        model = CVAE1D(**kwargs).to(device)
        model.load_state_dict(ckpt["state_dict"])
    elif model_type == "eeggan_net":
        model = EEGGANNetGenerator(**kwargs).to(device)
        model.load_state_dict(ckpt["generator_state_dict"])
    elif model_type == "cwgan_gp":
        model = CWGANGenerator(**kwargs).to(device)
        model.load_state_dict(ckpt["generator_state_dict"])
    elif model_type == "conditional_ddpm":
        model = ConditionalDDPM1D(**kwargs).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        raise ValueError(f"Unsupported generator type in checkpoint: {model_type}")

    model.eval()
    norm = ZScoreNormalizer().load_state_dict(ckpt["normalizer"])
    return model, norm, ckpt


@torch.no_grad()
def sample_by_class(
    ckpt_path: str | Path,
    n_per_class: int,
    num_classes: int,
    device: str = "auto",
) -> Dict[str, np.ndarray]:
    device = resolve_device(device)
    model, norm, ckpt = load_generator(ckpt_path, device=device)
    model_type = normalize_generator_type(str(ckpt.get("model_type", "cvae")))

    xs = []
    ys = []
    for cls in range(num_classes):
        y = torch.full((n_per_class,), cls, dtype=torch.long, device=device)

        if model_type == "conditional_ddpm":
            ddpm_steps = int(
                ckpt.get("gen_cfg", {}).get("sample", {}).get(
                    "ddpm_steps",
                    ckpt.get("model_kwargs", {}).get("diffusion_steps", 100),
                )
            )
            x_norm = model.sample(y, num_steps=ddpm_steps)
        else:
            x_norm = model.sample(y)

        x = x_norm.cpu().numpy() * norm.std_ + norm.mean_
        xs.append(x.astype(np.float32))
        ys.append(np.full((n_per_class,), cls, dtype=np.int64))

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    sid = np.asarray([f"synth_{i:07d}" for i in range(x_all.shape[0])])
    return {"X": x_all, "y": y_all, "sample_id": sid}


def save_synth_npz(path: str | Path, synth: Dict[str, np.ndarray]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=synth["X"], y=synth["y"], sample_id=synth["sample_id"])
