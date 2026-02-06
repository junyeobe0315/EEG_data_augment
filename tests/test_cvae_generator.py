from __future__ import annotations

import numpy as np

from src.models.generators.base import normalize_generator_type
from src.train.train_generator import train_generator, LoadedGeneratorSampler


def test_normalize_generator_type_cvae_alias() -> None:
    """cVAE aliases should normalize to canonical key."""
    assert normalize_generator_type("cvae") == "cvae"
    assert normalize_generator_type("conditional_vae") == "cvae"


def test_cvae_train_and_sample(tmp_path) -> None:
    """cVAE should train for one epoch and sample expected shapes."""
    x = np.random.randn(16, 4, 128).astype(np.float32)
    y = np.random.randint(0, 3, size=(16,), dtype=np.int64)

    out = train_generator(
        x_train=x,
        y_train=y,
        model_type="cvae",
        model_cfg={
            "latent_dim": 16,
            "base_channels": 16,
            "cond_dim": 8,
            "beta_kl": 1.0e-3,
        },
        train_cfg={
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "save_every": 1,
            "num_workers": 0,
        },
        run_dir=tmp_path,
        seed=0,
    )
    assert len(out["ckpts"]) == 1

    sampler = LoadedGeneratorSampler(out["ckpts"][0], device="cpu")
    ys = np.asarray([0, 1, 2], dtype=np.int64)
    syn = sampler.sample(ys)
    assert syn.shape == (3, 4, 128)
    assert syn.dtype == np.float32

