from __future__ import annotations

import scripts.tune_hparams as tune_mod


def test_apply_generator_epoch_policy_for_gan_and_vae() -> None:
    """Generator epoch policy should set long max-epochs + early stopping for GAN/VAE."""
    tuning_cfg = {
        "generator_epoch_policy": {
            "enabled": True,
            "max_epochs": {"cwgan_gp": 10000, "cvae": 10000, "ddpm": 400},
            "early_stopping": {
                "enabled": True,
                "patience_epochs": 500,
                "min_delta": 0.0,
                "mode": "min",
                "monitor": {"cwgan_gp": "loss_g", "cvae": "loss"},
            },
            "save_every": 50,
            "disable_ddpm_early_stopping": True,
        }
    }

    gan_out = tune_mod._apply_generator_epoch_policy({"train": {"epochs": 200}}, tuning_cfg, "cwgan_gp")
    assert int(gan_out["train"]["epochs"]) == 10000
    assert bool(gan_out["train"]["early_stopping"]["enabled"]) is True
    assert int(gan_out["train"]["early_stopping"]["patience_epochs"]) == 500
    assert str(gan_out["train"]["early_stopping"]["monitor"]) == "loss_g"
    assert int(gan_out["train"]["save_every"]) == 50

    vae_out = tune_mod._apply_generator_epoch_policy({"train": {"epochs": 120}}, tuning_cfg, "cvae")
    assert int(vae_out["train"]["epochs"]) == 10000
    assert str(vae_out["train"]["early_stopping"]["monitor"]) == "loss"

    ddpm_out = tune_mod._apply_generator_epoch_policy({"train": {"epochs": 1000}}, tuning_cfg, "ddpm")
    assert int(ddpm_out["train"]["epochs"]) == 400
    assert bool(ddpm_out["train"]["early_stopping"]["enabled"]) is False

