from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.apply_tuned_configs as apply_mod


def test_apply_tuned_configs_smoke(monkeypatch, tmp_path: Path) -> None:
    """apply_tuned_configs should write tuned eegnet/cwgan/qc files."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "generators").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "tuning").mkdir(parents=True, exist_ok=True)

    (tmp_path / "configs" / "models" / "eegnet.yaml").write_text(
        "model:\n  F1: 8\ntrain:\n  lr: 0.001\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "generators" / "cwgan_gp.yaml").write_text(
        "model:\n  latent_dim: 128\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "qc.yaml").write_text(
        "psd:\n  z_threshold: 2.5\n",
        encoding="utf-8",
    )
    best = {
        "eegnet": {"params": {"model.F1": 16, "train.lr": 0.0007}},
        "genaug_qc": {"params": {"generator": {"model.latent_dim": 64}, "qc": {"psd.z_threshold": 3.0}}},
    }
    (tmp_path / "artifacts" / "tuning" / "best_params.json").write_text(
        json.dumps(best, ensure_ascii=True),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_tuned_configs.py",
            "--best",
            "artifacts/tuning/best_params.json",
            "--out_dir",
            "configs/tuned",
        ],
    )
    apply_mod.main()

    assert (tmp_path / "configs" / "tuned" / "models" / "eegnet.yaml").exists()
    assert (tmp_path / "configs" / "tuned" / "generators" / "cwgan_gp.yaml").exists()
    assert (tmp_path / "configs" / "tuned" / "qc.yaml").exists()
