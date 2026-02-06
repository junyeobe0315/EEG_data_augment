from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import scripts.tune_hparams as tune_mod


def test_tune_hparams_smoke_and_no_test_leak(monkeypatch, tmp_path: Path) -> None:
    """Smoke-test tuning script with mocked runs and enforce alpha_search-only eval."""
    calls = []

    def _stub_run_experiment(**kwargs):
        calls.append(kwargs)
        assert kwargs["stage"] == "alpha_search"
        assert kwargs["compute_distance"] is False
        base = 0.40 + float(kwargs["r"]) * 0.01
        if kwargs["method"] == "GenAug":
            base += 0.03
        return {"val_kappa": base}

    cfg = {
        "dataset": {"name": "bci2a", "index_path": "./dummy/index.csv"},
        "preprocess": {},
        "split": {},
        "qc": {},
        "experiment": {"alpha_search": {}},
        "models": {
            "eegnet": {"model": {}, "train": {}, "evaluation": {}},
            "eegconformer": {"model": {}, "train": {}, "evaluation": {}},
            "ctnet": {"model": {}, "train": {}, "evaluation": {}},
            "svm": {"model": {}, "train": {}, "evaluation": {}},
        },
        "generators": {"cwgan_gp": {"model": {}, "train": {}}, "ddpm": {"model": {}, "train": {}}},
    }

    tuning_cfg = {
        "seed": 2026,
        "objective_metric": "val_kappa",
        "pilot": {"subjects": [1], "seeds": [0], "r_list": [0.01, 0.1]},
        "screening": {"seeds": [0], "r_list": [0.01]},
        "budget": {"eegnet_trials": 3, "genaug_trials": 3, "top_k": 1},
        "genaug": {"alpha_ratio_ref": 1.0, "qc_on": True},
        "paths": {
            "tuning_root": str(tmp_path / "artifacts" / "tuning"),
            "trials_csv": str(tmp_path / "results" / "tuning_trials.csv"),
            "best_params_json": str(tmp_path / "artifacts" / "tuning" / "best_params.json"),
        },
    }
    space_cfg = {
        "eegnet": {
            "model.F1": {"type": "choice", "values": [8, 12]},
            "train.lr": {"type": "log_uniform", "low": 1e-4, "high": 1e-3},
        },
        "cwgan_gp": {
            "model.latent_dim": {"type": "choice", "values": [64, 128]},
            "train.lr": {"type": "log_uniform", "low": 1e-4, "high": 2e-4},
        },
        "qc": {
            "psd.z_threshold": {"type": "choice", "values": [2.0, 2.5]},
        },
    }

    def _stub_load_yaml(path, overrides=None):
        p = str(path)
        if p.endswith("tuning.yaml"):
            return tuning_cfg
        if p.endswith("hparam_space.yaml"):
            return space_cfg
        raise AssertionError(f"unexpected load_yaml path: {path}")

    monkeypatch.setattr(tune_mod, "_load_all_configs", lambda overrides, config_pack="base": cfg)
    monkeypatch.setattr(tune_mod, "_validate_inputs", lambda cfg, tuning_cfg: None)
    monkeypatch.setattr(tune_mod, "run_experiment", _stub_run_experiment)
    monkeypatch.setattr(tune_mod, "load_yaml", _stub_load_yaml)

    argv = [
        "tune_hparams.py",
        "--max_trials",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    tune_mod.main()

    trials_csv = tmp_path / "results" / "tuning_trials.csv"
    best_json = tmp_path / "artifacts" / "tuning" / "best_params.json"
    assert trials_csv.exists()
    assert best_json.exists()

    df = pd.read_csv(trials_csv)
    assert len(df) > 0
    best = json.loads(best_json.read_text(encoding="utf-8"))
    assert "eegnet" in best and "genaug_qc" in best
    assert len(calls) > 0
