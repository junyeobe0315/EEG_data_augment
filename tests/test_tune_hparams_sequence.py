from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.tune_hparams as tune_mod


def test_tune_hparams_gen_sequence_runs_multiple_generators(monkeypatch, tmp_path: Path) -> None:
    """--gen_sequence should execute GenAug trials for each listed generator in order."""
    calls = []

    def _stub_run_experiment(**kwargs):
        calls.append(kwargs)
        assert kwargs["stage"] == "alpha_search"
        assert kwargs["compute_distance"] is False
        if kwargs["method"] == "C0":
            return {"val_kappa": 0.40}
        gen_bonus = {"cwgan_gp": 0.06, "cvae": 0.04}.get(kwargs["generator"], 0.0)
        return {"val_kappa": 0.40 + gen_bonus}

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
        "generators": {
            "cwgan_gp": {"model": {}, "train": {}},
            "cvae": {"model": {}, "train": {}},
            "ddpm": {"model": {}, "train": {}},
        },
    }

    tuning_cfg = {
        "seed": 2026,
        "objective_metric": "val_kappa",
        "pilot": {"subjects": [1], "seeds": [0], "r_list": [0.01]},
        "screening": {"seeds": [0], "r_list": [0.01]},
        "budget": {"eegnet_trials": 1, "genaug_trials": 1, "top_k": 1},
        "genaug": {"alpha_ratio_ref": 1.0, "qc_on": True, "generator": "cwgan_gp"},
        "generator_epoch_policy": {"enabled": False},
        "paths": {
            "tuning_root": str(tmp_path / "artifacts" / "tuning"),
            "trials_csv": str(tmp_path / "results" / "tuning_trials.csv"),
            "best_params_json": str(tmp_path / "artifacts" / "tuning" / "best_params.json"),
        },
    }
    space_cfg = {
        "eegnet": {"model.F1": {"type": "choice", "values": [8]}},
        "cwgan_gp": {"model.latent_dim": {"type": "choice", "values": [64]}},
        "cvae": {"model.latent_dim": {"type": "choice", "values": [32]}},
        "qc": {"psd.z_threshold": {"type": "choice", "values": [2.5]}},
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
        "1",
        "--gen_sequence",
        "cwgan_gp",
        "cvae",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    tune_mod.main()

    gen_calls = [c for c in calls if c["method"] == "GenAug"]
    used_generators = {c["generator"] for c in gen_calls}
    assert used_generators == {"cwgan_gp", "cvae"}

    best_json = tmp_path / "artifacts" / "tuning" / "best_params.json"
    best = json.loads(best_json.read_text(encoding="utf-8"))
    assert best["gen_sequence"] == ["cwgan_gp", "cvae"]
    assert len(best["genaug_qc_all"]) == 2

