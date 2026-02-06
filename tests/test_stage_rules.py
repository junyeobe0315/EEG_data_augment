from __future__ import annotations

import pytest

from scripts.run_grid import _run_group


def _fake_row(kwargs: dict) -> dict:
    return {
        "subject": kwargs["subject"],
        "seed": kwargs["seed"],
        "r": kwargs["r"],
        "classifier": kwargs["classifier"],
        "method": kwargs["method"],
        "generator": kwargs["generator"],
        "qc_on": kwargs["qc_on"],
        "alpha_ratio": kwargs["alpha_ratio"],
    }


def test_alpha_search_runs_genaug_only_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage alpha_search should skip non-GenAug methods unless explicitly enabled."""
    calls: list[dict] = []

    def _stub_run_experiment(**kwargs):
        calls.append(kwargs)
        return _fake_row(kwargs)

    monkeypatch.setattr("scripts.run_grid.run_experiment", _stub_run_experiment)

    cfg = {
        "experiment": {
            "methods": ["C0", "C1", "C2", "GenAug"],
            "classifiers": ["eegnet", "svm"],
            "generators": ["cwgan_gp"],
            "alpha_ratio_list": [0.25, 0.5],
            "qc_on": [False, True],
            "stage": {"screening_classifier": "eegnet"},
        },
        "dataset": {},
        "preprocess": {},
        "split": {},
        "models": {},
        "generators": {},
        "qc": {},
    }

    rows = _run_group(
        subject=1,
        seed=0,
        r=0.1,
        cfg=cfg,
        stage="alpha_search",
        alpha_star={},
        compute_distance=False,
        alpha_search_cfg={},
        existing_keys=set(),
        include_baselines_in_alpha_search=False,
        config_pack="base",
    )

    assert rows
    assert all(row["method"] == "GenAug" for row in rows)
    assert all(call["method"] == "GenAug" for call in calls)
    assert all(call["classifier"] == "eegnet" for call in calls)


def test_final_eval_requires_alpha_star(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage final_eval must fail when alpha* is missing for r."""

    def _stub_run_experiment(**kwargs):
        return _fake_row(kwargs)

    monkeypatch.setattr("scripts.run_grid.run_experiment", _stub_run_experiment)

    cfg = {
        "experiment": {
            "methods": ["GenAug"],
            "classifiers": ["eegnet"],
            "generators": ["cwgan_gp"],
            "alpha_ratio_list": [0.25],
            "qc_on": [False],
            "stage": {"screening_classifier": "eegnet"},
        },
        "dataset": {},
        "preprocess": {},
        "split": {},
        "models": {},
        "generators": {},
        "qc": {},
    }

    with pytest.raises(KeyError):
        _run_group(
            subject=1,
            seed=0,
            r=0.1,
            cfg=cfg,
            stage="final_eval",
            alpha_star={},
            compute_distance=False,
            alpha_search_cfg={},
            existing_keys=set(),
            include_baselines_in_alpha_search=False,
            config_pack="base",
        )
