from __future__ import annotations

from pathlib import Path

from src.utils.results import append_result, load_results, make_run_id


def test_results_primary_key_skip(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    row = {
        "subject": 1,
        "seed": 0,
        "r": 0.1,
        "classifier": "eegnet",
        "method": "C0",
        "generator": "none",
        "qc_on": False,
        "alpha_ratio": 0.0,
        "run_id": make_run_id(
            {
                "subject": 1,
                "seed": 0,
                "r": 0.1,
                "classifier": "eegnet",
                "method": "C0",
                "generator": "none",
                "qc_on": False,
                "alpha_ratio": 0.0,
            }
        ),
    }

    assert append_result(path, row) is True
    assert append_result(path, row) is False
    df = load_results(path)
    assert len(df) == 1
