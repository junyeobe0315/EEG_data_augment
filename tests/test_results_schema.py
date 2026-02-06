from __future__ import annotations

from pathlib import Path

from src.utils.results import append_result, load_results, make_run_id


def test_results_primary_key_skip(tmp_path: Path) -> None:
    """Ensure primary-key collisions are skipped when appending results.

    Inputs:
    - tmp_path: pytest temp directory for isolated file IO.

    Outputs:
    - Asserts only one row is stored for duplicate primary keys.

    Internal logic:
    - Append the same row twice and verify the second append is skipped.
    """
    path = tmp_path / "results.csv"
    row = {
        "config_pack": "base",
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
                "config_pack": "base",
            }
        ),
    }

    assert append_result(path, row) is True
    assert append_result(path, row) is False
    df = load_results(path)
    assert len(df) == 1
