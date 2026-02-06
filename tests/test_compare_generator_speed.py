from __future__ import annotations

import subprocess
import sys

import pandas as pd


def test_compare_generator_speed_script(tmp_path) -> None:
    """compare_generator_speed.py should aggregate runtime by generator."""
    rows = [
        {"method": "GenAug", "classifier": "eegnet", "generator": "cwgan_gp", "runtime_sec": 12.0, "r": 0.1},
        {"method": "GenAug", "classifier": "eegnet", "generator": "cwgan_gp", "runtime_sec": 14.0, "r": 0.1},
        {"method": "GenAug", "classifier": "eegnet", "generator": "cvae", "runtime_sec": 4.0, "r": 0.1},
        {"method": "GenAug", "classifier": "eegnet", "generator": "cvae", "runtime_sec": 5.0, "r": 0.1},
    ]
    results_path = tmp_path / "results.csv"
    out_path = tmp_path / "speed.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/compare_generator_speed.py",
            "--results",
            str(results_path),
            "--out",
            str(out_path),
            "--classifier",
            "eegnet",
        ],
        check=True,
    )

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert set(df["generator"]) == {"cwgan_gp", "cvae"}
    assert "mean_runtime_sec" in df.columns

