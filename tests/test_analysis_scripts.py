from __future__ import annotations

import subprocess
import sys

import pandas as pd


def test_analysis_scripts_run(tmp_path) -> None:
    """summarize_results.py and plot_results.py should run on a minimal results table."""
    rows = []
    for subject in [1, 2]:
        for seed in [0, 1]:
            rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "r": 0.1,
                    "method": "C0",
                    "classifier": "eegnet",
                    "generator": "none",
                    "alpha_ratio": 0.0,
                    "qc_on": False,
                    "acc": 0.50,
                    "kappa": 0.30,
                    "macro_f1": 0.40,
                    "val_acc": 0.55,
                    "val_kappa": 0.35,
                    "val_macro_f1": 0.45,
                    "pass_rate": 1.0,
                    "oversample_factor": 1.0,
                    "distance": 0.30,
                    "dist_mmd": 0.30,
                }
            )
            rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "r": 0.1,
                    "method": "GenAug",
                    "classifier": "eegnet",
                    "generator": "cwgan_gp",
                    "alpha_ratio": 0.5,
                    "qc_on": True,
                    "acc": 0.58,
                    "kappa": 0.40,
                    "macro_f1": 0.49,
                    "val_acc": 0.60,
                    "val_kappa": 0.42,
                    "val_macro_f1": 0.50,
                    "pass_rate": 0.82,
                    "oversample_factor": 1.3,
                    "distance": 0.22,
                    "dist_mmd": 0.22,
                }
            )

    results_path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)

    summary_path = tmp_path / "summary.csv"
    paired_path = tmp_path / "paired_stats.csv"
    gain_path = tmp_path / "gain_distance.csv"
    fig_dir = tmp_path / "figs"

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--results",
            str(results_path),
            "--out",
            str(summary_path),
            "--paired_out",
            str(paired_path),
            "--gain_out",
            str(gain_path),
            "--metric",
            "kappa",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_results.py",
            "--results",
            str(results_path),
            "--metric",
            "kappa",
            "--out_dir",
            str(fig_dir),
        ],
        check=True,
    )

    assert summary_path.exists()
    assert paired_path.exists()
    assert gain_path.exists()
    assert any(fig_dir.glob("*.png"))
