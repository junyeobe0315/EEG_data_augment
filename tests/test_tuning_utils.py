from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.tune_hparams import sample_from_space
from src.utils.tuning_results import append_tuning_trial


def test_sample_from_space_bounds() -> None:
    """Random sampler should always stay within declared bounds."""
    rng = np.random.default_rng(7)
    space = {
        "a": {"type": "choice", "values": [1, 2, 3]},
        "b": {"type": "log_uniform", "low": 1e-4, "high": 1e-2},
    }
    for _ in range(100):
        p = sample_from_space(space, rng)
        assert p["a"] in {1, 2, 3}
        assert 1e-4 <= float(p["b"]) <= 1e-2


def test_append_tuning_trial_resume_safe(tmp_path: Path) -> None:
    """Duplicate tuning trial PK should be skipped."""
    path = tmp_path / "tuning_trials.csv"
    row = {
        "trial_id": "eegnet_screen_000",
        "target": "eegnet",
        "phase": "screen",
        "status": "ok",
    }
    assert append_tuning_trial(path, row) is True
    assert append_tuning_trial(path, row) is False
    df = pd.read_csv(path)
    assert len(df) == 1
