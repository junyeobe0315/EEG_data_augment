from __future__ import annotations

import numpy as np

from src.data.normalize import ZScoreNormalizer
from src.qc.qc_pipeline import fit_qc


def test_normalizer_fit_scope() -> None:
    x_train = np.ones((10, 2, 3), dtype=np.float32)
    x_test = np.zeros((10, 2, 3), dtype=np.float32)
    norm = ZScoreNormalizer().fit(x_train)
    x_norm = norm.transform(x_train)
    assert np.allclose(x_norm.mean(), 0.0, atol=1e-6)
    x_test_norm = norm.transform(x_test)
    # If train mean is 1, test should be shifted by -1
    assert np.allclose(x_test_norm.mean(), -1.0, atol=1e-6)


def test_qc_fit_scope() -> None:
    x_train = np.random.randn(8, 2, 50).astype(np.float32)
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    qc_cfg = {
        "psd": {"enabled": True, "band": [8.0, 30.0], "z_threshold": 2.5},
        "covariance": {"enabled": True, "dist_threshold_quantile": 0.95},
    }
    state = fit_qc(x_train, y_train, sfreq=250, cfg=qc_cfg)
    assert "psd" in state and "cov" in state
