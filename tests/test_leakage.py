from __future__ import annotations

import numpy as np

from src.data.normalize import ZScoreNormalizer
from src.qc.qc_pipeline import fit_qc


def test_normalizer_fit_scope() -> None:
    """Check that normalization is fit on train data only.

    Inputs:
    - x_train: constant-valued training data.
    - x_test: constant-valued test data with a different mean.

    Outputs:
    - Asserts training data normalizes to mean 0.
    - Asserts test data shifts by -(1/eps) due to zero std.

    Internal logic:
    - Fits on constant train data then checks transform behavior on train/test.
    """
    x_train = np.ones((10, 2, 3), dtype=np.float32)
    x_test = np.zeros((10, 2, 3), dtype=np.float32)
    norm = ZScoreNormalizer().fit(x_train)
    x_norm = norm.transform(x_train)
    assert np.allclose(x_norm.mean(), 0.0, atol=1e-6)
    x_test_norm = norm.transform(x_test)
    # If train std is 0, z-score uses eps; expected shift is -1/eps.
    expected = -(1.0 / norm.eps)
    assert np.allclose(x_test_norm.mean(), expected, atol=1e-6)


def test_qc_fit_scope() -> None:
    """Ensure QC statistics can be fit without leaking test data.

    Inputs:
    - Small synthetic x_train/y_train arrays.

    Outputs:
    - Asserts QC state contains expected PSD/Cov entries.

    Internal logic:
    - Fits QC on random data and verifies expected keys exist.
    """
    x_train = np.random.randn(8, 2, 50).astype(np.float32)
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    qc_cfg = {
        "psd": {"enabled": True, "band": [8.0, 30.0], "z_threshold": 2.5},
        "covariance": {"enabled": True, "dist_threshold_quantile": 0.95},
    }
    state = fit_qc(x_train, y_train, sfreq=250, cfg=qc_cfg)
    assert "psd" in state and "cov" in state
