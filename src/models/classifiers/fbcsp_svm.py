from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class FBCSPExtractor:
    """Filter-Bank CSP feature extractor for MI decoding."""

    def __init__(self, sfreq: int = 250, bands: list[list[float]] | None = None, n_components: int = 4):
        self.sfreq = int(sfreq)
        self.bands = bands or [
            [4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]
        ]
        self.n_components = int(n_components)
        self.csp_list: list[CSP] = []

    def _bandpass(self, x: np.ndarray, l: float, h: float) -> np.ndarray:
        sos = butter(4, [l, h], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, x, axis=-1).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FBCSPExtractor":
        self.csp_list = []
        for l, h in self.bands:
            xb = self._bandpass(x, float(l), float(h))
            csp = CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False, cov_est="epoch")
            csp.fit(xb, y)
            self.csp_list.append(csp)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.csp_list:
            raise RuntimeError("FBCSPExtractor must be fitted before transform")
        feats = []
        for (l, h), csp in zip(self.bands, self.csp_list):
            xb = self._bandpass(x, float(l), float(h))
            feats.append(csp.transform(xb))
        return np.concatenate(feats, axis=1).astype(np.float32)


@dataclass
class SVMClassifier:
    c: float = 1.0
    kernel: str = "linear"
    gamma: str = "scale"
    sfreq: int = 250
    bands: list[list[float]] = field(default_factory=lambda: [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]])
    n_components: int = 4

    def __post_init__(self) -> None:
        self.extractor = FBCSPExtractor(sfreq=self.sfreq, bands=self.bands, n_components=self.n_components)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(C=self.c, kernel=self.kernel, gamma=self.gamma, class_weight="balanced")),
            ]
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        feat = self.extractor.fit(x, y).transform(x)
        self.pipeline.fit(feat, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        feat = self.extractor.transform(x)
        return self.pipeline.predict(feat)
