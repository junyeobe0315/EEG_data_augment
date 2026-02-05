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
        """Initialize the FBCSP feature extractor.

        Inputs:
        - sfreq: sampling rate.
        - bands: list of [low, high] band edges.
        - n_components: CSP components per band.

        Outputs:
        - FBCSPExtractor with empty CSP list.

        Internal logic:
        - Stores band definitions and defers CSP fitting until fit().
        """
        self.sfreq = int(sfreq)
        self.bands = bands or [
            [4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]
        ]
        self.n_components = int(n_components)
        self.csp_list: list[CSP] = []

    def _bandpass(self, x: np.ndarray, l: float, h: float) -> np.ndarray:
        """Bandpass filter EEG signals.

        Inputs:
        - x: ndarray [N, C, T]
        - l/h: band edges

        Outputs:
        - ndarray [N, C, T] filtered.

        Internal logic:
        - Applies a 4th-order Butterworth bandpass with zero-phase filtering.
        """
        sos = butter(4, [l, h], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, x, axis=-1).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FBCSPExtractor":
        """Fit CSP models for each filter band.

        Inputs:
        - x: ndarray [N, C, T]
        - y: ndarray [N]

        Outputs:
        - self (fitted).

        Internal logic:
        - Filters by band then fits a CSP model per band and stores them.
        """
        self.csp_list = []
        for l, h in self.bands:
            xb = self._bandpass(x, float(l), float(h))
            csp = CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False, cov_est="epoch")
            csp.fit(xb, y)
            self.csp_list.append(csp)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform EEG to FBCSP feature vectors.

        Inputs:
        - x: ndarray [N, C, T]

        Outputs:
        - ndarray [N, D] concatenated CSP features.

        Internal logic:
        - Applies each band CSP transform and concatenates features.
        """
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
        """Initialize CSP extractor and SVM pipeline.

        Inputs:
        - None (uses dataclass fields).

        Outputs:
        - None (initializes extractor and sklearn pipeline).

        Internal logic:
        - Creates FBCSPExtractor and a StandardScaler+SVC pipeline.
        """
        self.extractor = FBCSPExtractor(sfreq=self.sfreq, bands=self.bands, n_components=self.n_components)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(C=self.c, kernel=self.kernel, gamma=self.gamma, class_weight="balanced")),
            ]
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit FBCSP features and SVM classifier.

        Inputs:
        - x: ndarray [N, C, T]
        - y: ndarray [N]

        Outputs:
        - None (fits extractor and SVM pipeline).

        Internal logic:
        - Fits CSP per band, transforms to features, then fits the SVM.
        """
        feat = self.extractor.fit(x, y).transform(x)
        self.pipeline.fit(feat, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels from EEG.

        Inputs:
        - x: ndarray [N, C, T]

        Outputs:
        - ndarray [N] predicted labels.

        Internal logic:
        - Transforms EEG to FBCSP features and calls sklearn predict.
        """
        feat = self.extractor.transform(x)
        return self.pipeline.predict(feat)
