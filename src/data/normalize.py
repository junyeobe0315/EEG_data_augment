from __future__ import annotations

import numpy as np


class ZScoreNormalizer:
    def __init__(self, eps: float = 1.0e-6):
        """Initialize a channel-wise z-score normalizer.

        Inputs:
        - eps: small value added to std to avoid division by zero.

        Outputs:
        - ZScoreNormalizer instance with empty mean/std.

        Internal logic:
        - Stores epsilon and defers statistics until fit().
        """
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None  # [1, C, 1]
        self.std_: np.ndarray | None = None  # [1, C, 1]

    def fit(self, x: np.ndarray) -> "ZScoreNormalizer":
        """Fit mean and std from training data.

        Inputs:
        - x: ndarray [N, C, T] training data only.

        Outputs:
        - self (fitted).

        Internal logic:
        - Computes per-channel mean/std across N and T dimensions.
        """
        # x: [N, C, T]
        self.mean_ = x.mean(axis=(0, 2), keepdims=True)  # per-channel mean
        self.std_ = x.std(axis=(0, 2), keepdims=True) + self.eps  # per-channel std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize data using stored mean/std.

        Inputs:
        - x: ndarray [N, C, T].

        Outputs:
        - normalized ndarray [N, C, T] float32.

        Internal logic:
        - Applies (x - mean) / std and casts to float32.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer not fitted.")
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def state_dict(self) -> dict:
        """Return a serializable state dict with mean/std/eps.

        Inputs:
        - None (uses fitted state).

        Outputs:
        - dict with keys: mean, std, eps.

        Internal logic:
        - Packages numpy arrays into a plain Python dict for JSON/torch save.
        """
        return {"mean": self.mean_, "std": self.std_, "eps": self.eps}

    @classmethod
    def from_state(cls, state: dict) -> "ZScoreNormalizer":
        """Reconstruct a normalizer from a saved state dict.

        Inputs:
        - state: dict with mean/std/eps.

        Outputs:
        - ZScoreNormalizer with restored parameters.

        Internal logic:
        - Initializes with eps then assigns mean/std arrays directly.
        """
        inst = cls(eps=float(state.get("eps", 1.0e-6)))
        inst.mean_ = state.get("mean")
        inst.std_ = state.get("std")
        return inst
