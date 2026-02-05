from __future__ import annotations

import numpy as np


class ZScoreNormalizer:
    def __init__(self, eps: float = 1.0e-6):
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "ZScoreNormalizer":
        # x: [N, C, T]
        self.mean_ = x.mean(axis=(0, 2), keepdims=True)
        self.std_ = x.std(axis=(0, 2), keepdims=True) + self.eps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer not fitted.")
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean_, "std": self.std_, "eps": self.eps}

    @classmethod
    def from_state(cls, state: dict) -> "ZScoreNormalizer":
        inst = cls(eps=float(state.get("eps", 1.0e-6)))
        inst.mean_ = state.get("mean")
        inst.std_ = state.get("std")
        return inst
