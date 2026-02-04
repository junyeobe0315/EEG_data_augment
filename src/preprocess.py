from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ZScoreNormalizer:
    eps: float = 1e-6
    mode: str = "channel_global"
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "ZScoreNormalizer":
        # x_train: [N, C, T]
        mode = str(self.mode).strip().lower()
        if mode == "channel_global":
            self.mean_ = x_train.mean(axis=(0, 2), keepdims=True)
            self.std_ = x_train.std(axis=(0, 2), keepdims=True)
        elif mode == "channel_timepoint":
            # Match ATCNet preprocessing: per-channel, per-timepoint stats over trials.
            self.mean_ = x_train.mean(axis=0, keepdims=True)
            self.std_ = x_train.std(axis=0, keepdims=True)
        elif mode == "global_scalar":
            # Match Conformer/CTNet scripts: single scalar mean/std from full train tensor.
            self.mean_ = x_train.mean(axis=(0, 1, 2), keepdims=True)
            self.std_ = x_train.std(axis=(0, 1, 2), keepdims=True)
        else:
            raise ValueError(f"Unsupported normalization mode: {self.mode}")
        self.std_ = self.std_ + self.eps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer must be fitted on train split first.")
        return (x - self.mean_) / self.std_

    def state_dict(self) -> Dict[str, np.ndarray]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer has no fitted stats.")
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> "ZScoreNormalizer":
        self.mean_ = state["mean"]
        self.std_ = state["std"]
        return self
