from __future__ import annotations

from pathlib import Path
import numpy as np


def _r_tag(r: float) -> str:
    return str(float(r)).replace(".", "p")


def load_split_indices(
    dataset: str,
    subject: int,
    seed: int,
    r: float,
    root: str | Path = "./artifacts/splits",
) -> dict[str, np.ndarray]:
    base = Path(root) / dataset / f"subject_{subject:02d}" / f"seed{seed}"
    tr_full = np.load(base / "T_train_full_idx.npy")
    val = np.load(base / "T_val_idx.npy")
    test = np.load(base / "E_test_idx.npy")
    sub = np.load(base / f"r{_r_tag(r)}" / "T_train_sub_idx.npy")
    return {
        "train_full": tr_full,
        "train_sub": sub,
        "val": val,
        "test": test,
    }
