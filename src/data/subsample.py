from __future__ import annotations

from pathlib import Path
import numpy as np


def _r_tag(r: float) -> str:
    """Convert a float r to a filename-safe tag (e.g., 0.1 -> 0p1).

    Inputs:
    - r: float ratio.

    Outputs:
    - string tag used in split directory names.

    Internal logic:
    - Converts float to string and replaces '.' with 'p' for filesystem safety.
    """
    return str(float(r)).replace(".", "p")


def load_split_indices(
    dataset: str,
    subject: int,
    seed: int,
    r: float,
    root: str | Path = "./artifacts/splits",
) -> dict[str, np.ndarray]:
    """Load saved split indices for a subject/seed/r.

    Inputs:
    - dataset: dataset name.
    - subject: subject integer.
    - seed: split seed.
    - r: low-data fraction.
    - root: root split directory.

    Outputs:
    - dict with train_full/train_sub/val/test index arrays.

    Internal logic:
    - Loads precomputed .npy files from the split directory tree.
    """
    base = Path(root) / dataset / f"subject_{subject:02d}" / f"seed{seed}"
    tr_full = np.load(base / "T_train_full_idx.npy")  # full T train indices
    val = np.load(base / "T_val_idx.npy")  # T validation indices
    test = np.load(base / "E_test_idx.npy")  # E test indices
    sub = np.load(base / f"r{_r_tag(r)}" / "T_train_sub_idx.npy")  # low-data subset indices
    return {
        "train_full": tr_full,
        "train_sub": sub,
        "val": val,
        "test": test,
    }
