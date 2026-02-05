from __future__ import annotations

from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """Wrap EEG arrays into a torch Dataset.

        Inputs:
        - x: ndarray [N, C, T] float-like.
        - y: ndarray [N] int-like labels.

        Outputs:
        - Dataset object yielding (x_i, y_i) tensors.

        Internal logic:
        - Converts arrays to float32/int64 for stable torch usage.
        """
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        """Return number of samples.

        Inputs:
        - None (uses stored arrays).

        Outputs:
        - int length of dataset (N).

        Internal logic:
        - Reads the first dimension of the cached numpy array.
        """
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single (x, y) sample.

        Inputs:
        - idx: sample index.

        Outputs:
        - x_i: torch.Tensor [C, T] float32
        - y_i: torch.Tensor scalar int64

        Internal logic:
        - Indexes numpy arrays and converts to torch tensors on the fly.
        """
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def load_index(index_path: str | Path) -> pd.DataFrame:
    """Load the cached index CSV.

    Inputs:
    - index_path: path to index.csv.

    Outputs:
    - pandas DataFrame with sample metadata.

    Internal logic:
    - Uses pandas CSV loading without any filtering.
    """
    return pd.read_csv(index_path)


def load_samples(index_df: pd.DataFrame, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load X/y arrays by index from cached .npz files.

    Inputs:
    - index_df: DataFrame with file path + row columns.
    - indices: array of row indices into index_df.

    Outputs:
    - x: ndarray [N, C, T] float32
    - y: ndarray [N] int64

    Internal logic:
    - Groups by file to minimize IO, then reorders to original index order.
    """
    if len(indices) == 0:
        return np.empty((0,)), np.empty((0,))

    sel = index_df.loc[indices].copy()
    sel["_order"] = np.arange(len(sel))  # preserve requested ordering

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    orders: list[np.ndarray] = []

    for file_path, group in sel.groupby("file"):
        arr = np.load(Path(file_path))  # cached per-subject/session data
        rows = group["row"].to_numpy()  # row indices within the .npz
        x = arr["X"][rows]
        y = arr["y"][rows]
        xs.append(x)
        ys.append(y)
        orders.append(group["_order"].to_numpy())

    if not xs:
        return np.empty((0,)), np.empty((0,))

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    order_all = np.concatenate(orders, axis=0)
    sort_idx = np.argsort(order_all)
    return x_all[sort_idx].astype(np.float32), y_all[sort_idx].astype(np.int64)
