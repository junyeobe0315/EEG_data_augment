from __future__ import annotations

from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def load_index(index_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(index_path)


def load_samples(index_df: pd.DataFrame, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) == 0:
        return np.empty((0,)), np.empty((0,))

    sel = index_df.loc[indices].copy()
    sel["_order"] = np.arange(len(sel))

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    orders: list[np.ndarray] = []

    for file_path, group in sel.groupby("file"):
        arr = np.load(Path(file_path))
        rows = group["row"].to_numpy()
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
