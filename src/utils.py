from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_exp_id(prefix: str, **kwargs: Any) -> str:
    parts = [prefix] + [f"{k}-{v}" for k, v in kwargs.items()]
    return "__".join(parts)


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    dev = str(device or "auto")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def proportional_allocation(counts: np.ndarray, total: int) -> np.ndarray:
    counts = counts.astype(np.float64)
    out = np.zeros_like(counts, dtype=np.int64)
    if total <= 0 or float(counts.sum()) <= 0:
        return out
    raw = counts / counts.sum() * float(total)
    base = np.floor(raw).astype(np.int64)
    remain = int(total - int(base.sum()))
    if remain > 0:
        frac = raw - base
        order = np.argsort(-frac)
        base[order[:remain]] += 1
    return base.astype(np.int64)


def build_ckpt_payload(
    norm: Any,
    shape: Dict[str, int],
    n_classes: int,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = {
        "normalizer": norm.state_dict(),
        "shape": dict(shape),
        "n_classes": int(n_classes),
    }
    if extra:
        payload.update(extra)
    return payload
