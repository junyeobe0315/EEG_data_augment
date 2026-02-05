from __future__ import annotations

import hashlib
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


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=indent)


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


def stable_hash_seed(base_seed: int, payload: Dict[str, Any], digest_len: int = 8) -> int:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    digest = int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:digest_len], 16)
    return int((int(base_seed) + digest) % (2**32 - 1))


def in_allowed_grid(split: dict, split_cfg: dict, data_cfg: dict) -> bool:
    seed = int(split.get("seed", -1))
    p = float(split.get("low_data_frac", 1.0))
    subject = split.get("subject", None)

    allowed_seeds = set(int(s) for s in split_cfg.get("seeds", []))
    allowed_p = [float(x) for x in split_cfg.get("low_data_fracs", [])]
    allowed_subjects = set(int(s) for s in data_cfg.get("subjects", []))

    if allowed_seeds and seed not in allowed_seeds:
        return False
    if allowed_p and not any(abs(p - ap) < 1e-12 for ap in allowed_p):
        return False
    if subject is not None and allowed_subjects:
        try:
            if int(subject) not in allowed_subjects:
                return False
        except Exception:
            return False
    return True


def find_split_files(root: str | Path, split_cfg: dict) -> list[Path]:
    split_dir = Path(root) / "data" / "splits"
    files = sorted(split_dir.glob("subject_*_seed_*_p_*.json"))
    if files:
        return files
    protocol = str(split_cfg.get("protocol", "")).strip()
    if protocol:
        legacy = sorted(split_dir.glob(f"split_{protocol}_seed*.json"))
        if legacy:
            return legacy
    return []


def require_split_files(root: str | Path, split_cfg: dict) -> list[Path]:
    files = find_split_files(root, split_cfg)
    if not files:
        raise RuntimeError("No split files found. Run `python main.py make-splits` first.")
    return files


def p_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def parse_p_tag(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        raise ValueError("Empty p-tag value")
    if "p" in s and "." not in s:
        s = s.replace("p", ".")
    return float(s)


def split_file_path(root: str | Path, subject: int, seed: int, p: float) -> Path:
    split_dir = Path(root) / "data" / "splits"
    return split_dir / f"subject_{int(subject):02d}_seed_{int(seed)}_p_{p_tag(p)}.json"


def load_split_any(path_like: str | Path, split_name: str, root: str | Path) -> Any:
    p = Path(path_like)
    if not p.exists():
        p = Path(root) / "data" / "splits" / f"{split_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Split not found: {path_like} / {p}")
    return load_json(p)


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
