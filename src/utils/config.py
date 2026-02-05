from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import os
import yaml


def _coerce_value(raw: str) -> Any:
    text = raw.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    if not overrides:
        return cfg
    out = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        path, raw = item.split("=", 1)
        keys = [p for p in path.strip().split(".") if p]
        if not keys:
            continue
        cursor = out
        for k in keys[:-1]:
            if k not in cursor or not isinstance(cursor[k], dict):
                cursor[k] = {}
            cursor = cursor[k]
        cursor[keys[-1]] = _coerce_value(raw)
    return out


def load_yaml(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    env_overrides = []
    if "EEG_CFG_OVERRIDES" in os.environ:
        try:
            env_overrides = json.loads(os.environ["EEG_CFG_OVERRIDES"])
        except Exception:
            env_overrides = []
    merged = apply_overrides(cfg, env_overrides)
    return apply_overrides(merged, overrides)


def save_yaml(path: str | Path, cfg: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def config_hash(cfg: dict) -> str:
    import hashlib

    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
