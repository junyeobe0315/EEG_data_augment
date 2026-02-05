from __future__ import annotations

import copy
import json
import os
from typing import Any

import yaml


def apply_paper_preset(
    cfg: dict,
    model_type: str,
    epoch_cap: int | None = None,
    include_scheduler: bool = False,
    disable_step_control: bool = False,
) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("model", {})["type"] = model_type

    pp = out.get("paper_presets", {}).get(model_type, {})
    for key in ("epochs", "batch_size", "lr", "weight_decay", "num_workers", "device"):
        if key in pp:
            out.setdefault("train", {})[key] = pp[key]

    if include_scheduler and "scheduler" in pp:
        out.setdefault("train", {})["scheduler"] = copy.deepcopy(pp["scheduler"])
    if disable_step_control:
        out.setdefault("train", {}).setdefault("step_control", {})["enabled"] = False

    if epoch_cap is not None and epoch_cap > 0:
        out.setdefault("train", {})
        out["train"]["epochs"] = min(int(out["train"].get("epochs", epoch_cap)), int(epoch_cap))

    return out


def build_gen_cfg(
    base_cfg: dict,
    gen_model: str,
    sweep_cfg: dict,
    override_batch_size: int | None = None,
    apply_train: bool = True,
    apply_sample: bool = True,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("model", {})["type"] = gen_model

    if not bool(sweep_cfg.get("apply_vram_presets", True)):
        return cfg

    profile = str(sweep_cfg.get("vram_profile", "6gb"))
    preset = cfg.get("vram_presets", {}).get(profile, {}).get(gen_model, {})

    if apply_train and "batch_size" in preset:
        cfg.setdefault("train", {})["batch_size"] = int(preset["batch_size"])
    if apply_sample and "n_per_class" in preset:
        cfg.setdefault("sample", {})["n_per_class"] = int(preset["n_per_class"])
    if apply_sample and "ddpm_steps" in preset:
        cfg.setdefault("sample", {})["ddpm_steps"] = int(preset["ddpm_steps"])
    if apply_sample and "diffusion_steps" in preset:
        cfg.setdefault("model", {}).setdefault("conditional_ddpm", {})["diffusion_steps"] = int(preset["diffusion_steps"])

    if override_batch_size is not None and override_batch_size > 0:
        cfg.setdefault("train", {})["batch_size"] = int(override_batch_size)

    return cfg


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def parse_overrides(pairs: list[str]) -> dict[str, list[tuple[str, Any]]]:
    overrides: dict[str, list[tuple[str, Any]]] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Invalid override (expected key=value): {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if "." not in key:
            raise ValueError(f"Override key must start with config name (e.g., gen.train.lr): {key}")
        cfg_name, path = key.split(".", 1)
        cfg_name = cfg_name.strip()
        path = path.strip()
        if not cfg_name or not path:
            raise ValueError(f"Invalid override key: {key}")
        overrides.setdefault(cfg_name, []).append((path, _parse_override_value(value)))
    return overrides


def _split_override_env(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            items = json.loads(raw)
            return [str(x) for x in items]
        except Exception:
            return [raw]
    return [s for s in (x.strip() for x in raw.split(";")) if s]


def load_overrides_from_env(env_key: str = "EEG_CFG_OVERRIDES") -> dict[str, list[tuple[str, Any]]]:
    raw = os.environ.get(env_key, "")
    if not raw:
        return {}
    return parse_overrides(_split_override_env(raw))


def apply_overrides(cfg: dict, items: list[tuple[str, Any]]) -> dict:
    for path, value in items:
        keys = [k for k in str(path).split(".") if k]
        if not keys:
            continue
        node = cfg
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return cfg


def apply_env_overrides(cfg: dict, cfg_name: str, env_key: str = "EEG_CFG_OVERRIDES") -> dict:
    overrides = load_overrides_from_env(env_key)
    if cfg_name in overrides:
        apply_overrides(cfg, overrides[cfg_name])
    return cfg
