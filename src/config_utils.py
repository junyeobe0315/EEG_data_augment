from __future__ import annotations

import copy


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
