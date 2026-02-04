#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dataio import load_processed_index
from src.models_gen import normalize_generator_type
from src.train_gen import train_generative_model
from src.utils import load_yaml, make_exp_id, set_seed


def _load_split(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_gen_cfg(base_cfg: dict, gen_model: str, sweep_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["type"] = gen_model

    if not bool(sweep_cfg.get("apply_vram_presets", True)):
        return cfg

    profile = str(sweep_cfg.get("vram_profile", "6gb"))
    preset = cfg.get("vram_presets", {}).get(profile, {}).get(gen_model, {})

    if "batch_size" in preset:
        cfg["train"]["batch_size"] = int(preset["batch_size"])
    if "n_per_class" in preset:
        cfg["sample"]["n_per_class"] = int(preset["n_per_class"])
    if "ddpm_steps" in preset:
        cfg["sample"]["ddpm_steps"] = int(preset["ddpm_steps"])
    if "diffusion_steps" in preset:
        cfg.setdefault("model", {}).setdefault("conditional_ddpm", {})["diffusion_steps"] = int(preset["diffusion_steps"])

    return cfg


def _in_allowed_grid(split: dict, split_cfg: dict, data_cfg: dict) -> bool:
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


def main() -> None:
    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    gen_cfg = load_yaml(ROOT / "configs/gen.yaml")
    sweep_cfg = load_yaml(ROOT / "configs/sweep.yaml")
    split_cfg = load_yaml(ROOT / "configs/split.yaml")

    default_model = normalize_generator_type(str(gen_cfg["model"].get("type", "cvae")))
    gen_models = [normalize_generator_type(m) for m in sweep_cfg.get("gen_models", [default_model])]

    index_df = load_processed_index(data_cfg["index_path"])

    # New protocol: subject/seed/p split files.
    split_files = sorted((ROOT / "data/splits").glob("subject_*_seed_*_p_*.json"))
    if not split_files:
        # Backward compatibility
        split_files = sorted((ROOT / "data/splits").glob(f"split_{split_cfg['protocol']}_seed*.json"))
    if not split_files:
        raise RuntimeError("No split files found. Run scripts/01_make_splits.py first.")

    for gen_model in gen_models:
        run_cfg = _build_gen_cfg(gen_cfg, gen_model=gen_model, sweep_cfg=sweep_cfg)

        for sf in split_files:
            split = _load_split(sf)
            if not _in_allowed_grid(split, split_cfg=split_cfg, data_cfg=data_cfg):
                continue
            seed = int(split["seed"])
            set_seed(seed)

            subject = split.get("subject", "all")
            p = split.get("low_data_frac", 1.0)
            exp_id = make_exp_id(
                "gen",
                protocol=split.get("protocol", split_cfg["protocol"]),
                subject=subject,
                seed=seed,
                p=p,
                split=sf.stem,
                gmodel=gen_model,
            )
            out_dir = ROOT / "runs/gen" / exp_id
            train_generative_model(split, index_df, run_cfg, load_yaml(ROOT / "configs/preprocess.yaml"), out_dir)
            print(f"[done] {exp_id} bs={run_cfg['train']['batch_size']}")


if __name__ == "__main__":
    main()
