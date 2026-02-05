#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from _script_utils import project_root

ROOT = project_root(__file__)

from src.dataio import load_processed_index
from src.models_gen import normalize_generator_type
from src.train_gen import train_generative_model
from src.utils import find_split_files, in_allowed_grid, load_json, load_yaml, make_exp_id, set_seed, stable_hash_seed


def _build_gen_cfg(
    base_cfg: dict,
    gen_model: str,
    sweep_cfg: dict,
    override_batch_size: int | None = None,
) -> dict:
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

    if override_batch_size is not None and override_batch_size > 0:
        cfg["train"]["batch_size"] = int(override_batch_size)

    return cfg


def _stable_gen_seed(
    base_seed: int,
    split_stem: str,
    subject: int | str,
    p: float,
    gen_model: str,
) -> int:
    return stable_hash_seed(
        base_seed=base_seed,
        payload={
            "split": str(split_stem),
            "subject": str(subject),
            "p": float(p),
            "gen_model": str(gen_model),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train generative models for EEG augmentation.")
    parser.add_argument(
        "--batch-size",
        "--gen-batch",
        "--gen_batch",
        dest="batch_size",
        type=int,
        default=None,
        help="Override generator training batch size for all gen models.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    gen_cfg = load_yaml(ROOT / "configs/gen.yaml")
    clf_cfg = load_yaml(ROOT / "configs/clf.yaml")
    qc_cfg = load_yaml(ROOT / "configs/qc.yaml")
    sweep_cfg = load_yaml(ROOT / "configs/sweep.yaml")
    split_cfg = load_yaml(ROOT / "configs/split.yaml")
    pp_cfg = load_yaml(ROOT / "configs/preprocess.yaml")

    default_model = normalize_generator_type(str(gen_cfg["model"].get("type", "cvae")))
    gen_models = [normalize_generator_type(m) for m in sweep_cfg.get("gen_models", [default_model])]

    index_df = load_processed_index(data_cfg["index_path"])

    # New protocol: subject/seed/p split files.
    split_files = find_split_files(ROOT, split_cfg)
    if not split_files:
        raise RuntimeError("No split files found. Run scripts/01_make_splits.py first.")

    for gen_model in gen_models:
        run_cfg = _build_gen_cfg(
            gen_cfg,
            gen_model=gen_model,
            sweep_cfg=sweep_cfg,
            override_batch_size=args.batch_size,
        )

        for sf in split_files:
            split = load_json(sf)
            if not in_allowed_grid(split, split_cfg=split_cfg, data_cfg=data_cfg):
                continue
            seed_base = int(split["seed"])

            subject = split.get("subject", "all")
            p = split.get("low_data_frac", 1.0)
            seed_run = _stable_gen_seed(
                base_seed=seed_base,
                split_stem=sf.stem,
                subject=subject,
                p=float(p),
                gen_model=gen_model,
            )
            set_seed(seed_run)
            exp_id = make_exp_id(
                "gen",
                protocol=split.get("protocol", split_cfg["protocol"]),
                subject=subject,
                seed=seed_base,
                p=p,
                split=sf.stem,
                gmodel=gen_model,
            )
            out_dir = ROOT / "runs/gen" / exp_id
            run_cfg.setdefault("data", {})["sfreq"] = int(data_cfg.get("sfreq", 250))
            train_generative_model(
                split=split,
                index_df=index_df,
                gen_cfg=run_cfg,
                preprocess_cfg=pp_cfg,
                out_dir=out_dir,
                qc_cfg=qc_cfg,
                clf_cfg=clf_cfg,
                base_seed=seed_run,
            )
            print(f"[done] {exp_id} bs={run_cfg['train']['batch_size']} run_seed={seed_run}")


if __name__ == "__main__":
    main()
