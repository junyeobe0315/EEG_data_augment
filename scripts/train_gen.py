#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._script_utils import project_root
except ImportError:  # pragma: no cover - direct script execution
    from _script_utils import project_root

ROOT = project_root(__file__)

from src.config_utils import build_gen_cfg
from src.dataio import load_processed_index
from src.models_gen import normalize_generator_type
from src.train_gen import train_generative_model
from src.utils import in_allowed_grid, load_json, load_yaml, make_exp_id, require_split_files, set_seed, stable_hash_seed


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
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override generator training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override generator base LR (also applies to model-specific LRs if not set).",
    )
    parser.add_argument(
        "--cvae-lr",
        type=float,
        default=None,
        help="Override CVAE LR.",
    )
    parser.add_argument(
        "--gan-lr",
        type=float,
        default=None,
        help="Override GAN LR (eeggan/cwgan).",
    )
    parser.add_argument(
        "--ddpm-lr",
        type=float,
        default=None,
        help="Override DDPM LR.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd"],
        default=None,
        help="Override optimizer type.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override optimizer weight decay.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader worker count.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if output checkpoint already exists.",
    )
    return parser.parse_args()


def _apply_train_overrides(run_cfg: dict, args: argparse.Namespace) -> None:
    tcfg = run_cfg.setdefault("train", {})

    if args.batch_size is not None:
        tcfg["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        tcfg["epochs"] = int(args.epochs)
    if args.lr is not None:
        lr = float(args.lr)
        tcfg["lr"] = lr
        if args.cvae_lr is None:
            tcfg["cvae_lr"] = lr
        if args.gan_lr is None:
            tcfg["gan_lr"] = lr
        if args.ddpm_lr is None:
            tcfg["ddpm_lr"] = lr
    if args.cvae_lr is not None:
        tcfg["cvae_lr"] = float(args.cvae_lr)
    if args.gan_lr is not None:
        tcfg["gan_lr"] = float(args.gan_lr)
    if args.ddpm_lr is not None:
        tcfg["ddpm_lr"] = float(args.ddpm_lr)
    if args.optimizer is not None:
        tcfg["optimizer"] = str(args.optimizer)
    if args.weight_decay is not None:
        tcfg["weight_decay"] = float(args.weight_decay)
    if args.num_workers is not None:
        tcfg["num_workers"] = int(args.num_workers)
    if args.device is not None:
        tcfg["device"] = str(args.device)


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
    split_files = require_split_files(ROOT, split_cfg)

    for gen_model in gen_models:
        run_cfg = build_gen_cfg(
            gen_cfg,
            gen_model=gen_model,
            sweep_cfg=sweep_cfg,
            override_batch_size=args.batch_size,
        )
        _apply_train_overrides(run_cfg, args)

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
                gen=gen_model,
            )
            out_dir = ROOT / "runs/gen" / exp_id
            if not args.force and (out_dir / "ckpt.pt").exists():
                print(f"[skip] {exp_id} (ckpt.pt exists)")
                continue
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
