#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dataio import load_processed_index, load_samples_by_ids
from src.models_gen import normalize_generator_type
from src.qc import run_qc
from src.sample_gen import sample_by_class, save_synth_npz
from src.utils import ensure_dir, load_yaml, make_exp_id


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

    if "n_per_class" in preset:
        cfg["sample"]["n_per_class"] = int(preset["n_per_class"])
    if "ddpm_steps" in preset:
        cfg["sample"]["ddpm_steps"] = int(preset["ddpm_steps"])

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
    qc_cfg = load_yaml(ROOT / "configs/qc.yaml")

    default_model = normalize_generator_type(str(gen_cfg["model"].get("type", "cvae")))
    gen_models = [normalize_generator_type(m) for m in sweep_cfg.get("gen_models", [default_model])]

    index_df = load_processed_index(data_cfg["index_path"])
    synth_dir = ensure_dir(ROOT / "runs/synth")
    qc_dir = ensure_dir(ROOT / "runs/synth_qc")
    metric_dir = ensure_dir(ROOT / "results/metrics")

    split_files = sorted((ROOT / "data/splits").glob("subject_*_seed_*_p_*.json"))
    if not split_files:
        split_files = sorted((ROOT / "data/splits").glob(f"split_{split_cfg['protocol']}_seed*.json"))

    reports = []
    for gen_model in gen_models:
        run_cfg = _build_gen_cfg(gen_cfg, gen_model=gen_model, sweep_cfg=sweep_cfg)

        for sf in split_files:
            split = _load_split(sf)
            if not _in_allowed_grid(split, split_cfg=split_cfg, data_cfg=data_cfg):
                continue
            seed = int(split["seed"])
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
            ckpt_path = ROOT / "runs/gen" / exp_id / "ckpt.pt"
            if not ckpt_path.exists():
                print(f"Skip split {sf.name}, gen={gen_model}: checkpoint not found")
                continue

            synth = sample_by_class(
                ckpt_path=ckpt_path,
                n_per_class=int(run_cfg["sample"].get("n_per_class", 300)),
                num_classes=len(data_cfg["class_names"]),
                device=run_cfg["train"].get("device", "cpu"),
            )
            synth_path = synth_dir / f"synth_{gen_model}_{sf.stem}.npz"
            save_synth_npz(synth_path, synth)

            x_real_train, y_real_train = load_samples_by_ids(index_df, split["train_ids"])
            kept, report = run_qc(
                real_x=x_real_train,
                synth=synth,
                sfreq=int(data_cfg["sfreq"]),
                cfg=qc_cfg,
                real_y=y_real_train,
            )
            kept_path = qc_dir / f"synth_qc_{gen_model}_{sf.stem}.npz"
            save_synth_npz(kept_path, kept)

            reports.append({"split": sf.stem, "subject": subject, "seed": seed, "p": p, "gen_model": gen_model, **report})

    if reports:
        out_csv = metric_dir / f"qc_{split_cfg['protocol']}.csv"
        pd.DataFrame(reports).to_csv(out_csv, index=False)
        print(f"Saved QC report -> {out_csv}")


if __name__ == "__main__":
    main()
