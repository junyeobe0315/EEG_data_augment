#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from _script_utils import project_root

import numpy as np
import pandas as pd

ROOT = project_root(__file__)

from src.dataio import load_processed_index, load_samples_by_ids
from src.models_gen import normalize_generator_type
from src.qc import run_qc
from src.sample_gen import sample_by_class, save_synth_npz
from src.utils import ensure_dir, in_allowed_grid, load_json, load_yaml, make_exp_id, require_split_files, save_json, set_seed, stable_hash_seed


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stable_sampling_seed(
    base_seed: int,
    split_stem: str,
    subject: int | str,
    p: float,
    gen_model: str,
    stage: str,
) -> int:
    return stable_hash_seed(
        base_seed=base_seed,
        payload={
            "split": str(split_stem),
            "subject": str(subject),
            "p": float(p),
            "gen_model": str(gen_model),
            "stage": str(stage),
        },
    )


def _active_ratio_list(sweep_cfg: dict) -> list[float]:
    ratios = [float(x) for x in sweep_cfg.get("ratio_list", [0.0])]
    stage = str(sweep_cfg.get("stage_mode", "full"))
    if stage == "full":
        full_ratios = sweep_cfg.get("full_stage_ratios")
        if full_ratios is not None:
            ratios = [float(x) for x in full_ratios]
    return ratios


def _resolve_n_per_class(
    run_cfg: dict,
    y_real_train: np.ndarray,
    num_classes: int,
    sweep_cfg: dict,
    qc_cfg: dict,
) -> tuple[int, dict]:
    base_n = int(run_cfg["sample"].get("n_per_class", 300))
    dynamic = bool(run_cfg["sample"].get("dynamic_n_per_class", True))
    buffer = float(run_cfg["sample"].get("dynamic_buffer", 1.2))
    max_ratio = max([r for r in _active_ratio_list(sweep_cfg) if r > 0.0], default=0.0)

    expected_keep = float(qc_cfg.get("target_keep_ratio", 1.0))
    expected_keep = min(max(expected_keep, 0.05), 1.0)
    counts = np.bincount(y_real_train.astype(np.int64), minlength=int(num_classes))
    max_real_class = int(counts.max()) if len(counts) > 0 else 0

    resolved = int(base_n)
    if dynamic and max_ratio > 0 and max_real_class > 0:
        required = int(np.ceil((max_ratio * max_real_class / expected_keep) * max(buffer, 1.0)))
        resolved = max(base_n, required)

    meta = {
        "sample_n_per_class_base": int(base_n),
        "sample_n_per_class_effective": int(resolved),
        "sample_n_per_class_dynamic": bool(dynamic),
        "sample_n_per_class_dynamic_buffer": float(buffer),
        "max_ratio_for_sampling": float(max_ratio),
        "expected_qc_keep_ratio": float(expected_keep),
        "max_real_class_count": int(max_real_class),
    }
    return resolved, meta


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

    split_files = require_split_files(ROOT, split_cfg)

    reports = []
    for gen_model in gen_models:
        run_cfg = _build_gen_cfg(gen_cfg, gen_model=gen_model, sweep_cfg=sweep_cfg)

        for sf in split_files:
            split = load_json(sf)
            if not in_allowed_grid(split, split_cfg=split_cfg, data_cfg=data_cfg):
                continue
            seed = int(split["seed"])
            subject = split.get("subject", "all")
            try:
                subject_i = int(subject)
            except Exception:
                subject_i = -1
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

            x_real_train, y_real_train = load_samples_by_ids(index_df, split["train_ids"])
            n_per_class, sample_meta = _resolve_n_per_class(
                run_cfg=run_cfg,
                y_real_train=y_real_train,
                num_classes=len(data_cfg["class_names"]),
                sweep_cfg=sweep_cfg,
                qc_cfg=qc_cfg,
            )

            synth_seed = _stable_sampling_seed(
                base_seed=seed,
                split_stem=sf.stem,
                subject=subject,
                p=float(p),
                gen_model=gen_model,
                stage="synth",
            )
            qc_seed = _stable_sampling_seed(
                base_seed=seed,
                split_stem=sf.stem,
                subject=subject,
                p=float(p),
                gen_model=gen_model,
                stage="qc",
            )
            set_seed(synth_seed)
            synth = sample_by_class(
                ckpt_path=ckpt_path,
                n_per_class=int(n_per_class),
                num_classes=len(data_cfg["class_names"]),
                device=run_cfg["train"].get("device", "cpu"),
            )
            synth_path = synth_dir / f"synth_{gen_model}_{sf.stem}.npz"
            save_synth_npz(synth_path, synth)

            set_seed(qc_seed)
            kept, report = run_qc(
                real_x=x_real_train,
                synth=synth,
                sfreq=int(data_cfg["sfreq"]),
                cfg=qc_cfg,
                real_y=y_real_train,
            )
            kept_path = qc_dir / f"synth_qc_{gen_model}_{sf.stem}.npz"
            save_synth_npz(kept_path, kept)

            synth_meta = {
                "split": sf.stem,
                "subject": subject_i,
                "seed": int(seed),
                "low_data_frac": float(p),
                "generator": gen_model,
                "generator_ckpt_path": str(ckpt_path),
                "generator_ckpt_sha256": _sha256_file(ckpt_path),
                "synth_seed": synth_seed,
                "qc_seed": qc_seed,
                "sample_n_per_class": int(n_per_class),
                "ddpm_steps": int(run_cfg["sample"].get("ddpm_steps", 0)),
                "n_synth_before_qc": int(report.get("n_before", synth["X"].shape[0])),
                "n_synth_after_qc": int(report.get("n_after", kept["X"].shape[0])),
                "qc_keep_ratio": float(report.get("keep_ratio", 0.0)),
                **sample_meta,
            }
            save_json(synth_path.with_suffix(".meta.json"), synth_meta)
            save_json(kept_path.with_suffix(".meta.json"), {**synth_meta, "qc_enabled": True})
            save_json(kept_path.with_suffix(".report.json"), report)

            reports.append({"split": sf.stem, "subject": subject, "seed": seed, "p": p, "gen_model": gen_model, **report})

    if reports:
        out_csv = metric_dir / f"qc_{split_cfg['protocol']}.csv"
        pd.DataFrame(reports).to_csv(out_csv, index=False)
        print(f"Saved QC report -> {out_csv}")


if __name__ == "__main__":
    main()
