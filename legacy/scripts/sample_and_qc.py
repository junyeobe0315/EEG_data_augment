#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    from scripts._script_utils import project_root
except ImportError:  # pragma: no cover - direct script execution
    from _script_utils import project_root

import numpy as np
import pandas as pd

ROOT = project_root(__file__)

from src.config_utils import build_gen_cfg
from src.dataio import load_processed_index, load_samples_by_ids
from src.models_gen import normalize_generator_type
from src.qc import run_qc
from src.sample_gen import sample_by_class, save_synth_npz
from src.parallel import run_subprocess_tasks
from src.utils import ensure_dir, in_allowed_grid, load_json, load_yaml, make_exp_id, require_split_files, save_json, set_seed, stable_hash_seed


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample generators and run QC (with caching).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-sample and re-run QC even if outputs already exist.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (uses subprocess scheduling).",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated CUDA device list for parallel jobs.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values (e.g., qc.psd.z_threshold=2.0).",
    )
    return parser.parse_args()


def _parse_devices(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [d.strip() for d in str(raw).split(",") if d.strip()]


def _parse_task(raw: str) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid --task JSON: {raw}")


def _build_task_cmd(args: argparse.Namespace, task: dict) -> list[str]:
    cmd = [sys.executable, str(ROOT / "main.py"), "sample-qc", "--task", json.dumps(task)]
    if args.force:
        cmd.append("--force")
    for item in args.set or []:
        cmd += ["--set", item]
    return cmd


def _build_tasks(
    gen_models: list[str],
    split_files: list[Path],
    split_cfg: dict,
    data_cfg: dict,
    force: bool,
    qc_dir: Path,
) -> list[dict]:
    tasks: list[dict] = []
    for gen_model in gen_models:
        for sf in split_files:
            split = load_json(sf)
            if not in_allowed_grid(split, split_cfg=split_cfg, data_cfg=data_cfg):
                continue
            kept_path = qc_dir / f"synth_qc_{gen_model}_{sf.stem}.npz"
            report_path = kept_path.with_suffix(".report.json")
            if (not force) and kept_path.exists() and report_path.exists():
                continue
            tasks.append({"gen_model": gen_model, "split_file": str(sf)})
    return tasks


def _collect_reports(
    gen_models: list[str],
    split_files: list[Path],
    split_cfg: dict,
    data_cfg: dict,
    qc_dir: Path,
    metric_dir: Path,
) -> None:
    reports = []
    for gen_model in gen_models:
        for sf in split_files:
            split = load_json(sf)
            if not in_allowed_grid(split, split_cfg=split_cfg, data_cfg=data_cfg):
                continue
            report_path = qc_dir / f"synth_qc_{gen_model}_{sf.stem}.report.json"
            if not report_path.exists():
                continue
            report = load_json(report_path)
            subject = split.get("subject", "all")
            seed = int(split.get("seed", -1))
            p = float(split.get("low_data_frac", 1.0))
            reports.append({"split": sf.stem, "subject": subject, "seed": seed, "p": p, "gen_model": gen_model, **report})

    if reports:
        out_csv = metric_dir / f"qc_{split_cfg['protocol']}.csv"
        pd.DataFrame(reports).to_csv(out_csv, index=False)
        print(f"Saved QC report -> {out_csv}")


def main() -> None:
    args = _parse_args()
    if args.set:
        os.environ["EEG_CFG_OVERRIDES"] = json.dumps(list(args.set))
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

    task = _parse_task(args.task)
    if task is not None:
        gen_models = [normalize_generator_type(str(task["gen_model"]))]
        split_files = [Path(task["split_file"])]
        args.jobs = 1

    if int(args.jobs) > 1 and task is None:
        devices = _parse_devices(args.devices)
        tasks = _build_tasks(
            gen_models,
            split_files,
            split_cfg=split_cfg,
            data_cfg=data_cfg,
            force=args.force,
            qc_dir=qc_dir,
        )
        if not tasks:
            print("[info] No tasks to run.")
            return
        scheduled = []
        for i, t in enumerate(tasks):
            env = {}
            if devices:
                env["CUDA_VISIBLE_DEVICES"] = devices[i % len(devices)]
            label = f"qc={t['gen_model']} split={Path(t['split_file']).stem}"
            scheduled.append({"cmd": _build_task_cmd(args, t), "env": env, "label": label})
        run_subprocess_tasks(scheduled, max_workers=int(args.jobs), label="sample-qc")
        _collect_reports(gen_models, split_files, split_cfg, data_cfg, qc_dir=qc_dir, metric_dir=metric_dir)
        return

    reports = []
    for gen_model in gen_models:
        run_cfg = build_gen_cfg(gen_cfg, gen_model=gen_model, sweep_cfg=sweep_cfg, apply_train=False)

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

            synth_path = synth_dir / f"synth_{gen_model}_{sf.stem}.npz"
            kept_path = qc_dir / f"synth_qc_{gen_model}_{sf.stem}.npz"
            report_path = kept_path.with_suffix(".report.json")

            if (not args.force) and kept_path.exists() and report_path.exists():
                report = load_json(report_path)
                reports.append({"split": sf.stem, "subject": subject, "seed": seed, "p": p, "gen_model": gen_model, **report})
                print(f"[skip] {sf.stem} gen={gen_model} (qc outputs exist)")
                continue

            exp_id = make_exp_id(
                "gen",
                protocol=split.get("protocol", split_cfg["protocol"]),
                subject=subject,
                seed=seed,
                p=p,
                split=sf.stem,
                gen=gen_model,
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
            if (not args.force) and synth_path.exists():
                arr = np.load(synth_path)
                synth = {k: arr[k] for k in arr.files}
                print(f"[skip] {sf.stem} gen={gen_model} (synth exists)")
            else:
                set_seed(synth_seed)
                synth = sample_by_class(
                    ckpt_path=ckpt_path,
                    n_per_class=int(n_per_class),
                    num_classes=len(data_cfg["class_names"]),
                    device=run_cfg["train"].get("device", "cpu"),
                )
                save_synth_npz(synth_path, synth)

            set_seed(qc_seed)
            kept, report = run_qc(
                real_x=x_real_train,
                synth=synth,
                sfreq=int(data_cfg["sfreq"]),
                cfg=qc_cfg,
                real_y=y_real_train,
            )
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
    from src.cli_deprecated import exit_deprecated
    exit_deprecated("sample-qc")
