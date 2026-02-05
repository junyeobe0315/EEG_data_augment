#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

try:
    from scripts._script_utils import project_root
except ImportError:  # pragma: no cover - direct script execution
    from _script_utils import project_root

import pandas as pd

ROOT = project_root(__file__)

from src.dataio import load_processed_index
from src.models_clf import normalize_classifier_type
from src.models_gen import normalize_generator_type
from src.parallel import run_subprocess_tasks
from src.train_clf import train_classifier
from src.utils import ensure_dir, in_allowed_grid, load_json, load_yaml, make_exp_id, require_split_files, set_seed, stable_hash_seed


def _build_clf_cfg(
    base_cfg: dict,
    clf_model: str,
    sweep_cfg: dict,
    override_batch_size: int | None = None,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["type"] = clf_model

    use_paper = bool(sweep_cfg.get("use_paper_presets", cfg.get("paper_presets", {}).get("enabled", False)))
    if use_paper:
        pp = cfg.get("paper_presets", {}).get(clf_model, {})
        for key in ("epochs", "batch_size", "lr", "weight_decay", "num_workers", "device"):
            if key in pp:
                cfg["train"][key] = pp[key]
        if override_batch_size is not None and override_batch_size > 0:
            cfg["train"]["batch_size"] = int(override_batch_size)
        return cfg

    if not bool(sweep_cfg.get("apply_vram_presets", True)):
        return cfg

    profile = str(sweep_cfg.get("vram_profile", "6gb"))
    preset = cfg.get("vram_presets", {}).get(profile, {}).get(clf_model, {})
    if "batch_size" in preset and int(preset["batch_size"]) > 0:
        cfg["train"]["batch_size"] = int(preset["batch_size"])

    if override_batch_size is not None and override_batch_size > 0:
        cfg["train"]["batch_size"] = int(override_batch_size)

    return cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifiers with augmentation modes.")
    parser.add_argument(
        "--batch-size",
        "--clf-batch",
        "--clf_batch",
        dest="batch_size",
        type=int,
        default=None,
        help="Override classifier training batch size for all clf models.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override classifier training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override classifier learning rate.",
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
        help="Re-train even if output metrics already exist.",
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
        "--collect-only",
        action="store_true",
        help="Only collect metrics CSV from existing runs; do not train.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values (e.g., clf.train.lr=0.001).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable speed-oriented settings (AMP/TF32/pin_memory).",
    )
    return parser.parse_args()


def _apply_train_overrides(cfg: dict, args: argparse.Namespace) -> None:
    tcfg = cfg.setdefault("train", {})

    if args.batch_size is not None:
        tcfg["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        tcfg["epochs"] = int(args.epochs)
    if args.lr is not None:
        tcfg["lr"] = float(args.lr)
    if args.optimizer is not None:
        tcfg["optimizer"] = str(args.optimizer)
    if args.weight_decay is not None:
        tcfg["weight_decay"] = float(args.weight_decay)
    if args.num_workers is not None:
        tcfg["num_workers"] = int(args.num_workers)
    if args.device is not None:
        tcfg["device"] = str(args.device)


def _apply_fast_overrides(cfg: dict, enable: bool) -> None:
    if not enable:
        return
    tcfg = cfg.setdefault("train", {})
    amp_cfg = tcfg.setdefault("amp", {})
    amp_cfg["enabled"] = True
    amp_cfg.setdefault("dtype", "float16")
    dl_cfg = tcfg.setdefault("dataloader", {})
    dl_cfg["pin_memory"] = True
    dl_cfg["persistent_workers"] = True
    dl_cfg.setdefault("prefetch_factor", 2)
    cuda_cfg = tcfg.setdefault("cuda", {})
    cuda_cfg["tf32"] = True
    cuda_cfg.setdefault("matmul_precision", "high")


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
    task_ref = {k: task.get(k) for k in ("split_file", "clf_model", "mode", "ratio", "gen_model", "qc_on")}
    cmd = [sys.executable, str(ROOT / "main.py"), "train-clf", "--task", json.dumps(task_ref)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.optimizer is not None:
        cmd += ["--optimizer", str(args.optimizer)]
    if args.weight_decay is not None:
        cmd += ["--weight-decay", str(args.weight_decay)]
    if args.num_workers is not None:
        cmd += ["--num-workers", str(args.num_workers)]
    if args.device is not None:
        cmd += ["--device", str(args.device)]
    if args.force:
        cmd.append("--force")
    if args.fast:
        cmd.append("--fast")
    for item in args.set or []:
        cmd += ["--set", item]
    return cmd


def _maybe_load_metrics(out_dir: Path, force: bool) -> dict | None:
    metrics_path = out_dir / "metrics.json"
    if (not force) and metrics_path.exists():
        print(f"[skip] {out_dir.name} (metrics.json exists)")
        return load_json(metrics_path)
    return None


def _find_synth_npz(root: Path, gen_model: str, split_stem: str, qc_on: bool) -> Path:
    sub = "runs/synth_qc" if qc_on else "runs/synth"
    name = f"synth_qc_{gen_model}_{split_stem}.npz" if qc_on else f"synth_{gen_model}_{split_stem}.npz"
    p = root / sub / name
    if p.exists():
        return p

    legacy = f"synth_qc_{split_stem}.npz" if qc_on else f"synth_{split_stem}.npz"
    return root / sub / legacy


def _resolve_stage_lists(
    sweep_cfg: dict,
    clf_models: list[str],
    gen_models: list[str],
    ratio_list: list[float],
    qc_on: list[bool],
) -> tuple[list[str], list[str], list[float], list[bool]]:
    stage = str(sweep_cfg.get("stage_mode", "full"))

    if stage == "screening":
        clf_models = [normalize_classifier_type(str(sweep_cfg.get("screening_classifier", "eegnet")))]
        qc_on = [True]
        return clf_models, gen_models, ratio_list, qc_on

    if stage == "full":
        full_ratios = sweep_cfg.get("full_stage_ratios")
        if full_ratios is not None:
            ratio_list = [float(a) for a in full_ratios]
        return clf_models, gen_models, ratio_list, qc_on

    raise ValueError(f"Unknown stage_mode: {stage}")


def _condition_name(mode: str, gen_model: str | None = None) -> str:
    if mode == "none":
        return "C0_no_aug"
    if mode == "classical":
        return "C1_classical"
    if mode == "mixup":
        return "C2_hard_mix"
    if mode == "gen_aug":
        return f"GenAug_{gen_model}" if gen_model else "GenAug"
    return f"Other_{mode}"


def _stable_condition_seed(
    base_seed: int,
    split_stem: str,
    subject: int,
    p: float,
    clf_model: str,
    mode: str,
    ratio: float,
    gen_model: str,
    qc_on: bool,
) -> int:
    return stable_hash_seed(
        base_seed=base_seed,
        payload={
            "split": split_stem,
            "subject": int(subject),
            "p": float(p),
            "clf_model": str(clf_model),
            "mode": str(mode),
            "ratio": float(ratio),
            "gen_model": str(gen_model),
            "qc_on": bool(qc_on),
        },
    )


def _iter_tasks(
    split_files: list[Path],
    data_cfg: dict,
    split_cfg: dict,
    clf_cfg: dict,
    gen_cfg: dict,
    sweep_cfg: dict,
    args: argparse.Namespace,
    task_filter: dict | None = None,
    skip_existing: bool = False,
) -> list[dict]:
    default_clf = normalize_classifier_type(str(clf_cfg["model"].get("type", "eegnet")))
    default_gen = normalize_generator_type(str(gen_cfg["model"].get("type", "cvae")))

    clf_models = [normalize_classifier_type(m) for m in sweep_cfg.get("clf_models", [default_clf])]
    gen_models = [normalize_generator_type(m) for m in sweep_cfg.get("gen_models", [default_gen])]
    ratio_list = [float(a) for a in sweep_cfg.get("ratio_list", [0.0, 0.25, 0.5, 1.0, 2.0])]
    qc_on_list = [bool(v) for v in sweep_cfg.get("qc_on", [True])]
    clf_models, gen_models, ratio_list, qc_on_list = _resolve_stage_lists(
        sweep_cfg,
        clf_models,
        gen_models,
        ratio_list,
        qc_on_list,
    )

    tasks: list[dict] = []
    for sf in split_files:
        split = load_json(sf)
        if not in_allowed_grid(split, split_cfg=split_cfg, data_cfg=data_cfg):
            continue
        seed = int(split["seed"])
        subject = int(split.get("subject", -1))
        p = float(split.get("low_data_frac", 1.0))

        for clf_model in clf_models:
            clf_run_cfg = _build_clf_cfg(
                clf_cfg,
                clf_model=clf_model,
                sweep_cfg=sweep_cfg,
                override_batch_size=args.batch_size,
            )
            _apply_train_overrides(clf_run_cfg, args)
            _apply_fast_overrides(clf_run_cfg, args.fast)
            modes = [str(m) for m in clf_run_cfg.get("augmentation", {}).get("modes", ["none"])]
            evaluate_test = bool(clf_run_cfg.get("evaluation", {}).get("evaluate_test", False))

            for mode in modes:
                if mode == "none":
                    ratio = 0.0
                    run_seed = _stable_condition_seed(
                        base_seed=seed,
                        split_stem=sf.stem,
                        subject=subject,
                        p=p,
                        clf_model=clf_model,
                        mode=mode,
                        ratio=ratio,
                        gen_model="none",
                        qc_on=False,
                    )
                    exp_id = make_exp_id(
                        "clf",
                        split=sf.stem,
                        subject=subject,
                        p=p,
                        clf=clf_model,
                        mode=mode,
                        ratio=ratio,
                        qc=False,
                    )
                    out_dir = ROOT / "runs/clf" / exp_id
                    if skip_existing and (out_dir / "metrics.json").exists():
                        continue
                    task = {
                        "split_file": str(sf),
                        "split": split,
                        "subject": subject,
                        "p": p,
                        "seed": seed,
                        "protocol": split.get("protocol", split_cfg["protocol"]),
                        "clf_model": clf_model,
                        "gen_model": "none",
                        "condition": _condition_name(mode=mode),
                        "mode": mode,
                        "run_seed": run_seed,
                        "ratio": ratio,
                        "alpha_tilde": float(ratio / (1.0 + ratio)),
                        "qc_on": False,
                        "run_dir": str(out_dir),
                        "aug_npz": str(out_dir / "aug_used.npz"),
                        "synth_npz": "",
                        "evaluate_test": evaluate_test,
                        "clf_cfg": clf_run_cfg,
                    }
                    if task_filter and not _task_matches(task, task_filter):
                        continue
                    tasks.append(task)
                    continue

                if mode in {"classical", "mixup", "paper_sr"}:
                    for ratio in ratio_list:
                        ratio = float(ratio)
                        if ratio <= 0:
                            continue
                        run_seed = _stable_condition_seed(
                            base_seed=seed,
                            split_stem=sf.stem,
                            subject=subject,
                            p=p,
                            clf_model=clf_model,
                            mode=mode,
                            ratio=ratio,
                            gen_model="none",
                            qc_on=False,
                        )
                        exp_id = make_exp_id(
                            "clf",
                            split=sf.stem,
                            subject=subject,
                            p=p,
                            clf=clf_model,
                            mode=mode,
                            ratio=ratio,
                            qc=False,
                        )
                        out_dir = ROOT / "runs/clf" / exp_id
                        if skip_existing and (out_dir / "metrics.json").exists():
                            continue
                        task = {
                            "split_file": str(sf),
                            "split": split,
                            "subject": subject,
                            "p": p,
                            "seed": seed,
                            "protocol": split.get("protocol", split_cfg["protocol"]),
                            "clf_model": clf_model,
                            "gen_model": "none",
                            "condition": _condition_name(mode=mode),
                            "mode": mode,
                            "run_seed": run_seed,
                            "ratio": ratio,
                            "alpha_tilde": float(ratio / (1.0 + ratio)),
                            "qc_on": False,
                            "run_dir": str(out_dir),
                            "aug_npz": str(out_dir / "aug_used.npz"),
                            "synth_npz": "",
                            "evaluate_test": evaluate_test,
                            "clf_cfg": clf_run_cfg,
                        }
                        if task_filter and not _task_matches(task, task_filter):
                            continue
                        tasks.append(task)
                    continue

                if mode == "gen_aug":
                    for ratio in ratio_list:
                        ratio = float(ratio)
                        if ratio <= 0:
                            continue
                        for gen_model in gen_models:
                            for qc_on in qc_on_list:
                                run_seed = _stable_condition_seed(
                                    base_seed=seed,
                                    split_stem=sf.stem,
                                    subject=subject,
                                    p=p,
                                    clf_model=clf_model,
                                    mode=mode,
                                    ratio=ratio,
                                    gen_model=gen_model,
                                    qc_on=bool(qc_on),
                                )
                                synth_npz = _find_synth_npz(
                                    ROOT,
                                    gen_model=gen_model,
                                    split_stem=sf.stem,
                                    qc_on=bool(qc_on),
                                )
                                if not synth_npz.exists():
                                    print(
                                        f"Skip {sf.stem} clf={clf_model} gen={gen_model} ratio={ratio} qc={qc_on}: {synth_npz} missing"
                                    )
                                    continue
                                exp_id = make_exp_id(
                                    "clf",
                                    split=sf.stem,
                                    subject=subject,
                                    p=p,
                                    clf=clf_model,
                                    gen=gen_model,
                                    mode=mode,
                                    ratio=ratio,
                                    qc=bool(qc_on),
                                )
                                out_dir = ROOT / "runs/clf" / exp_id
                                if skip_existing and (out_dir / "metrics.json").exists():
                                    continue
                                task = {
                                    "split_file": str(sf),
                                    "split": split,
                                    "subject": subject,
                                    "p": p,
                                    "seed": seed,
                                    "protocol": split.get("protocol", split_cfg["protocol"]),
                                    "clf_model": clf_model,
                                    "gen_model": gen_model,
                                    "condition": _condition_name(mode=mode, gen_model=gen_model),
                                    "mode": mode,
                                    "run_seed": run_seed,
                                    "ratio": ratio,
                                    "alpha_tilde": float(ratio / (1.0 + ratio)),
                                    "qc_on": bool(qc_on),
                                    "run_dir": str(out_dir),
                                    "aug_npz": str(out_dir / "aug_used.npz"),
                                    "synth_npz": str(synth_npz),
                                    "evaluate_test": evaluate_test,
                                    "clf_cfg": clf_run_cfg,
                                }
                                if task_filter and not _task_matches(task, task_filter):
                                    continue
                                tasks.append(task)
                    continue

                print(f"[warn] unsupported augmentation mode: {mode}")

    return tasks


def _task_matches(task: dict, filt: dict) -> bool:
    for key in ("split_file", "clf_model", "mode", "ratio", "gen_model", "qc_on"):
        if key not in filt:
            continue
        if key == "ratio":
            if abs(float(task.get(key, 0.0)) - float(filt.get(key))) > 1e-12:
                return False
            continue
        if key == "qc_on":
            if bool(task.get(key)) != bool(filt.get(key)):
                return False
            continue
        if str(task.get(key)) != str(filt.get(key)):
            return False
    return True


def _run_task(task: dict, index_df: pd.DataFrame, pp_cfg: dict, force: bool) -> None:
    out_dir = Path(task["run_dir"])
    metrics_path = out_dir / "metrics.json"
    if (not force) and metrics_path.exists():
        print(f"[skip] {out_dir.name} (metrics.json exists)")
        return

    set_seed(int(task["run_seed"]))
    train_classifier(
        task["split"],
        index_df,
        task["clf_cfg"],
        pp_cfg,
        out_dir,
        mode=task["mode"],
        ratio=float(task["ratio"]),
        synth_npz=str(task["synth_npz"]) if task.get("synth_npz") else None,
        evaluate_test=bool(task.get("evaluate_test", False)),
    )


def _collect_metrics(tasks: list[dict], metric_dir: Path, split_cfg: dict) -> None:
    rows = []
    for task in tasks:
        metrics_path = Path(task["run_dir"]) / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.json for {task['run_dir']}")
        m = load_json(metrics_path)
        rows.append(
            {
                "split": Path(task["split_file"]).stem,
                "split_file": str(task["split_file"]),
                "subject": task["subject"],
                "p": task["p"],
                "seed": task["seed"],
                "protocol": task["protocol"],
                "clf_model": task["clf_model"],
                "gen_model": task["gen_model"],
                "condition": task["condition"],
                "mode": task["mode"],
                "run_seed": task["run_seed"],
                "ratio": task["ratio"],
                "alpha_tilde": task["alpha_tilde"],
                "qc_on": task["qc_on"],
                "run_dir": task["run_dir"],
                "aug_npz": task["aug_npz"],
                "synth_npz": task["synth_npz"],
                **m,
            }
        )

    if rows:
        out_csv = metric_dir / f"clf_{split_cfg['protocol']}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved clf metrics -> {out_csv}")


def main() -> None:
    args = _parse_args()
    if args.set:
        os.environ["EEG_CFG_OVERRIDES"] = json.dumps(list(args.set))
    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    split_cfg = load_yaml(ROOT / "configs/split.yaml")
    pp_cfg = load_yaml(ROOT / "configs/preprocess.yaml")
    clf_cfg = load_yaml(ROOT / "configs/clf.yaml")
    gen_cfg = load_yaml(ROOT / "configs/gen.yaml")
    sweep_cfg = load_yaml(ROOT / "configs/sweep.yaml")

    index_df = load_processed_index(data_cfg["index_path"])
    metric_dir = ensure_dir(ROOT / "results/metrics")

    split_files = require_split_files(ROOT, split_cfg)
    task_filter = _parse_task(args.task)
    if task_filter is not None:
        args.jobs = 1

    if args.collect_only:
        tasks_all = _iter_tasks(
            split_files=split_files,
            data_cfg=data_cfg,
            split_cfg=split_cfg,
            clf_cfg=clf_cfg,
            gen_cfg=gen_cfg,
            sweep_cfg=sweep_cfg,
            args=args,
            task_filter=task_filter,
            skip_existing=False,
        )
        _collect_metrics(tasks_all, metric_dir=metric_dir, split_cfg=split_cfg)
        return

    if int(args.jobs) > 1 and task_filter is None:
        devices = _parse_devices(args.devices)
        tasks_to_run = _iter_tasks(
            split_files=split_files,
            data_cfg=data_cfg,
            split_cfg=split_cfg,
            clf_cfg=clf_cfg,
            gen_cfg=gen_cfg,
            sweep_cfg=sweep_cfg,
            args=args,
            task_filter=None,
            skip_existing=not args.force,
        )
        if not tasks_to_run:
            print("[info] No tasks to run.")
            return
        scheduled = []
        for i, t in enumerate(tasks_to_run):
            env = {}
            if devices:
                env["CUDA_VISIBLE_DEVICES"] = devices[i % len(devices)]
            label = f"clf={t['clf_model']} mode={t['mode']} split={Path(t['split_file']).stem}"
            scheduled.append({"cmd": _build_task_cmd(args, t), "env": env, "label": label})
        run_subprocess_tasks(scheduled, max_workers=int(args.jobs), label="train-clf")
        tasks_all = _iter_tasks(
            split_files=split_files,
            data_cfg=data_cfg,
            split_cfg=split_cfg,
            clf_cfg=clf_cfg,
            gen_cfg=gen_cfg,
            sweep_cfg=sweep_cfg,
            args=args,
            task_filter=None,
            skip_existing=False,
        )
        _collect_metrics(tasks_all, metric_dir=metric_dir, split_cfg=split_cfg)
        return

    tasks = _iter_tasks(
        split_files=split_files,
        data_cfg=data_cfg,
        split_cfg=split_cfg,
        clf_cfg=clf_cfg,
        gen_cfg=gen_cfg,
        sweep_cfg=sweep_cfg,
        args=args,
        task_filter=task_filter,
        skip_existing=False,
    )
    if not tasks:
        print("[info] No tasks to run.")
        return

    for task in tasks:
        _run_task(task, index_df=index_df, pp_cfg=pp_cfg, force=args.force)

    if task_filter is None:
        _collect_metrics(tasks, metric_dir=metric_dir, split_cfg=split_cfg)


if __name__ == "__main__":
    from src.cli_deprecated import exit_deprecated
    exit_deprecated("train-clf")
