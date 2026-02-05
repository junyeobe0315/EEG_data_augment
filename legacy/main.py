#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

from src.utils import ensure_dir

from scripts import eval_and_aggregate as s_eval_aggregate
from scripts import final_test_eval as s_final_test
from scripts import make_splits as s_make_splits
from scripts import migrate_run_dirs as s_migrate_runs
from scripts import official_faithful_track as s_official_faithful
from scripts import paper_noaug as s_paper_noaug
from scripts import paper_track as s_paper_track
from scripts import prepare_data as s_prepare_data
from scripts import prepare_paper_track_data as s_prepare_paper_track
from scripts import run_small_pilot as s_small_pilot
from scripts import sample_and_qc as s_sample_qc
from scripts import step1_noaug_baseline as s_step1_noaug
from scripts import train_clf as s_train_clf
from scripts import train_gen as s_train_gen
from scripts import validate_pipeline as s_validate


PIPELINE_STEPS = (
    "prepare-data",
    "make-splits",
    "train-gen",
    "sample-qc",
    "train-clf",
    "eval-aggregate",
)


def _run_module(mod, argv: list[str]) -> None:
    old_argv = sys.argv
    sys.argv = [str(getattr(mod, "__file__", "module"))] + list(argv)
    try:
        mod.main()
    finally:
        sys.argv = old_argv


def _ensure_common_dirs(root: Path) -> None:
    ensure_dir(root / "data/raw")
    ensure_dir(root / "data/processed")
    ensure_dir(root / "data/splits")
    ensure_dir(root / "runs/gen")
    ensure_dir(root / "runs/synth")
    ensure_dir(root / "runs/synth_qc")
    ensure_dir(root / "runs/clf")
    ensure_dir(root / "results/metrics")
    ensure_dir(root / "results/tables")
    ensure_dir(root / "results/figures")


def _ensure_dataset(root: Path) -> None:
    raw_dir = root / "data/raw/BCICIV_2a_gdf"
    if (not raw_dir.exists()) and (root / "BCICIV_2a_gdf").exists():
        raw_dir.symlink_to(root / "BCICIV_2a_gdf")
        print(f"[info] symlinked dataset -> {raw_dir}")
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"dataset not found at {raw_dir}. Place BCICIV_2a_gdf under data/raw or project root."
        )


def _assert_any(pattern: str, msg: str) -> None:
    if not list(Path(".").glob(pattern)):
        raise RuntimeError(msg)


def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _resolve_steps(steps_raw: str | None, skip_raw: str | None) -> list[str]:
    steps = _parse_csv_list(steps_raw)
    skip = set(_parse_csv_list(skip_raw))

    if steps:
        unknown = [s for s in steps if s not in PIPELINE_STEPS]
        if unknown:
            raise ValueError(f"Unknown steps: {unknown}. Allowed: {', '.join(PIPELINE_STEPS)}")
    else:
        steps = list(PIPELINE_STEPS)

    steps = [s for s in steps if s not in skip]
    if not steps:
        raise ValueError("No pipeline steps selected. Check --steps/--skip.")
    return steps


def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma list of steps to run. Default: all.",
    )
    p.add_argument(
        "--skip",
        type=str,
        default=None,
        help="Comma list of steps to skip.",
    )
    p.add_argument("--print-steps", action="store_true", help="Print resolved steps and exit.")
    p.add_argument("--dry-run", action="store_true", help="Only show steps; do not execute.")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values (e.g., gen.train.batch_size=64).",
    )
    p.add_argument("--jobs", type=int, default=1, help="Default parallel jobs for train-gen/sample-qc/train-clf.")
    p.add_argument("--gen-jobs", type=int, default=None, help="Parallel jobs for train-gen.")
    p.add_argument("--qc-jobs", type=int, default=None, help="Parallel jobs for sample-qc.")
    p.add_argument("--clf-jobs", type=int, default=None, help="Parallel jobs for train-clf.")
    p.add_argument("--devices", type=str, default=None, help="Comma-separated CUDA device list for parallel jobs.")

    p.add_argument("--gen-batch", "--gen_batch", type=int, default=None, help="Override generator batch size.")
    p.add_argument("--gen-epochs", type=int, default=None, help="Override generator epochs.")
    p.add_argument("--gen-lr", type=float, default=None, help="Override generator base LR.")
    p.add_argument("--gen-cvae-lr", type=float, default=None, help="Override generator CVAE LR.")
    p.add_argument("--gen-gan-lr", type=float, default=None, help="Override generator GAN LR.")
    p.add_argument("--gen-ddpm-lr", type=float, default=None, help="Override generator DDPM LR.")
    p.add_argument(
        "--gen-optimizer",
        choices=["adam", "adamw", "sgd"],
        default=None,
        help="Override generator optimizer type.",
    )
    p.add_argument("--gen-weight-decay", type=float, default=None, help="Override generator weight decay.")
    p.add_argument("--gen-num-workers", type=int, default=None, help="Override generator dataloader workers.")
    p.add_argument("--gen-device", type=str, default=None, help="Override generator device.")

    p.add_argument("--clf-batch", "--clf_batch", type=int, default=None, help="Override classifier batch size.")
    p.add_argument("--clf-epochs", type=int, default=None, help="Override classifier epochs.")
    p.add_argument("--clf-lr", type=float, default=None, help="Override classifier learning rate.")
    p.add_argument(
        "--clf-optimizer",
        choices=["adam", "adamw", "sgd"],
        default=None,
        help="Override classifier optimizer type.",
    )
    p.add_argument("--clf-weight-decay", type=float, default=None, help="Override classifier weight decay.")
    p.add_argument("--clf-num-workers", type=int, default=None, help="Override classifier dataloader workers.")
    p.add_argument("--clf-device", type=str, default=None, help="Override classifier device.")

    p.add_argument("--force", action="store_true", help="Re-run even if outputs already exist.")
    p.add_argument("--fast", action="store_true", help="Enable speed-oriented settings (AMP/TF32/pin_memory).")


def _build_gen_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.gen_batch is not None:
        out += ["--batch-size", str(args.gen_batch)]
    if args.gen_epochs is not None:
        out += ["--epochs", str(args.gen_epochs)]
    if args.gen_lr is not None:
        out += ["--lr", str(args.gen_lr)]
    if args.gen_cvae_lr is not None:
        out += ["--cvae-lr", str(args.gen_cvae_lr)]
    if args.gen_gan_lr is not None:
        out += ["--gan-lr", str(args.gen_gan_lr)]
    if args.gen_ddpm_lr is not None:
        out += ["--ddpm-lr", str(args.gen_ddpm_lr)]
    if args.gen_optimizer is not None:
        out += ["--optimizer", str(args.gen_optimizer)]
    if args.gen_weight_decay is not None:
        out += ["--weight-decay", str(args.gen_weight_decay)]
    if args.gen_num_workers is not None:
        out += ["--num-workers", str(args.gen_num_workers)]
    if args.gen_device is not None:
        out += ["--device", str(args.gen_device)]
    if args.force:
        out.append("--force")
    if args.fast:
        out.append("--fast")
    for item in args.set or []:
        out += ["--set", item]
    gen_jobs = args.gen_jobs if args.gen_jobs is not None else args.jobs
    if gen_jobs and int(gen_jobs) > 1:
        out += ["--jobs", str(gen_jobs)]
    if args.devices:
        out += ["--devices", str(args.devices)]
    return out


def _build_clf_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.clf_batch is not None:
        out += ["--batch-size", str(args.clf_batch)]
    if args.clf_epochs is not None:
        out += ["--epochs", str(args.clf_epochs)]
    if args.clf_lr is not None:
        out += ["--lr", str(args.clf_lr)]
    if args.clf_optimizer is not None:
        out += ["--optimizer", str(args.clf_optimizer)]
    if args.clf_weight_decay is not None:
        out += ["--weight-decay", str(args.clf_weight_decay)]
    if args.clf_num_workers is not None:
        out += ["--num-workers", str(args.clf_num_workers)]
    if args.clf_device is not None:
        out += ["--device", str(args.clf_device)]
    if args.force:
        out.append("--force")
    if args.fast:
        out.append("--fast")
    for item in args.set or []:
        out += ["--set", item]
    clf_jobs = args.clf_jobs if args.clf_jobs is not None else args.jobs
    if clf_jobs and int(clf_jobs) > 1:
        out += ["--jobs", str(clf_jobs)]
    if args.devices:
        out += ["--devices", str(args.devices)]
    return out


def _build_qc_args(args: argparse.Namespace) -> list[str]:
    out = ["--force"] if args.force else []
    for item in args.set or []:
        out += ["--set", item]
    qc_jobs = args.qc_jobs if args.qc_jobs is not None else args.jobs
    if qc_jobs and int(qc_jobs) > 1:
        out += ["--jobs", str(qc_jobs)]
    if args.devices:
        out += ["--devices", str(args.devices)]
    return out


def _run_pipeline(args: argparse.Namespace) -> None:
    steps = _resolve_steps(args.steps, args.skip)
    if args.print_steps or args.dry_run:
        print("[pipeline] steps:")
        for s in steps:
            print(f" - {s}")
        if args.dry_run:
            return

    root = Path(".").resolve()
    _ensure_common_dirs(root)
    if any(step != "eval-aggregate" for step in steps):
        _ensure_dataset(root)

    gen_args = _build_gen_args(args)
    clf_args = _build_clf_args(args)
    qc_args = _build_qc_args(args)

    for step in steps:
        if step == "prepare-data":
            _run_module(s_prepare_data, [])
            _assert_any("data/processed/index.csv", "processed index missing: data/processed/index.csv")
        elif step == "make-splits":
            _run_module(s_make_splits, [])
            _assert_any("data/splits/subject_*_seed_*_p_*.json", "no split files found under data/splits")
        elif step == "train-gen":
            _run_module(s_train_gen, gen_args)
            _assert_any("runs/gen/*/ckpt.pt", "generator checkpoints missing under runs/gen/*/ckpt.pt")
        elif step == "sample-qc":
            _run_module(s_sample_qc, qc_args)
            _assert_any("runs/synth/*.npz", "synthetic samples missing under runs/synth")
        elif step == "train-clf":
            _run_module(s_train_clf, clf_args)
            _assert_any("results/metrics/clf_*.csv", "classifier metrics missing under results/metrics")
        elif step == "eval-aggregate":
            _run_module(s_eval_aggregate, [])
            _assert_any("results/tables/*.csv", "aggregation tables missing under results/tables")
        else:
            raise RuntimeError(f"Unsupported pipeline step: {step}")

    if "train-clf" in steps:
        print("[info] Main sweep finished with evaluate_test=false by default.")
        print("[info] For final test-only evaluation, run:")
        print(
            "       python main.py final-test --input-csv results/metrics/clf_cross_session.csv "
            "--output-csv results/metrics/clf_cross_session_test.csv"
        )
        print("       python main.py eval-aggregate --metrics-file clf_cross_session_test.csv")


def _add_forward_subcommand(
    subparsers: argparse._SubParsersAction,
    name: str,
    mod,
    help_text: str,
) -> None:
    sp = subparsers.add_parser(name, help=help_text)
    sp.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values (e.g., gen.train.batch_size=64).",
    )
    sp.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the underlying command.")
    sp.set_defaults(func=lambda ns, _m=mod: _run_module(_m, list(ns.args)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EEG augmentation pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    p_pipe = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline (prepare -> splits -> gen -> qc -> clf -> aggregate).",
    )
    _add_pipeline_args(p_pipe)
    p_pipe.set_defaults(func=_run_pipeline)

    _add_forward_subcommand(subparsers, "prepare-data", s_prepare_data, "Prepare processed dataset index.")
    _add_forward_subcommand(subparsers, "make-splits", s_make_splits, "Create split files.")
    _add_forward_subcommand(subparsers, "train-gen", s_train_gen, "Train generator models.")
    _add_forward_subcommand(subparsers, "sample-qc", s_sample_qc, "Sample generators and run QC.")
    _add_forward_subcommand(subparsers, "train-clf", s_train_clf, "Train classifiers.")
    _add_forward_subcommand(subparsers, "eval-aggregate", s_eval_aggregate, "Aggregate metrics and figures.")
    _add_forward_subcommand(subparsers, "final-test", s_final_test, "Final test-only evaluation.")
    _add_forward_subcommand(subparsers, "step1-noaug", s_step1_noaug, "Step1 baseline (no augmentation).")
    _add_forward_subcommand(subparsers, "paper-noaug", s_paper_noaug, "Paper-track baseline (no aug).")
    _add_forward_subcommand(subparsers, "prepare-paper-track-data", s_prepare_paper_track, "Prepare paper-track data.")
    _add_forward_subcommand(subparsers, "paper-track", s_paper_track, "Run paper-faithful track.")
    _add_forward_subcommand(subparsers, "official-faithful", s_official_faithful, "Run official-faithful track.")
    _add_forward_subcommand(subparsers, "validate", s_validate, "Validate configs and splits.")
    _add_forward_subcommand(subparsers, "pilot", s_small_pilot, "Run a quick pilot.")
    _add_forward_subcommand(subparsers, "migrate-runs", s_migrate_runs, "Migrate run directory layout.")

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    argv = list(argv or sys.argv[1:])
    if not argv:
        argv = ["pipeline"]

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(2)

    if getattr(args, "set", None):
        os.environ["EEG_CFG_OVERRIDES"] = json.dumps(list(args.set))

    args.func(args)


if __name__ == "__main__":
    main()
