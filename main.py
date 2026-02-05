#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        raise FileNotFoundError(f"dataset not found at {raw_dir}. Place BCICIV_2a_gdf under data/raw or project root.")


def _assert_any(pattern: str, msg: str) -> None:
    if not list(Path(".").glob(pattern)):
        raise RuntimeError(msg)


def _pipeline(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Run full pipeline (prepare -> splits -> gen -> qc -> clf -> aggregate).")
    ap.add_argument("--gen-batch", "--gen_batch", type=int, default=None, help="Override generator batch size.")
    ap.add_argument("--clf-batch", "--clf_batch", type=int, default=None, help="Override classifier batch size.")
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs already exist.")
    args = ap.parse_args(argv)

    root = Path(".").resolve()
    _ensure_common_dirs(root)
    _ensure_dataset(root)

    _run_module(s_prepare_data, [])
    if not (root / "data/processed/index.csv").exists():
        raise RuntimeError("processed index missing: data/processed/index.csv")

    _run_module(s_make_splits, [])
    _assert_any("data/splits/subject_*_seed_*_p_*.json", "no split files found under data/splits")

    gen_args: list[str] = []
    clf_args: list[str] = []
    qc_args: list[str] = []
    if args.gen_batch is not None:
        gen_args += ["--batch-size", str(args.gen_batch)]
    if args.clf_batch is not None:
        clf_args += ["--batch-size", str(args.clf_batch)]
    if args.force:
        gen_args.append("--force")
        clf_args.append("--force")
        qc_args.append("--force")

    _run_module(s_train_gen, gen_args)
    _assert_any("runs/gen/*/ckpt.pt", "generator checkpoints missing under runs/gen/*/ckpt.pt")

    _run_module(s_sample_qc, qc_args)
    _assert_any("runs/synth/*.npz", "synthetic samples missing under runs/synth")

    _run_module(s_train_clf, clf_args)
    _assert_any("results/metrics/clf_*.csv", "classifier metrics missing under results/metrics")

    _run_module(s_eval_aggregate, [])
    _assert_any("results/tables/*.csv", "aggregation tables missing under results/tables")

    print("[info] Main sweep finished with evaluate_test=false by default.")
    print("[info] For final test-only evaluation, run:")
    print("       python main.py final-test --input-csv results/metrics/clf_cross_session.csv --output-csv results/metrics/clf_cross_session_test.csv")
    print("       python main.py eval-aggregate --metrics-file clf_cross_session_test.csv")


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or sys.argv[1:])
    if not argv:
        argv = ["pipeline"]

    cmd = argv[0]
    rest = argv[1:]

    if cmd in {"pipeline", "all"}:
        _pipeline(rest)
        return

    commands = {
        "prepare-data": s_prepare_data,
        "make-splits": s_make_splits,
        "train-gen": s_train_gen,
        "sample-qc": s_sample_qc,
        "train-clf": s_train_clf,
        "eval-aggregate": s_eval_aggregate,
        "final-test": s_final_test,
        "step1-noaug": s_step1_noaug,
        "paper-noaug": s_paper_noaug,
        "prepare-paper-track-data": s_prepare_paper_track,
        "paper-track": s_paper_track,
        "official-faithful": s_official_faithful,
        "validate": s_validate,
        "pilot": s_small_pilot,
        "migrate-runs": s_migrate_runs,
    }

    if cmd in commands:
        _run_module(commands[cmd], rest)
        return

    print(f"[error] Unknown command: {cmd}")
    print("Use one of: pipeline, prepare-data, make-splits, train-gen, sample-qc, train-clf, eval-aggregate, final-test,")
    print("            step1-noaug, paper-noaug, prepare-paper-track-data, paper-track, official-faithful, validate, pilot, migrate-runs")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
