#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _script_utils import project_root

import pandas as pd

ROOT = project_root(__file__)

from src.dataio import load_processed_index
from src.models_clf import normalize_classifier_type
from src.models_gen import normalize_generator_type
from src.train_clf import train_classifier
from src.utils import load_yaml, p_tag, set_seed, split_file_path


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def validate_configs() -> None:
    split_cfg = load_yaml(ROOT / "configs/split.yaml")
    sweep_cfg = load_yaml(ROOT / "configs/sweep.yaml")
    clf_cfg = load_yaml(ROOT / "configs/clf.yaml")

    _assert(str(split_cfg.get("protocol", "")) == "cross_session", "split.protocol must be cross_session")

    low_data = [float(x) for x in split_cfg.get("low_data_fracs", [])]
    _assert(len(low_data) > 0, "split.low_data_fracs must not be empty")
    _assert(all(0 < x <= 1.0 for x in low_data), "split.low_data_fracs must be in (0,1]")

    ratios = [float(x) for x in sweep_cfg.get("ratio_list", [])]
    _assert(len(ratios) > 0, "sweep.ratio_list must not be empty")
    _assert(any(abs(a - 0.0) < 1e-12 for a in ratios), "sweep.ratio_list must include 0.0 baseline")

    seeds = [int(x) for x in split_cfg.get("seeds", [])]
    _assert(len(seeds) > 0, "split.seeds must not be empty")

    for m in sweep_cfg.get("clf_models", []):
        normalize_classifier_type(str(m))
    for g in sweep_cfg.get("gen_models", []):
        normalize_generator_type(str(g))

    modes = [str(m) for m in clf_cfg.get("augmentation", {}).get("modes", [])]
    _assert("none" in modes, "clf.augmentation.modes must include 'none'")



def validate_data_and_splits() -> None:
    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    split_cfg = load_yaml(ROOT / "configs/split.yaml")

    idx = load_processed_index(data_cfg["index_path"])
    _assert(len(idx) > 0, "processed index is empty")

    split_dir = ROOT / "data/splits"
    _assert(split_dir.exists(), "data/splits directory missing")

    p_files = sorted(split_dir.glob("subject_*_seed_*_p_*.json"))
    _assert(len(p_files) > 0, "no low-data split files found")

    allowed_seeds = set(int(s) for s in split_cfg.get("seeds", []))
    allowed_p = set(float(p) for p in split_cfg.get("low_data_fracs", []))
    seen = set()

    for sf in p_files:
        with open(sf, "r", encoding="utf-8") as f:
            sp = json.load(f)
        seed = int(sp["seed"])
        p = float(sp.get("low_data_frac", 1.0))
        subj = int(sp["subject"])
        if seed in allowed_seeds and p in allowed_p:
            seen.add((subj, seed, p))

    # Expect full cartesian grid for configured subjects/seeds/p.
    subjects = [int(s) for s in data_cfg.get("subjects", [])]
    expected = {(s, k, p) for s in subjects for k in allowed_seeds for p in allowed_p}
    missing = sorted(expected - seen)
    _assert(len(missing) == 0, f"missing split entries count={len(missing)} (example: {missing[:3]})")


def run_smoke() -> None:
    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    split_cfg = load_yaml(ROOT / "configs/split.yaml")
    pp_cfg = load_yaml(ROOT / "configs/preprocess.yaml")
    clf_cfg = load_yaml(ROOT / "configs/clf.yaml")

    subject = int(data_cfg["subjects"][0])
    seed = int(split_cfg["seeds"][0])
    p = float(split_cfg["low_data_fracs"][0])
    sf = split_file_path(ROOT, subject=subject, seed=seed, p=p)
    with open(sf, "r", encoding="utf-8") as f:
        split = json.load(f)

    idx = load_processed_index(data_cfg["index_path"])

    cfg = dict(clf_cfg)
    cfg["model"] = dict(clf_cfg["model"])
    cfg["train"] = dict(clf_cfg["train"])
    cfg["augmentation"] = dict(clf_cfg["augmentation"])

    cfg["model"]["type"] = "eegnet_tf_faithful"
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = 8
    cfg["train"].setdefault("step_control", {})
    cfg["train"]["step_control"]["enabled"] = True
    cfg["train"]["step_control"]["total_steps"] = 10
    cfg["train"]["step_control"]["steps_per_eval"] = 5
    cfg["augmentation"]["modes"] = ["none"]

    set_seed(seed)
    out_dir = ROOT / "runs/clf" / "pipeline_validation_smoke"
    metrics = train_classifier(
        split=split,
        index_df=idx,
        clf_cfg=cfg,
        preprocess_cfg=pp_cfg,
        out_dir=out_dir,
        mode="none",
        ratio=0.0,
    )
    print(f"[smoke] acc={metrics['acc']:.4f} kappa={metrics['kappa']:.4f} f1={metrics['f1_macro']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate pipeline integrity and optionally run a tiny smoke training.")
    ap.add_argument("--smoke", action="store_true", help="Run 1-epoch tiny training smoke test.")
    args = ap.parse_args()

    validate_configs()
    print("[ok] config validation")

    validate_data_and_splits()
    print("[ok] data/split validation")

    if args.smoke:
        run_smoke()
        print("[ok] smoke training")


if __name__ == "__main__":
    main()
