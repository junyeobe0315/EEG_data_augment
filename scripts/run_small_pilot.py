#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

try:
    from scripts._script_utils import project_root
except ImportError:  # pragma: no cover - direct script execution
    from _script_utils import project_root

import pandas as pd

ROOT = project_root(__file__)

from src.dataio import load_processed_index, load_samples_by_ids
from src.models_clf import normalize_classifier_type
from src.models_gen import normalize_generator_type
from src.qc import run_qc
from src.sample_gen import sample_by_class, save_synth_npz
from src.train_clf import train_classifier
from src.train_gen import train_generative_model
from src.utils import ensure_dir, load_json, load_yaml, make_exp_id, save_json, set_seed, split_file_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a small one-shot pilot (gen + synth/qc + clf baseline/gen_aug).")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.20)
    ap.add_argument("--gen-model", type=str, default="cvae")
    ap.add_argument("--clf-model", type=str, default="eegnet_tf_faithful")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--qc-on", action="store_true")
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--gen-epochs", type=int, default=3)
    ap.add_argument("--clf-steps", type=int, default=200)
    ap.add_argument("--n-per-class", type=int, default=120)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    split_cfg = load_yaml(ROOT / "configs/split.yaml")
    pp_cfg = load_yaml(ROOT / "configs/preprocess.yaml")
    gen_cfg = load_yaml(ROOT / "configs/gen.yaml")
    clf_cfg = load_yaml(ROOT / "configs/clf.yaml")
    qc_cfg = load_yaml(ROOT / "configs/qc.yaml")

    split_path = split_file_path(ROOT, subject=int(args.subject), seed=int(args.seed), p=float(args.p))
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}. Run `python main.py make-splits` first.")

    split = load_json(split_path)
    index_df = load_processed_index(ROOT / data_cfg["index_path"])

    gen_model = normalize_generator_type(str(args.gen_model))
    clf_model = normalize_classifier_type(str(args.clf_model))

    pilot_meta = {
        "subject": int(args.subject),
        "seed": int(args.seed),
        "p": float(args.p),
        "gen": gen_model,
        "clf": clf_model,
        "ratio": float(args.ratio),
    }
    if str(args.tag).strip():
        pilot_meta["tag"] = str(args.tag).strip()
    pilot_id = make_exp_id("pilot", **pilot_meta)

    # 1) Train generator (lightweight settings)
    gen_run_cfg = copy.deepcopy(gen_cfg)
    gen_run_cfg["model"]["type"] = gen_model
    gen_run_cfg["train"]["epochs"] = int(args.gen_epochs)
    gen_run_cfg["train"]["num_workers"] = 0
    gen_run_cfg.setdefault("train", {}).setdefault("checkpoint_every", 1)
    gen_run_cfg.setdefault("checkpoint_selection", {})["enabled"] = False
    gen_run_cfg.setdefault("sample", {})["n_per_class"] = int(args.n_per_class)
    gen_run_cfg.setdefault("data", {})["sfreq"] = int(data_cfg.get("sfreq", 250))

    gen_out = ROOT / "runs/gen" / pilot_id
    set_seed(int(args.seed) + 10)
    train_generative_model(
        split=split,
        index_df=index_df,
        gen_cfg=gen_run_cfg,
        preprocess_cfg=pp_cfg,
        out_dir=gen_out,
        qc_cfg=qc_cfg,
        clf_cfg=clf_cfg,
        base_seed=int(args.seed) + 10,
    )
    ckpt_path = gen_out / "ckpt.pt"

    # 2) Sample + QC
    set_seed(int(args.seed) + 20)
    synth = sample_by_class(
        ckpt_path=ckpt_path,
        n_per_class=int(args.n_per_class),
        num_classes=len(data_cfg["class_names"]),
        device=gen_run_cfg["train"].get("device", "auto"),
    )

    synth_dir = ensure_dir(ROOT / "runs/synth")
    synth_path = synth_dir / f"{pilot_id}.npz"
    save_synth_npz(synth_path, synth)

    x_real_train, y_real_train = load_samples_by_ids(index_df, split["train_ids"])
    set_seed(int(args.seed) + 30)
    synth_used = synth
    synth_used_path = synth_path
    qc_report = {
        "n_before": int(synth["X"].shape[0]),
        "n_after": int(synth["X"].shape[0]),
        "keep_ratio": 1.0,
        "qc_on": False,
    }
    if bool(args.qc_on):
        kept, report = run_qc(
            real_x=x_real_train,
            synth=synth,
            sfreq=int(data_cfg["sfreq"]),
            cfg=qc_cfg,
            real_y=y_real_train,
        )
        qc_dir = ensure_dir(ROOT / "runs/synth_qc")
        synth_used_path = qc_dir / f"{pilot_id}.npz"
        save_synth_npz(synth_used_path, kept)
        synth_used = kept
        qc_report = {**report, "qc_on": True}

    # 3) Train classifier baseline + gen_aug
    clf_run_cfg = copy.deepcopy(clf_cfg)
    clf_run_cfg["model"]["type"] = clf_model
    clf_run_cfg.setdefault("evaluation", {})["evaluate_test"] = bool(args.evaluate_test)
    clf_run_cfg.setdefault("train", {}).setdefault("step_control", {})
    clf_run_cfg["train"]["step_control"]["enabled"] = True
    clf_run_cfg["train"]["step_control"]["total_steps"] = int(args.clf_steps)
    clf_run_cfg["train"]["step_control"]["steps_per_eval"] = max(10, int(args.clf_steps // 5))

    base_id = make_exp_id("pilot_clf", **pilot_meta, mode="baseline")
    base_out = ROOT / "runs/clf" / base_id
    set_seed(int(args.seed) + 40)
    m_base = train_classifier(
        split=split,
        index_df=index_df,
        clf_cfg=clf_run_cfg,
        preprocess_cfg=pp_cfg,
        out_dir=base_out,
        mode="none",
        ratio=0.0,
        evaluate_test=bool(args.evaluate_test),
    )

    genaug_id = make_exp_id("pilot_clf", **pilot_meta, mode="genaug")
    genaug_out = ROOT / "runs/clf" / genaug_id
    set_seed(int(args.seed) + 50)
    m_gen = train_classifier(
        split=split,
        index_df=index_df,
        clf_cfg=clf_run_cfg,
        preprocess_cfg=pp_cfg,
        out_dir=genaug_out,
        mode="gen_aug",
        ratio=float(args.ratio),
        synth_npz=str(synth_used_path),
        evaluate_test=bool(args.evaluate_test),
    )

    rows = [
        {
            "pilot_id": pilot_id,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "p": float(args.p),
            "gen_model": gen_model,
            "clf_model": clf_model,
            "condition": "C0_no_aug",
            "ratio_requested": 0.0,
            "synth_npz": "",
            **m_base,
        },
        {
            "pilot_id": pilot_id,
            "subject": int(args.subject),
            "seed": int(args.seed),
            "p": float(args.p),
            "gen_model": gen_model,
            "clf_model": clf_model,
            "condition": "GenAug",
            "ratio_requested": float(args.ratio),
            "synth_npz": str(synth_used_path),
            **m_gen,
        },
    ]

    out_dir = ensure_dir(ROOT / "results/metrics")
    out_csv = out_dir / f"{pilot_id}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    report = {
        "pilot_id": pilot_id,
        "split_file": str(split_path),
        "generator_ckpt": str(ckpt_path),
        "synth_path": str(synth_path),
        "synth_used_path": str(synth_used_path),
        "synth_n_before_qc": int(synth["X"].shape[0]),
        "synth_n_after_qc": int(synth_used["X"].shape[0]),
        "qc": qc_report,
        "baseline_acc": float(m_base.get("acc", float("nan"))),
        "genaug_acc": float(m_gen.get("acc", float("nan"))),
        "acc_gain": float(m_gen.get("acc", float("nan")) - m_base.get("acc", float("nan"))),
    }
    out_json = out_dir / f"{pilot_id}.json"
    save_json(out_json, report)

    print(f"[pilot] saved metrics -> {out_csv}")
    print(f"[pilot] saved summary -> {out_json}")


if __name__ == "__main__":
    main()
