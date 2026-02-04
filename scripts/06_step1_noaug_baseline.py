#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dataio import load_processed_index
from src.models_clf import normalize_classifier_type
from src.train_clf import train_classifier
from src.utils import ensure_dir, load_yaml, set_seed


def _load_split(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    split_cfg = load_yaml(ROOT / "configs/split.yaml")
    pp_cfg = load_yaml(ROOT / "configs/preprocess.yaml")
    clf_cfg = load_yaml(ROOT / "configs/clf.yaml")
    sweep_cfg = load_yaml(ROOT / "configs/sweep.yaml")

    p_target = float(sweep_cfg.get("step1_p", 1.0))
    profile = str(sweep_cfg.get("vram_profile", "6gb"))
    clf_models = [normalize_classifier_type(m) for m in sweep_cfg.get("clf_models", ["eegnet", "svm", "eeg_conformer", "atcnet", "ctnet"])]
    use_paper = bool(sweep_cfg.get("use_paper_presets", clf_cfg.get("paper_presets", {}).get("enabled", False)))

    index_df = load_processed_index(data_cfg["index_path"])
    out_dir = ensure_dir(ROOT / "results/metrics")

    tag = str(p_target).replace(".", "p")
    split_files = sorted((ROOT / "data/splits").glob(f"subject_*_seed_*_p_{tag}.json"))
    if not split_files:
        raise RuntimeError(f"No split files for p={p_target}. Run scripts/01_make_splits.py first.")

    rows = []
    for sf in split_files:
        split = _load_split(sf)
        subject = int(split["subject"])
        seed = int(split["seed"])
        set_seed(seed)

        for clf_model in clf_models:
            cfg = copy.deepcopy(clf_cfg)
            cfg["model"]["type"] = clf_model
            cfg["augmentation"]["modes"] = ["none"]

            if use_paper:
                pp = cfg.get("paper_presets", {}).get(clf_model, {})
                for key in ("epochs", "batch_size", "lr", "weight_decay", "num_workers", "device"):
                    if key in pp:
                        cfg["train"][key] = pp[key]
            else:
                bs = cfg.get("vram_presets", {}).get(profile, {}).get(clf_model, {}).get("batch_size", 0)
                if int(bs) > 0:
                    cfg["train"]["batch_size"] = int(bs)

            run_dir = ROOT / "runs/clf" / f"step1_none__subj-{subject:02d}__seed-{seed}__p-{tag}__clf-{clf_model}"
            m = train_classifier(split, index_df, cfg, pp_cfg, run_dir, mode="none", ratio=0.0)
            rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "p": p_target,
                    "protocol": split_cfg["protocol"],
                    "clf_model": clf_model,
                    "mode": "none",
                    **m,
                }
            )
            print(f"done subject={subject:02d} seed={seed} clf={clf_model} acc={m['acc']:.4f}")

    out_csv = out_dir / f"baseline_cross_session_p_{tag}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")

    summary = pd.DataFrame(rows).groupby("clf_model")[["acc", "kappa", "f1_macro"]].agg(["mean", "std"]).reset_index()
    summary_csv = ROOT / "results/tables" / f"baseline_cross_session_p_{tag}_summary.csv"
    ensure_dir(summary_csv.parent)
    summary.to_csv(summary_csv, index=False)
    print(f"Saved -> {summary_csv}")


if __name__ == "__main__":
    main()
