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
from src.models_gen import normalize_generator_type
from src.train_clf import train_classifier
from src.utils import ensure_dir, load_yaml, make_exp_id, set_seed


def _load_split(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_clf_cfg(base_cfg: dict, clf_model: str, sweep_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["type"] = clf_model

    use_paper = bool(sweep_cfg.get("use_paper_presets", cfg.get("paper_presets", {}).get("enabled", False)))
    if use_paper:
        pp = cfg.get("paper_presets", {}).get(clf_model, {})
        for key in ("epochs", "batch_size", "lr", "weight_decay", "num_workers", "device"):
            if key in pp:
                cfg["train"][key] = pp[key]
        # "논문 설정 그대로"를 우선한다.
        return cfg

    if not bool(sweep_cfg.get("apply_vram_presets", True)):
        return cfg

    profile = str(sweep_cfg.get("vram_profile", "6gb"))
    preset = cfg.get("vram_presets", {}).get(profile, {}).get(clf_model, {})
    if "batch_size" in preset and int(preset["batch_size"]) > 0:
        cfg["train"]["batch_size"] = int(preset["batch_size"])

    return cfg


def _find_synth_npz(root: Path, gen_model: str, split_stem: str, qc_on: bool) -> Path:
    sub = "runs/synth_qc" if qc_on else "runs/synth"
    name = f"synth_qc_{gen_model}_{split_stem}.npz" if qc_on else f"synth_{gen_model}_{split_stem}.npz"
    p = root / sub / name
    if p.exists():
        return p

    legacy = f"synth_qc_{split_stem}.npz" if qc_on else f"synth_{split_stem}.npz"
    return root / sub / legacy


def _resolve_stage_lists(sweep_cfg: dict, clf_models: list[str], gen_models: list[str]) -> tuple[list[str], list[str], list[float], list[bool]]:
    stage = str(sweep_cfg.get("stage_mode", "full"))
    ratios = [float(r) for r in sweep_cfg.get("synth_ratio_list", [0.0, 0.1, 0.3, 0.5])]
    qc_on = [bool(v) for v in sweep_cfg.get("qc_on", [True, False])]

    if stage == "screening":
        clf_models = [normalize_classifier_type(str(sweep_cfg.get("screening_classifier", "eegnet")))]
        qc_on = [True]
        return clf_models, gen_models, ratios, qc_on

    if stage == "full":
        # optional reduced ratio list for stage2
        full_ratios = sweep_cfg.get("full_stage_ratios")
        if full_ratios is not None:
            ratios = [float(r) for r in full_ratios]
        return clf_models, gen_models, ratios, qc_on

    raise ValueError(f"Unknown stage_mode: {stage}")


def main() -> None:
    data_cfg = load_yaml(ROOT / "configs/data.yaml")
    split_cfg = load_yaml(ROOT / "configs/split.yaml")
    pp_cfg = load_yaml(ROOT / "configs/preprocess.yaml")
    clf_cfg = load_yaml(ROOT / "configs/clf.yaml")
    gen_cfg = load_yaml(ROOT / "configs/gen.yaml")
    sweep_cfg = load_yaml(ROOT / "configs/sweep.yaml")

    default_clf = normalize_classifier_type(str(clf_cfg["model"].get("type", "eegnet")))
    default_gen = normalize_generator_type(str(gen_cfg["model"].get("type", "cvae")))

    clf_models = [normalize_classifier_type(m) for m in sweep_cfg.get("clf_models", [default_clf])]
    gen_models = [normalize_generator_type(m) for m in sweep_cfg.get("gen_models", [default_gen])]
    clf_models, gen_models, synth_ratio_list, qc_on_list = _resolve_stage_lists(sweep_cfg, clf_models, gen_models)

    index_df = load_processed_index(data_cfg["index_path"])
    metric_dir = ensure_dir(ROOT / "results/metrics")

    split_files = sorted((ROOT / "data/splits").glob("subject_*_seed_*_p_*.json"))
    if not split_files:
        split_files = sorted((ROOT / "data/splits").glob(f"split_{split_cfg['protocol']}_seed*.json"))

    rows = []
    for sf in split_files:
        split = _load_split(sf)
        seed = int(split["seed"])
        set_seed(seed)
        subject = split.get("subject", "all")
        p = float(split.get("low_data_frac", 1.0))

        for clf_model in clf_models:
            clf_run_cfg = _build_clf_cfg(clf_cfg, clf_model=clf_model, sweep_cfg=sweep_cfg)

            for mode in clf_run_cfg["augmentation"]["modes"]:
                if mode != "gen_aug":
                    exp_id = make_exp_id(
                        "clf",
                        split=sf.stem,
                        subject=subject,
                        p=p,
                        clf=clf_model,
                        mode=mode,
                        ratio=0.0,
                        qc=False,
                    )
                    out_dir = ROOT / "runs/clf" / exp_id
                    m = train_classifier(split, index_df, clf_run_cfg, pp_cfg, out_dir, mode=mode, synth_ratio=0.0)
                    rows.append(
                        {
                            "split": sf.stem,
                            "subject": subject,
                            "p": p,
                            "seed": seed,
                            "protocol": split.get("protocol", split_cfg["protocol"]),
                            "clf_model": clf_model,
                            "gen_model": "none",
                            "mode": mode,
                            "synth_ratio": 0.0,
                            "qc_on": False,
                            **m,
                        }
                    )
                    continue

                for ratio in synth_ratio_list:
                    ratio = float(ratio)
                    if ratio <= 0.0:
                        exp_id = make_exp_id(
                            "clf",
                            split=sf.stem,
                            subject=subject,
                            p=p,
                            clf=clf_model,
                            gen="none",
                            mode=mode,
                            ratio=0.0,
                            qc=False,
                        )
                        out_dir = ROOT / "runs/clf" / exp_id
                        m = train_classifier(split, index_df, clf_run_cfg, pp_cfg, out_dir, mode=mode, synth_ratio=0.0)
                        rows.append(
                            {
                                "split": sf.stem,
                                "subject": subject,
                                "p": p,
                                "seed": seed,
                                "protocol": split.get("protocol", split_cfg["protocol"]),
                                "clf_model": clf_model,
                                "gen_model": "none",
                                "mode": mode,
                                "synth_ratio": 0.0,
                                "qc_on": False,
                                **m,
                            }
                        )
                        continue

                    for gen_model in gen_models:
                        for qc_on in qc_on_list:
                            synth_npz = _find_synth_npz(ROOT, gen_model=gen_model, split_stem=sf.stem, qc_on=bool(qc_on))
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
                                qc=qc_on,
                            )
                            out_dir = ROOT / "runs/clf" / exp_id
                            m = train_classifier(
                                split,
                                index_df,
                                clf_run_cfg,
                                pp_cfg,
                                out_dir,
                                mode=mode,
                                synth_ratio=ratio,
                                synth_npz=str(synth_npz),
                            )
                            rows.append(
                                {
                                    "split": sf.stem,
                                    "subject": subject,
                                    "p": p,
                                    "seed": seed,
                                    "protocol": split.get("protocol", split_cfg["protocol"]),
                                    "clf_model": clf_model,
                                    "gen_model": gen_model,
                                    "mode": mode,
                                    "synth_ratio": ratio,
                                    "qc_on": bool(qc_on),
                                    **m,
                                }
                            )

    if rows:
        out_csv = metric_dir / f"clf_{split_cfg['protocol']}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved clf metrics -> {out_csv}")


if __name__ == "__main__":
    main()
