#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
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
    return parser.parse_args()


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
    payload = json.dumps(
        {
            "split": split_stem,
            "subject": int(subject),
            "p": float(p),
            "clf_model": str(clf_model),
            "mode": str(mode),
            "ratio": float(ratio),
            "gen_model": str(gen_model),
            "qc_on": bool(qc_on),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    digest = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)
    return int((int(base_seed) + digest) % (2**32 - 1))


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
    args = _parse_args()
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
    ratio_list = [float(a) for a in sweep_cfg.get("ratio_list", [0.0, 0.25, 0.5, 1.0, 2.0])]
    qc_on_list = [bool(v) for v in sweep_cfg.get("qc_on", [True])]
    clf_models, gen_models, ratio_list, qc_on_list = _resolve_stage_lists(
        sweep_cfg,
        clf_models,
        gen_models,
        ratio_list,
        qc_on_list,
    )

    index_df = load_processed_index(data_cfg["index_path"])
    metric_dir = ensure_dir(ROOT / "results/metrics")

    split_files = sorted((ROOT / "data/splits").glob("subject_*_seed_*_p_*.json"))
    if not split_files:
        split_files = sorted((ROOT / "data/splits").glob(f"split_{split_cfg['protocol']}_seed*.json"))

    rows = []
    for sf in split_files:
        split = _load_split(sf)
        if not _in_allowed_grid(split, split_cfg=split_cfg, data_cfg=data_cfg):
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
                    set_seed(run_seed)
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
                    m = train_classifier(
                        split,
                        index_df,
                        clf_run_cfg,
                        pp_cfg,
                        out_dir,
                        mode=mode,
                        ratio=ratio,
                        evaluate_test=evaluate_test,
                    )
                    rows.append(
                        {
                            "split": sf.stem,
                            "split_file": str(sf),
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
                            **m,
                        }
                    )
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
                        set_seed(run_seed)

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
                        m = train_classifier(
                            split,
                            index_df,
                            clf_run_cfg,
                            pp_cfg,
                            out_dir,
                            mode=mode,
                            ratio=ratio,
                            evaluate_test=evaluate_test,
                        )
                        rows.append(
                            {
                                "split": sf.stem,
                                "split_file": str(sf),
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
                                **m,
                            }
                        )
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
                                set_seed(run_seed)
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
                                m = train_classifier(
                                    split,
                                    index_df,
                                    clf_run_cfg,
                                    pp_cfg,
                                    out_dir,
                                    mode=mode,
                                    ratio=ratio,
                                    synth_npz=str(synth_npz),
                                    evaluate_test=evaluate_test,
                                )
                                rows.append(
                                    {
                                        "split": sf.stem,
                                        "split_file": str(sf),
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
                                        **m,
                                    }
                                )
                    continue

                print(f"[warn] unsupported augmentation mode: {mode}")

    if rows:
        out_csv = metric_dir / f"clf_{split_cfg['protocol']}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved clf metrics -> {out_csv}")


if __name__ == "__main__":
    main()
