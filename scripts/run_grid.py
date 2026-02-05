from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.pipeline import run_experiment
from src.utils.config import load_yaml
from src.utils.results import append_result, has_primary_key, load_results


def _load_all_configs(overrides: list[str]) -> dict:
    """Load dataset, model, generator, and experiment configs.

    Inputs:
    - overrides: list of key=value overrides.

    Outputs:
    - dict with dataset/preprocess/split/qc/experiment/models/generators configs.

    Internal logic:
    - Loads each YAML file once and merges overrides into each config.
    """
    cfg = {
        "dataset": load_yaml("configs/dataset_bci2a.yaml", overrides=overrides),
        "preprocess": load_yaml("configs/preprocess.yaml", overrides=overrides),
        "split": load_yaml("configs/split.yaml", overrides=overrides),
        "qc": load_yaml("configs/qc.yaml", overrides=overrides),
        "experiment": load_yaml("configs/experiment_grid.yaml", overrides=overrides),
        "models": {
            "eegnet": load_yaml("configs/models/eegnet.yaml", overrides=overrides),
            "eegconformer": load_yaml("configs/models/eegconformer.yaml", overrides=overrides),
            "ctnet": load_yaml("configs/models/ctnet.yaml", overrides=overrides),
            "svm": load_yaml("configs/models/fbcsp_svm.yaml", overrides=overrides),
        },
        "generators": {
            "cwgan_gp": load_yaml("configs/generators/cwgan_gp.yaml", overrides=overrides),
            "ddpm": load_yaml("configs/generators/ddpm.yaml", overrides=overrides),
        },
    }
    return cfg


def _load_alpha_star(path: str | Path) -> dict[str, float]:
    """Load alpha* values from JSON if present.

    Inputs:
    - path: file path to alpha_star.json.

    Outputs:
    - dict mapping r -> alpha_ratio.

    Internal logic:
    - Returns empty dict when file is missing to avoid hard failures.
    """
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """Run the experiment grid with stage control and resume-safe behavior.

    Inputs:
    - CLI args: stage, results path, overrides.

    Outputs:
    - Appends rows to results.csv and writes run artifacts.

    Internal logic:
    - Iterates subjects/seeds/r/method/classifier/alpha/qc, skipping existing rows.
    - Stage alpha_search uses screening classifier and full alpha grid.
    - Stage final_eval uses alpha*(r) from alpha_star.json.
    """
    parser = argparse.ArgumentParser(description="Run experiment grid")
    parser.add_argument("--stage", type=str, default=None, choices=["alpha_search", "final_eval", "full"])
    parser.add_argument("--results", type=str, default="results/results.csv")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = _load_all_configs(args.override)
    exp_cfg = cfg["experiment"]  # grid + stage settings
    stage = args.stage or exp_cfg.get("stage", {}).get("mode", "full")  # alpha_search/final_eval/full
    screening_classifier = exp_cfg.get("stage", {}).get("screening_classifier", "eegnet")  # EEGNet by default
    alpha_star_path = exp_cfg.get("stage", {}).get("alpha_star_path", "./artifacts/alpha_star.json")  # alpha* cache
    alpha_search_cfg = exp_cfg.get("alpha_search", {})  # proxy mode config

    methods = exp_cfg.get("methods", [])  # C0/C1/C2/GenAug
    classifiers = exp_cfg.get("classifiers", [])  # model list
    generators = exp_cfg.get("generators", [])  # generator list
    alpha_list = [float(x) for x in exp_cfg.get("alpha_ratio_list", [0.0])]  # synth ratios
    qc_on_list = [bool(x) for x in exp_cfg.get("qc_on", [False])]  # QC ablation flags

    results_df = load_results(args.results)
    alpha_star = _load_alpha_star(alpha_star_path)

    subjects = cfg["dataset"]["subjects"]
    seeds = cfg["split"]["seeds"]
    r_list = [float(r) for r in cfg["split"]["low_data_fracs"]]

    compute_distance = stage != "alpha_search"

    for subject in subjects:
        for seed in seeds:
            for r in r_list:
                for method in methods:
                    if stage == "alpha_search":
                        cls_list = [screening_classifier]
                    else:
                        cls_list = classifiers

                    for classifier in cls_list:
                        if method != "GenAug":
                            row_key = {
                                "subject": subject,
                                "seed": seed,
                                "r": r,
                                "classifier": classifier,
                                "method": method,
                                "generator": "none",
                                "qc_on": False,
                                "alpha_ratio": 0.0,
                            }
                            if has_primary_key(results_df, row_key):
                                continue
                            row = run_experiment(
                                subject=subject,
                                seed=seed,
                                r=r,
                                method=method,
                                classifier=classifier,
                                generator="none",
                                alpha_ratio=0.0,
                                qc_on=False,
                                dataset_cfg=cfg["dataset"],
                                preprocess_cfg=cfg["preprocess"],
                                split_cfg=cfg["split"],
                                model_cfgs=cfg["models"],
                                gen_cfgs=cfg["generators"],
                                qc_cfg=cfg["qc"],
                                results_path=args.results,
                                stage=stage,
                                compute_distance=compute_distance,
                                alpha_search_cfg=alpha_search_cfg,
                            )
                            appended = append_result(args.results, row)
                            if appended:
                                results_df = load_results(args.results)
                            continue

                        # GenAug branch
                        if stage == "alpha_search":
                            alpha_candidates = alpha_list
                        elif stage == "final_eval":
                            alpha_candidates = [float(alpha_star.get(str(r), alpha_list[-1]))]
                        else:
                            alpha_candidates = alpha_list

                        for generator in generators:
                            for alpha_ratio in alpha_candidates:
                                for qc_on in qc_on_list:
                                    row_key = {
                                        "subject": subject,
                                        "seed": seed,
                                        "r": r,
                                        "classifier": classifier,
                                        "method": method,
                                        "generator": generator,
                                        "qc_on": bool(qc_on),
                                        "alpha_ratio": float(alpha_ratio),
                                    }
                                    if has_primary_key(results_df, row_key):
                                        continue

                                    row = run_experiment(
                                        subject=subject,
                                        seed=seed,
                                        r=r,
                                        method="GenAug",
                                        classifier=classifier,
                                        generator=generator,
                                        alpha_ratio=float(alpha_ratio),
                                        qc_on=bool(qc_on),
                                        dataset_cfg=cfg["dataset"],
                                        preprocess_cfg=cfg["preprocess"],
                                        split_cfg=cfg["split"],
                                        model_cfgs=cfg["models"],
                                        gen_cfgs=cfg["generators"],
                                        qc_cfg=cfg["qc"],
                                        results_path=args.results,
                                        stage=stage,
                                        compute_distance=compute_distance,
                                        alpha_search_cfg=alpha_search_cfg,
                                    )
                                    appended = append_result(args.results, row)
                                    if appended:
                                        results_df = load_results(args.results)


if __name__ == "__main__":
    main()
