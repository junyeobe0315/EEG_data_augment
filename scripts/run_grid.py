from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.pipeline import run_experiment
from src.utils.config import load_yaml
from src.utils.results import append_result, has_primary_key, load_results, PRIMARY_KEY_FIELDS


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


def _row_key_tuple(row: dict) -> tuple:
    """Build a primary-key tuple from a row dict.

    Inputs:
    - row: dict containing PRIMARY_KEY_FIELDS keys.

    Outputs:
    - tuple with values in PRIMARY_KEY_FIELDS order.

    Internal logic:
    - Pulls fields in a fixed order to allow fast set membership checks.
    """
    return tuple(row.get(k) for k in PRIMARY_KEY_FIELDS)


def _existing_key_set(df) -> set[tuple]:
    """Create a set of primary-key tuples from a results DataFrame.

    Inputs:
    - df: pandas DataFrame (possibly empty).

    Outputs:
    - set of primary-key tuples.

    Internal logic:
    - Iterates rows and extracts PRIMARY_KEY_FIELDS into tuples.
    """
    if df is None or df.empty:
        return set()
    keys = set()
    for _, row in df.iterrows():
        keys.add(tuple(row.get(k) for k in PRIMARY_KEY_FIELDS))
    return keys


def _run_group(
    subject: int,
    seed: int,
    r: float,
    cfg: dict,
    stage: str,
    alpha_star: dict[str, float],
    compute_distance: bool,
    alpha_search_cfg: dict,
    existing_keys: set[tuple],
) -> list[dict]:
    """Run all grid rows for a (subject, seed, r) group.

    Inputs:
    - subject/seed/r: group identifiers.
    - cfg: config dict containing dataset/preprocess/split/models/generators/qc/experiment.
    - stage: alpha_search | final_eval | full.
    - alpha_star: mapping of r -> alpha_ratio.
    - compute_distance: whether to compute distance metrics.
    - alpha_search_cfg: alpha-search proxy settings.
    - existing_keys: set of primary-key tuples to skip.

    Outputs:
    - list of result row dicts for this group.

    Internal logic:
    - Executes the same nested loop as main(), but scoped to one group.
    """
    rows: list[dict] = []
    exp_cfg = cfg["experiment"]
    methods = exp_cfg.get("methods", [])
    classifiers = exp_cfg.get("classifiers", [])
    generators = exp_cfg.get("generators", [])
    alpha_list = [float(x) for x in exp_cfg.get("alpha_ratio_list", [0.0])]
    qc_on_list = [bool(x) for x in exp_cfg.get("qc_on", [False])]
    screening_classifier = exp_cfg.get("stage", {}).get("screening_classifier", "eegnet")

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
                if _row_key_tuple(row_key) in existing_keys:
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
                    results_path="results/results.csv",
                    stage=stage,
                    compute_distance=compute_distance,
                    alpha_search_cfg=alpha_search_cfg,
                )
                rows.append(row)
                continue

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
                        if _row_key_tuple(row_key) in existing_keys:
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
                            results_path="results/results.csv",
                            stage=stage,
                            compute_distance=compute_distance,
                            alpha_search_cfg=alpha_search_cfg,
                        )
                        rows.append(row)

    return rows


def main() -> None:
    """Run the experiment grid with stage control and resume-safe behavior.

    Inputs:
    - CLI args: stage, results path, overrides, n_jobs.

    Outputs:
    - Appends rows to results.csv and writes run artifacts.

    Internal logic:
    - Iterates subjects/seeds/r/method/classifier/alpha/qc, skipping existing rows.
    - Stage alpha_search uses screening classifier and full alpha grid.
    - Stage final_eval uses alpha*(r) from alpha_star.json.
    - When n_jobs > 1, executes groups in parallel with spawn context.
    """
    parser = argparse.ArgumentParser(description="Run experiment grid")
    parser.add_argument("--stage", type=str, default=None, choices=["alpha_search", "final_eval", "full"])
    parser.add_argument("--results", type=str, default="results/results.csv")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--n_jobs", type=int, default=1)
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
    existing_keys = _existing_key_set(results_df)
    alpha_star = _load_alpha_star(alpha_star_path)

    subjects = cfg["dataset"]["subjects"]
    seeds = cfg["split"]["seeds"]
    r_list = [float(r) for r in cfg["split"]["low_data_fracs"]]

    compute_distance = stage != "alpha_search"

    groups = [(subject, seed, r) for subject in subjects for seed in seeds for r in r_list]

    if int(args.n_jobs) <= 1:
        for subject, seed, r in groups:
            rows = _run_group(
                subject=subject,
                seed=seed,
                r=r,
                cfg=cfg,
                stage=stage,
                alpha_star=alpha_star,
                compute_distance=compute_distance,
                alpha_search_cfg=alpha_search_cfg,
                existing_keys=existing_keys,
            )
            for row in rows:
                appended = append_result(args.results, row)
                if appended:
                    existing_keys.add(_row_key_tuple(row))
        return

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(args.n_jobs), mp_context=ctx) as ex:
        futures = {
            ex.submit(
                _run_group,
                subject,
                seed,
                r,
                cfg,
                stage,
                alpha_star,
                compute_distance,
                alpha_search_cfg,
                existing_keys,
            ): (subject, seed, r)
            for subject, seed, r in groups
        }
        for fut in as_completed(futures):
            rows = fut.result()
            for row in rows:
                appended = append_result(args.results, row)
                if appended:
                    existing_keys.add(_row_key_tuple(row))


if __name__ == "__main__":
    main()
