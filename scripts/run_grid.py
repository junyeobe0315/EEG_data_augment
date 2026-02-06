from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.pipeline import run_experiment
from src.utils.config import load_yaml
from src.utils.config_pack import load_yaml_with_pack
from src.utils.logging import get_logger
from src.utils.results import append_result, has_primary_key, load_results, PRIMARY_KEY_FIELDS


def _load_all_configs(overrides: list[str], config_pack: str = "base") -> dict:
    """Load dataset, model, generator, and experiment configs.

    Inputs:
    - overrides: list of key=value overrides.

    Outputs:
    - dict with dataset/preprocess/split/qc/experiment/models/generators configs.

    Internal logic:
    - Loads each YAML file once and merges overrides into each config.
    """
    cfg = {
        "dataset": load_yaml_with_pack("configs/dataset_bci2a.yaml", config_pack=config_pack, overrides=overrides),
        "preprocess": load_yaml_with_pack("configs/preprocess.yaml", config_pack=config_pack, overrides=overrides),
        "split": load_yaml_with_pack("configs/split.yaml", config_pack=config_pack, overrides=overrides),
        "qc": load_yaml_with_pack("configs/qc.yaml", config_pack=config_pack, overrides=overrides),
        "experiment": load_yaml("configs/experiment_grid.yaml", overrides=overrides),
        "models": {
            "eegnet": load_yaml_with_pack("configs/models/eegnet.yaml", config_pack=config_pack, overrides=overrides),
            "eegconformer": load_yaml_with_pack("configs/models/eegconformer.yaml", config_pack=config_pack, overrides=overrides),
            "ctnet": load_yaml_with_pack("configs/models/ctnet.yaml", config_pack=config_pack, overrides=overrides),
            "svm": load_yaml_with_pack("configs/models/fbcsp_svm.yaml", config_pack=config_pack, overrides=overrides),
        },
        "generators": {
            "cwgan_gp": load_yaml_with_pack("configs/generators/cwgan_gp.yaml", config_pack=config_pack, overrides=overrides),
            "cvae": load_yaml_with_pack("configs/generators/cvae.yaml", config_pack=config_pack, overrides=overrides),
            "ddpm": load_yaml_with_pack("configs/generators/ddpm.yaml", config_pack=config_pack, overrides=overrides),
        },
        "config_pack": str(config_pack),
    }
    return cfg


def _load_alpha_star(path: str | Path) -> dict[float, float]:
    """Load alpha* values from JSON if present.

    Inputs:
    - path: file path to alpha_star.json.

    Outputs:
    - dict mapping r(float) -> alpha_ratio(float).

    Internal logic:
    - Returns empty dict when file is missing to avoid hard failures.
    """
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[float, float] = {}
    for k, v in raw.items():
        out[float(k)] = float(v)
    return out


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


def _log_progress(
    groups_done: int,
    groups_total: int,
    rows_appended: int,
    rows_attempted: int,
    last_group: tuple[int, int, float],
    start_time: float,
) -> None:
    """Print a concise progress line for the grid run.

    Inputs:
    - groups_done/groups_total: completed groups and total groups.
    - rows_appended/rows_attempted: rows actually appended and attempted.
    - last_group: (subject, seed, r) for the last completed group.
    - start_time: timestamp when the run started.

    Outputs:
    - None (prints a single progress line to stdout).

    Internal logic:
    - Computes elapsed time and prints a one-line summary without verbosity.
    """
    elapsed = max(1.0, time.time() - start_time)
    rate = rows_appended / elapsed
    subject, seed, r = last_group
    print(
        f"[progress] groups {groups_done}/{groups_total} | "
        f"rows {rows_appended}/{rows_attempted} | "
        f"last=(S{subject:02d}, seed={seed}, r={r}) | "
        f"elapsed={elapsed:.0f}s | {rate:.2f} rows/s",
        flush=True,
    )


def _safe_float(v: Any) -> float | None:
    """Best-effort float cast used by row logging helpers."""
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row_metric_for_log(row: dict) -> tuple[str, float | None]:
    """Pick the most relevant metric field for concise row logs."""
    for key in ("kappa", "val_kappa", "macro_f1", "val_macro_f1", "acc", "val_acc"):
        value = _safe_float(row.get(key))
        if value is not None:
            return key, value
    return "metric", None


def _format_row_log(row: dict, appended: bool) -> str:
    """Build a compact one-line row log."""
    metric_name, metric_value = _row_metric_for_log(row)
    metric_txt = f"{metric_name}={metric_value:.4f}" if metric_value is not None else f"{metric_name}=NA"
    runtime = _safe_float(row.get("runtime_sec"))
    runtime_txt = f"runtime={runtime:.1f}s" if runtime is not None else "runtime=NA"
    state = "append" if appended else "skip"
    return (
        f"[row:{state}] S{int(row.get('subject')):02d} seed={row.get('seed')} r={row.get('r')} "
        f"{row.get('method')}/{row.get('classifier')} gen={row.get('generator')} "
        f"alpha={row.get('alpha_ratio')} qc={row.get('qc_on')} {metric_txt} {runtime_txt}"
    )


def _run_group(
    subject: int,
    seed: int,
    r: float,
    cfg: dict,
    stage: str,
    alpha_star: dict[float, float],
    compute_distance: bool,
    alpha_search_cfg: dict,
    existing_keys: set[tuple],
    include_baselines_in_alpha_search: bool,
    config_pack: str,
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
        if stage == "alpha_search" and method != "GenAug" and not include_baselines_in_alpha_search:
            continue

        if stage == "alpha_search":
            cls_list = [screening_classifier]
        else:
            cls_list = classifiers

        for classifier in cls_list:
            if method != "GenAug":
                row_key = {
                    "config_pack": str(config_pack),
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
                    config_pack=str(config_pack),
                )
                rows.append(row)
                continue

            if stage == "alpha_search":
                alpha_candidates = alpha_list
            elif stage == "final_eval":
                if r not in alpha_star:
                    raise KeyError(
                        f"alpha* missing for r={r}. Run select_alpha.py first and ensure alpha_star.json covers all r."
                    )
                alpha_candidates = [float(alpha_star[r])]
            else:
                alpha_candidates = alpha_list

            for generator in generators:
                for alpha_ratio in alpha_candidates:
                    for qc_on in qc_on_list:
                        row_key = {
                            "config_pack": str(config_pack),
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
                            config_pack=str(config_pack),
                        )
                        rows.append(row)

    return rows


def main() -> None:
    """Run the experiment grid with stage control and resume-safe behavior.

    Inputs:
    - CLI args: stage, results path, overrides, n_jobs, log_every_groups.

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
    parser.add_argument("--config_pack", type=str, default="base")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--log_every_groups", type=int, default=5)
    parser.add_argument("--quiet_rows", action="store_true")
    args = parser.parse_args()
    logger = get_logger("run_grid")

    cfg = _load_all_configs(args.override, config_pack=args.config_pack)
    exp_cfg = cfg["experiment"]  # grid + stage settings
    stage = args.stage or exp_cfg.get("stage", {}).get("mode", "full")  # alpha_search/final_eval/full
    screening_classifier = exp_cfg.get("stage", {}).get("screening_classifier", "eegnet")  # EEGNet by default
    alpha_star_path = exp_cfg.get("stage", {}).get("alpha_star_path", "./artifacts/alpha_star.json")  # alpha* cache
    alpha_search_cfg = exp_cfg.get("alpha_search", {})  # proxy mode config
    include_baselines_in_alpha_search = bool(
        exp_cfg.get("stage", {}).get("alpha_search_include_baselines", False)
    )

    methods = exp_cfg.get("methods", [])  # C0/C1/C2/GenAug
    classifiers = exp_cfg.get("classifiers", [])  # model list
    generators = exp_cfg.get("generators", [])  # generator list
    alpha_list = [float(x) for x in exp_cfg.get("alpha_ratio_list", [0.0])]  # synth ratios
    qc_on_list = [bool(x) for x in exp_cfg.get("qc_on", [False])]  # QC ablation flags
    log_rows = not bool(args.quiet_rows)

    logger.info(
        "grid start stage=%s config_pack=%s n_jobs=%d results=%s methods=%s classifiers=%s generators=%s alpha=%s qc=%s",
        stage,
        args.config_pack,
        int(args.n_jobs),
        args.results,
        methods,
        classifiers,
        generators,
        alpha_list,
        qc_on_list,
    )

    results_df = load_results(args.results)
    existing_keys = _existing_key_set(results_df)
    alpha_star = _load_alpha_star(alpha_star_path)

    subjects = cfg["dataset"]["subjects"]
    seeds = cfg["split"]["seeds"]
    r_list = [float(r) for r in cfg["split"]["low_data_fracs"]]
    if stage == "final_eval":
        missing_r = [r for r in r_list if r not in alpha_star]
        if missing_r:
            missing_txt = ", ".join(str(r) for r in missing_r)
            raise RuntimeError(
                f"Missing alpha* for r values: {missing_txt}. "
                "Run scripts/select_alpha.py to generate a complete alpha_star.json."
            )

    compute_distance = stage != "alpha_search"

    groups = [(subject, seed, r) for subject in subjects for seed in seeds for r in r_list]
    groups_total = len(groups)
    groups_done = 0
    rows_appended = 0
    rows_attempted = 0
    start_time = time.time()
    log_every = max(1, int(args.log_every_groups))

    if int(args.n_jobs) <= 1:
        for subject, seed, r in groups:
            logger.info("group start S%02d seed=%d r=%s", subject, seed, r)
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
                include_baselines_in_alpha_search=include_baselines_in_alpha_search,
                config_pack=str(args.config_pack),
            )
            for row in rows:
                appended = append_result(args.results, row)
                if appended:
                    existing_keys.add(_row_key_tuple(row))
                    rows_appended += 1
                rows_attempted += 1
                if log_rows:
                    logger.info("%s", _format_row_log(row, appended=bool(appended)))
            groups_done += 1
            if groups_done % log_every == 0 or groups_done == groups_total:
                _log_progress(groups_done, groups_total, rows_appended, rows_attempted, (subject, seed, r), start_time)
            if not rows:
                logger.info("group skip S%02d seed=%d r=%s (all rows already present)", subject, seed, r)
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
                include_baselines_in_alpha_search,
                str(args.config_pack),
            ): (subject, seed, r)
            for subject, seed, r in groups
        }
        for fut in as_completed(futures):
            subject, seed, r = futures[fut]
            rows = fut.result()
            logger.info("group done S%02d seed=%d r=%s produced_rows=%d", subject, seed, r, len(rows))
            for row in rows:
                appended = append_result(args.results, row)
                if appended:
                    existing_keys.add(_row_key_tuple(row))
                    rows_appended += 1
                rows_attempted += 1
                if log_rows:
                    logger.info("%s", _format_row_log(row, appended=bool(appended)))
            groups_done += 1
            if groups_done % log_every == 0 or groups_done == groups_total:
                _log_progress(groups_done, groups_total, rows_appended, rows_attempted, (subject, seed, r), start_time)
    logger.info(
        "grid end stage=%s groups=%d appended=%d attempted=%d elapsed=%.1fs",
        stage,
        groups_done,
        rows_appended,
        rows_attempted,
        time.time() - start_time,
    )


if __name__ == "__main__":
    main()
