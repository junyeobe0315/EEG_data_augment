from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.pipeline import run_experiment
from src.utils.config import load_yaml, config_hash
from src.utils.config_pack import load_yaml_with_pack
from src.utils.io import ensure_dir, write_json
from src.utils.logging import get_logger
from src.utils.resource import get_git_commit
from src.utils.seed import set_global_seed, stable_hash_seed
from src.utils.tuning_results import append_tuning_trial


def _load_all_configs(overrides: list[str], config_pack: str = "base") -> dict:
    """Load runtime configs with config-pack fallback."""
    return {
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
    }


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = [k for k in str(dotted).split(".") if k]
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_dotted_params(cfg: dict, params: dict[str, Any]) -> dict:
    """Return a deep-copied config with dotted params applied."""
    out = copy.deepcopy(cfg)
    for k, v in params.items():
        _set_dotted(out, k, v)
    return out


def sample_from_space(space_cfg: dict[str, dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    """Sample one parameter set from a search-space spec."""
    params: dict[str, Any] = {}
    for dotted, spec in space_cfg.items():
        kind = str(spec.get("type", "choice")).lower()
        if kind == "choice":
            values = list(spec.get("values", []))
            if not values:
                raise ValueError(f"Empty choice values for {dotted}")
            idx = int(rng.integers(0, len(values)))
            params[dotted] = values[idx]
        elif kind == "log_uniform":
            low = float(spec["low"])
            high = float(spec["high"])
            if low <= 0 or high <= 0 or high < low:
                raise ValueError(f"Invalid log_uniform bounds for {dotted}: {low}, {high}")
            params[dotted] = float(math.exp(rng.uniform(math.log(low), math.log(high))))
        else:
            raise ValueError(f"Unsupported search-space type: {kind}")
    return params


@dataclass(frozen=True)
class Combo:
    subject: int
    seed: int
    r: float


def _build_combos(subjects: list[int], seeds: list[int], r_list: list[float]) -> list[Combo]:
    return [Combo(int(su), int(se), float(r)) for su in subjects for se in seeds for r in r_list]


def _mean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) > 0 else float("nan")


def _sort_trials(rows: list[dict]) -> list[dict]:
    def _score(x: dict) -> tuple[float, float]:
        obj = x.get("objective_value", np.nan)
        tie = x.get("tie_break_value", np.nan)
        obj_v = float(obj) if np.isfinite(obj) else float("-inf")
        tie_v = float(tie) if np.isfinite(tie) else float("-inf")
        return (obj_v, tie_v)

    return sorted(rows, key=_score, reverse=True)


def _trial_common(
    target: str,
    phase: str,
    trial_id: str,
    trial_seed: int,
    metric: str,
    params: dict[str, Any],
    start_time: float,
    objective_value: float,
    tie_break_value: float,
    n_eval: int,
    status: str,
    reason: str,
) -> dict[str, Any]:
    payload = {
        "target": target,
        "phase": phase,
        "trial_id": trial_id,
        "seed": int(trial_seed),
        "params": params,
    }
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "trial_id": trial_id,
        "target": target,
        "phase": phase,
        "status": status,
        "reason": reason,
        "config_hash": config_hash(payload),
        "run_id": config_hash({"run": payload, "metric": metric}),
        "seed": int(trial_seed),
        "objective_metric": metric,
        "objective_value": objective_value,
        "tie_break_value": tie_break_value,
        "n_eval": int(n_eval),
        "runtime_sec": float(time.time() - start_time),
        "params_json": json.dumps(params, ensure_ascii=True, sort_keys=True),
    }


def _eval_eegnet_trial(
    trial_id: str,
    trial_seed: int,
    phase: str,
    combos: list[Combo],
    metric: str,
    base_cfg: dict,
    eegnet_params: dict[str, Any],
    run_root: Path,
    config_pack: str,
) -> tuple[dict, dict]:
    start_time = time.time()
    model_cfgs = copy.deepcopy(base_cfg["models"])
    model_cfgs["eegnet"] = apply_dotted_params(model_cfgs["eegnet"], eegnet_params)
    alpha_search_cfg = base_cfg["experiment"].get("alpha_search", {})

    vals = []
    vals_low_r = []
    status = "ok"
    reason = ""
    try:
        for combo in combos:
            row = run_experiment(
                subject=combo.subject,
                seed=combo.seed,
                r=combo.r,
                method="C0",
                classifier="eegnet",
                generator="none",
                alpha_ratio=0.0,
                qc_on=False,
                dataset_cfg=base_cfg["dataset"],
                preprocess_cfg=base_cfg["preprocess"],
                split_cfg=base_cfg["split"],
                model_cfgs=model_cfgs,
                gen_cfgs=base_cfg["generators"],
                qc_cfg=base_cfg["qc"],
                results_path="results/results.csv",
                run_root=run_root / "runs",
                stage="alpha_search",
                compute_distance=False,
                alpha_search_cfg=alpha_search_cfg,
                config_pack=f"{config_pack}_tuning_{trial_id}",
            )
            v = float(row.get(metric, np.nan))
            if np.isfinite(v):
                vals.append(v)
                if combo.r <= 0.2:
                    vals_low_r.append(v)
    except Exception as e:
        status = "failed"
        reason = str(e)

    obj = _mean_or_nan(vals)
    tie = _mean_or_nan(vals_low_r)
    if status == "ok" and not np.isfinite(obj):
        status = "invalid"
        reason = f"{metric} is NaN"

    rec = _trial_common(
        target="eegnet",
        phase=phase,
        trial_id=trial_id,
        trial_seed=trial_seed,
        metric=metric,
        params=eegnet_params,
        start_time=start_time,
        objective_value=obj,
        tie_break_value=tie,
        n_eval=len(vals),
        status=status,
        reason=reason,
    )
    return rec, model_cfgs["eegnet"]


def _build_c0_cache(
    combos: list[Combo],
    metric: str,
    base_cfg: dict,
    eegnet_cfg: dict,
    run_root: Path,
    config_pack: str,
    cache_name: str,
) -> dict[Combo, float]:
    cache: dict[Combo, float] = {}
    model_cfgs = copy.deepcopy(base_cfg["models"])
    model_cfgs["eegnet"] = copy.deepcopy(eegnet_cfg)
    alpha_search_cfg = base_cfg["experiment"].get("alpha_search", {})

    for combo in combos:
        row = run_experiment(
            subject=combo.subject,
            seed=combo.seed,
            r=combo.r,
            method="C0",
            classifier="eegnet",
            generator="none",
            alpha_ratio=0.0,
            qc_on=False,
            dataset_cfg=base_cfg["dataset"],
            preprocess_cfg=base_cfg["preprocess"],
            split_cfg=base_cfg["split"],
            model_cfgs=model_cfgs,
            gen_cfgs=base_cfg["generators"],
            qc_cfg=base_cfg["qc"],
            results_path="results/results.csv",
            run_root=run_root / "runs" / cache_name,
            stage="alpha_search",
            compute_distance=False,
            alpha_search_cfg=alpha_search_cfg,
            config_pack=f"{config_pack}_{cache_name}",
        )
        cache[combo] = float(row.get(metric, np.nan))
    return cache


def _apply_generator_epoch_policy(gen_cfg: dict, tuning_cfg: dict, generator_type: str) -> dict:
    """Apply tuning-time epoch policy (max epochs + early stopping) to generator cfg."""
    out = copy.deepcopy(gen_cfg)
    policy = tuning_cfg.get("generator_epoch_policy", {})
    if not bool(policy.get("enabled", False)):
        return out

    train_cfg = out.setdefault("train", {})
    max_epochs_cfg = policy.get("max_epochs", {})
    default_max = 10000 if generator_type in {"cwgan_gp", "cvae"} else int(train_cfg.get("epochs", 200))
    target_epochs = int(max_epochs_cfg.get(generator_type, default_max))
    train_cfg["epochs"] = max(1, target_epochs)

    if generator_type == "ddpm":
        ddpm_cap = policy.get("ddpm_max_epochs", max_epochs_cfg.get("ddpm"))
        if ddpm_cap is not None:
            train_cfg["epochs"] = min(int(train_cfg["epochs"]), int(ddpm_cap))
        if bool(policy.get("disable_ddpm_early_stopping", True)):
            es = copy.deepcopy(train_cfg.get("early_stopping", {}))
            es["enabled"] = False
            train_cfg["early_stopping"] = es
        return out

    es_policy = policy.get("early_stopping", {})
    es_cfg = copy.deepcopy(train_cfg.get("early_stopping", {}))
    es_cfg["enabled"] = bool(es_policy.get("enabled", True))
    es_cfg["patience_epochs"] = int(es_policy.get("patience_epochs", 500))
    es_cfg["min_delta"] = float(es_policy.get("min_delta", 0.0))
    es_cfg["mode"] = str(es_policy.get("mode", "min"))
    monitor_cfg = es_policy.get("monitor", {})
    if isinstance(monitor_cfg, dict):
        es_cfg["monitor"] = str(monitor_cfg.get(generator_type, "loss"))
    else:
        es_cfg["monitor"] = str(monitor_cfg) if monitor_cfg else "loss"
    train_cfg["early_stopping"] = es_cfg

    if "save_every" in policy:
        train_cfg["save_every"] = int(policy["save_every"])
    return out


def _eval_genaug_trial(
    trial_id: str,
    trial_seed: int,
    phase: str,
    combos: list[Combo],
    metric: str,
    alpha_ratio_ref: float,
    base_cfg: dict,
    eegnet_cfg: dict,
    generator_type: str,
    gen_params: dict[str, Any],
    qc_params: dict[str, Any],
    tuning_cfg: dict,
    c0_cache: dict[Combo, float],
    run_root: Path,
    config_pack: str,
) -> tuple[dict, dict, dict]:
    start_time = time.time()
    model_cfgs = copy.deepcopy(base_cfg["models"])
    model_cfgs["eegnet"] = copy.deepcopy(eegnet_cfg)
    gen_cfgs = copy.deepcopy(base_cfg["generators"])
    gen_cfgs[generator_type] = apply_dotted_params(gen_cfgs[generator_type], gen_params)
    gen_cfgs[generator_type] = _apply_generator_epoch_policy(gen_cfgs[generator_type], tuning_cfg, generator_type)
    qc_cfg = apply_dotted_params(base_cfg["qc"], qc_params)
    alpha_search_cfg = base_cfg["experiment"].get("alpha_search", {})

    gains = []
    low_r_gen_vals = []
    status = "ok"
    reason = ""
    try:
        for combo in combos:
            base_v = float(c0_cache.get(combo, np.nan))
            row = run_experiment(
                subject=combo.subject,
                seed=combo.seed,
                r=combo.r,
                method="GenAug",
                classifier="eegnet",
                generator=generator_type,
                alpha_ratio=float(alpha_ratio_ref),
                qc_on=True,
                dataset_cfg=base_cfg["dataset"],
                preprocess_cfg=base_cfg["preprocess"],
                split_cfg=base_cfg["split"],
                model_cfgs=model_cfgs,
                gen_cfgs=gen_cfgs,
                qc_cfg=qc_cfg,
                results_path="results/results.csv",
                run_root=run_root / "runs",
                stage="alpha_search",
                compute_distance=False,
                alpha_search_cfg=alpha_search_cfg,
                config_pack=f"{config_pack}_tuning_{trial_id}",
            )
            gen_v = float(row.get(metric, np.nan))
            if np.isfinite(base_v) and np.isfinite(gen_v):
                gains.append(float(gen_v - base_v))
                if combo.r <= 0.2:
                    low_r_gen_vals.append(gen_v)
    except Exception as e:
        status = "failed"
        reason = str(e)

    obj = _mean_or_nan(gains)
    tie = _mean_or_nan(low_r_gen_vals)
    if status == "ok" and not np.isfinite(obj):
        status = "invalid"
        reason = "gain is NaN"

    params_all = {"generator_type": generator_type, "generator": gen_params, "qc": qc_params}
    rec = _trial_common(
        target="genaug_qc",
        phase=phase,
        trial_id=trial_id,
        trial_seed=trial_seed,
        metric=f"{metric}_gain",
        params=params_all,
        start_time=start_time,
        objective_value=obj,
        tie_break_value=tie,
        n_eval=len(gains),
        status=status,
        reason=reason,
    )
    return rec, gen_cfgs[generator_type], qc_cfg


def _validate_inputs(cfg: dict, tuning_cfg: dict) -> None:
    index_path = Path(cfg["dataset"]["index_path"])
    if not index_path.exists():
        raise FileNotFoundError(f"Missing preprocessed index: {index_path}")
    split_root = Path("./artifacts/splits") / cfg["dataset"]["name"]
    if not split_root.exists():
        raise FileNotFoundError(f"Missing split directory: {split_root}")
    if len(tuning_cfg.get("pilot", {}).get("subjects", [])) == 0:
        raise ValueError("tuning.pilot.subjects must be non-empty")


def _log_trial(logger, rec: dict) -> None:
    """Print a concise trial record summary."""
    obj = rec.get("objective_value", np.nan)
    tie = rec.get("tie_break_value", np.nan)
    obj_txt = f"{float(obj):.4f}" if np.isfinite(obj) else "nan"
    tie_txt = f"{float(tie):.4f}" if np.isfinite(tie) else "nan"
    logger.info(
        "[trial] %s %s id=%s status=%s obj=%s tie=%s n_eval=%s runtime=%.1fs",
        rec.get("target"),
        rec.get("phase"),
        rec.get("trial_id"),
        rec.get("status"),
        obj_txt,
        tie_txt,
        rec.get("n_eval"),
        float(rec.get("runtime_sec", 0.0)),
    )
    reason = str(rec.get("reason", "")).strip()
    if reason:
        logger.warning("[trial] id=%s reason=%s", rec.get("trial_id"), reason)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """Return list with duplicates removed while preserving input order."""
    out: list[str] = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def _resolve_generator_targets(args, tuning_cfg: dict, cfg: dict) -> list[str]:
    """Resolve generator tuning sequence from CLI or tuning config."""
    if args.gen_sequence:
        targets = _dedupe_keep_order([str(x) for x in args.gen_sequence])
    else:
        genaug_cfg = tuning_cfg.get("genaug", {})
        targets = [str(genaug_cfg.get("generator", "cwgan_gp"))]
    missing = [g for g in targets if g not in cfg.get("generators", {})]
    if missing:
        raise KeyError(f"Unknown tuning generator(s): {missing}")
    return targets


def _run_trial_jobs(
    eval_fn,
    jobs: list[dict[str, Any]],
    n_jobs: int,
) -> list[tuple[dict[str, Any], Any]]:
    """Run trial evaluation jobs either sequentially or in parallel.

    Outputs:
    - list of tuples: (job_dict, eval_fn_output)
    """
    if int(n_jobs) <= 1:
        out: list[tuple[dict[str, Any], Any]] = []
        for job in jobs:
            out.append((job, eval_fn(**job)))
        return out

    ctx = mp.get_context("spawn")
    out: list[tuple[dict[str, Any], Any]] = []
    with ProcessPoolExecutor(max_workers=int(n_jobs), mp_context=ctx) as ex:
        fut_to_job = {ex.submit(eval_fn, **job): job for job in jobs}
        for fut in as_completed(fut_to_job):
            out.append((fut_to_job[fut], fut.result()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune EEGNet and cWGAN/QC hyperparameters with random search")
    parser.add_argument("--config_pack", type=str, default="base")
    parser.add_argument("--tuning_cfg", type=str, default="configs/tuning.yaml")
    parser.add_argument("--space_cfg", type=str, default="configs/hparam_space.yaml")
    parser.add_argument("--gen_sequence", type=str, nargs="+", choices=["cwgan_gp", "cvae", "ddpm"], default=None)
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    logger = get_logger("tune_hparams")

    cfg = _load_all_configs(args.override, config_pack=args.config_pack)
    tuning_cfg = load_yaml(args.tuning_cfg, overrides=args.override)
    space_cfg = load_yaml(args.space_cfg, overrides=args.override)
    _validate_inputs(cfg, tuning_cfg)

    metric = str(tuning_cfg.get("objective_metric", "val_kappa"))
    base_seed = int(tuning_cfg.get("seed", 2026))
    set_global_seed(base_seed)

    pilot_subjects = [int(x) for x in tuning_cfg["pilot"]["subjects"]]
    pilot_seeds = [int(x) for x in tuning_cfg["pilot"]["seeds"]]
    pilot_r = [float(x) for x in tuning_cfg["pilot"]["r_list"]]

    screen_seeds = [int(x) for x in tuning_cfg["screening"]["seeds"]]
    screen_r = [float(x) for x in tuning_cfg["screening"]["r_list"]]

    budget_cfg = tuning_cfg.get("budget", {})
    eegnet_trials = int(budget_cfg.get("eegnet_trials", 18))
    gen_trials = int(budget_cfg.get("genaug_trials", 18))
    top_k = int(budget_cfg.get("top_k", 4))
    if args.max_trials is not None:
        eegnet_trials = min(eegnet_trials, int(args.max_trials))
        gen_trials = min(gen_trials, int(args.max_trials))

    genaug_cfg = tuning_cfg.get("genaug", {})
    alpha_ratio_ref = float(genaug_cfg.get("alpha_ratio_ref", 1.0))
    gen_targets = _resolve_generator_targets(args, tuning_cfg, cfg)
    paths_cfg = tuning_cfg.get("paths", {})
    tuning_root = ensure_dir(paths_cfg.get("tuning_root", "./artifacts/tuning"))
    trials_csv = Path(paths_cfg.get("trials_csv", "./results/tuning_trials.csv"))
    best_json = Path(paths_cfg.get("best_params_json", "./artifacts/tuning/best_params.json"))

    screen_combos = _build_combos(pilot_subjects, screen_seeds, screen_r)
    full_combos = _build_combos(pilot_subjects, pilot_seeds, pilot_r)
    logger.info(
        "tuning start base_pack=%s metric=%s seed=%d eeg_trials=%d gen_trials=%d top_k=%d n_jobs=%d",
        args.config_pack,
        metric,
        base_seed,
        eegnet_trials,
        gen_trials,
        top_k,
        int(args.n_jobs),
    )
    logger.info(
        "pilot subjects=%s seeds=%s r=%s | screen seeds=%s r=%s | alpha_ref=%.3f gen_sequence=%s",
        pilot_subjects,
        pilot_seeds,
        pilot_r,
        screen_seeds,
        screen_r,
        alpha_ratio_ref,
        gen_targets,
    )
    logger.info("artifacts tuning_root=%s trials_csv=%s", tuning_root, trials_csv)

    # 1) EEGNet tuning
    eeg_space = space_cfg.get("eegnet", {})
    eeg_rows_screen: list[dict] = []
    eeg_trial_payload: dict[str, dict] = {}
    eeg_screen_jobs: list[dict[str, Any]] = []
    for t in range(eegnet_trials):
        trial_id = f"eegnet_screen_{t:03d}"
        trial_seed = stable_hash_seed(base_seed, {"target": "eegnet", "trial": t, "phase": "screen"})
        rng = np.random.default_rng(trial_seed)
        params = sample_from_space(eeg_space, rng)
        eeg_screen_jobs.append(
            {
                "trial_id": trial_id,
                "trial_seed": trial_seed,
                "phase": "screen",
                "combos": screen_combos,
                "metric": metric,
                "base_cfg": cfg,
                "eegnet_params": params,
                "run_root": tuning_root / "eegnet",
                "config_pack": args.config_pack,
            }
        )

    for job, (rec, tuned_eegnet_cfg) in _run_trial_jobs(_eval_eegnet_trial, eeg_screen_jobs, int(args.n_jobs)):
        params = job["eegnet_params"]
        trial_id = str(job["trial_id"])
        append_tuning_trial(trials_csv, rec)
        write_json(tuning_root / "eegnet" / "trials" / f"{trial_id}.json", {"record": rec, "params": params})
        eeg_rows_screen.append(rec)
        eeg_trial_payload[trial_id] = {"params": params, "cfg": tuned_eegnet_cfg}
        _log_trial(logger, rec)

    eeg_rows_screen_ok = [r for r in eeg_rows_screen if r.get("status") == "ok" and np.isfinite(r.get("objective_value", np.nan))]
    eeg_top_ids = [r["trial_id"] for r in _sort_trials(eeg_rows_screen_ok)[:top_k]]
    logger.info("eegnet screen done valid=%d/%d top_ids=%s", len(eeg_rows_screen_ok), len(eeg_rows_screen), eeg_top_ids)

    eeg_rows_confirm: list[dict] = []
    eeg_confirm_jobs: list[dict[str, Any]] = []
    for tid in eeg_top_ids:
        params = eeg_trial_payload[tid]["params"]
        trial_seed = stable_hash_seed(base_seed, {"target": "eegnet", "trial_id": tid, "phase": "confirm"})
        eeg_confirm_jobs.append(
            {
                "trial_id": f"{tid}_confirm",
                "trial_seed": trial_seed,
                "phase": "confirm",
                "combos": full_combos,
                "metric": metric,
                "base_cfg": cfg,
                "eegnet_params": params,
                "run_root": tuning_root / "eegnet",
                "config_pack": args.config_pack,
            }
        )

    for job, (rec, tuned_eegnet_cfg) in _run_trial_jobs(_eval_eegnet_trial, eeg_confirm_jobs, int(args.n_jobs)):
        trial_id = str(job["trial_id"])
        # confirmation trials reuse the source screen params
        src_id = trial_id.replace("_confirm", "")
        params = eeg_trial_payload[src_id]["params"]
        append_tuning_trial(trials_csv, rec)
        write_json(tuning_root / "eegnet" / "trials" / f"{trial_id}.json", {"record": rec, "params": params})
        eeg_rows_confirm.append(rec)
        eeg_trial_payload[trial_id] = {"params": params, "cfg": tuned_eegnet_cfg}
        _log_trial(logger, rec)

    eeg_candidates = [r for r in eeg_rows_confirm if r.get("status") == "ok" and np.isfinite(r.get("objective_value", np.nan))]
    if not eeg_candidates:
        eeg_candidates = eeg_rows_screen_ok
    if not eeg_candidates:
        raise RuntimeError("No valid EEGNet tuning trials.")
    eeg_best_row = _sort_trials(eeg_candidates)[0]
    eeg_best_params = json.loads(eeg_best_row["params_json"])
    eeg_best_cfg = apply_dotted_params(cfg["models"]["eegnet"], eeg_best_params)
    logger.info(
        "eegnet best trial=%s phase=%s obj=%.4f tie=%.4f",
        eeg_best_row.get("trial_id"),
        eeg_best_row.get("phase"),
        float(eeg_best_row.get("objective_value", np.nan)),
        float(eeg_best_row.get("tie_break_value", np.nan)),
    )


    # 2) GenAug+QC tuning (supports sequential generator families)
    qc_space = space_cfg.get("qc", {})

    c0_cache_screen = _build_c0_cache(
        combos=screen_combos,
        metric=metric,
        base_cfg=cfg,
        eegnet_cfg=eeg_best_cfg,
        run_root=tuning_root / "genaug",
        config_pack=args.config_pack,
        cache_name="c0_screen",
    )
    c0_cache_full = _build_c0_cache(
        combos=full_combos,
        metric=metric,
        base_cfg=cfg,
        eegnet_cfg=eeg_best_cfg,
        run_root=tuning_root / "genaug",
        config_pack=args.config_pack,
        cache_name="c0_full",
    )
    logger.info(
        "genaug baseline cache built screen=%d full=%d",
        len(c0_cache_screen),
        len(c0_cache_full),
    )

    gen_best_entries: list[dict[str, Any]] = []
    for gen_target in gen_targets:
        gen_space = space_cfg.get(gen_target, {})
        if not gen_space:
            raise KeyError(f"Search space for generator '{gen_target}' not found in {args.space_cfg}")
        logger.info("genaug tuning start generator=%s", gen_target)

        gen_rows_screen: list[dict] = []
        gen_trial_payload: dict[str, dict] = {}
        gen_screen_jobs: list[dict[str, Any]] = []
        for t in range(gen_trials):
            trial_id = f"{gen_target}_genaug_screen_{t:03d}"
            trial_seed = stable_hash_seed(
                base_seed, {"target": "genaug_qc", "generator": gen_target, "trial": t, "phase": "screen"}
            )
            rng = np.random.default_rng(trial_seed)
            gen_params = sample_from_space(gen_space, rng)
            qc_params = sample_from_space(qc_space, rng)
            gen_screen_jobs.append(
                {
                    "trial_id": trial_id,
                    "trial_seed": trial_seed,
                    "phase": "screen",
                    "combos": screen_combos,
                    "metric": metric,
                    "alpha_ratio_ref": alpha_ratio_ref,
                    "base_cfg": cfg,
                    "eegnet_cfg": eeg_best_cfg,
                    "generator_type": gen_target,
                    "gen_params": gen_params,
                    "qc_params": qc_params,
                    "tuning_cfg": tuning_cfg,
                    "c0_cache": c0_cache_screen,
                    "run_root": tuning_root / "genaug",
                    "config_pack": args.config_pack,
                }
            )

        for job, (rec, tuned_gen_cfg, tuned_qc_cfg) in _run_trial_jobs(
            _eval_genaug_trial, gen_screen_jobs, int(args.n_jobs)
        ):
            trial_id = str(job["trial_id"])
            gen_params = job["gen_params"]
            qc_params = job["qc_params"]
            append_tuning_trial(trials_csv, rec)
            write_json(
                tuning_root / "genaug" / "trials" / f"{trial_id}.json",
                {"record": rec, "params": {"generator": gen_params, "qc": qc_params}},
            )
            gen_rows_screen.append(rec)
            gen_trial_payload[trial_id] = {
                "gen_params": gen_params,
                "qc_params": qc_params,
                "gen_cfg": tuned_gen_cfg,
                "qc_cfg": tuned_qc_cfg,
            }
            _log_trial(logger, rec)

        gen_rows_screen_ok = [
            r for r in gen_rows_screen if r.get("status") == "ok" and np.isfinite(r.get("objective_value", np.nan))
        ]
        gen_top_ids = [r["trial_id"] for r in _sort_trials(gen_rows_screen_ok)[:top_k]]
        logger.info(
            "genaug screen done generator=%s valid=%d/%d top_ids=%s",
            gen_target,
            len(gen_rows_screen_ok),
            len(gen_rows_screen),
            gen_top_ids,
        )

        gen_rows_confirm: list[dict] = []
        gen_confirm_jobs: list[dict[str, Any]] = []
        for tid in gen_top_ids:
            payload = gen_trial_payload[tid]
            trial_seed = stable_hash_seed(
                base_seed, {"target": "genaug_qc", "generator": gen_target, "trial_id": tid, "phase": "confirm"}
            )
            gen_confirm_jobs.append(
                {
                    "trial_id": f"{tid}_confirm",
                    "trial_seed": trial_seed,
                    "phase": "confirm",
                    "combos": full_combos,
                    "metric": metric,
                    "alpha_ratio_ref": alpha_ratio_ref,
                    "base_cfg": cfg,
                    "eegnet_cfg": eeg_best_cfg,
                    "generator_type": gen_target,
                    "gen_params": payload["gen_params"],
                    "qc_params": payload["qc_params"],
                    "tuning_cfg": tuning_cfg,
                    "c0_cache": c0_cache_full,
                    "run_root": tuning_root / "genaug",
                    "config_pack": args.config_pack,
                }
            )

        for job, (rec, tuned_gen_cfg, tuned_qc_cfg) in _run_trial_jobs(
            _eval_genaug_trial, gen_confirm_jobs, int(args.n_jobs)
        ):
            trial_id = str(job["trial_id"])
            gen_params = job["gen_params"]
            qc_params = job["qc_params"]
            append_tuning_trial(trials_csv, rec)
            write_json(
                tuning_root / "genaug" / "trials" / f"{trial_id}.json",
                {"record": rec, "params": {"generator": gen_params, "qc": qc_params}},
            )
            gen_rows_confirm.append(rec)
            gen_trial_payload[trial_id] = {
                "gen_params": gen_params,
                "qc_params": qc_params,
                "gen_cfg": tuned_gen_cfg,
                "qc_cfg": tuned_qc_cfg,
            }
            _log_trial(logger, rec)

        gen_candidates = [
            r for r in gen_rows_confirm if r.get("status") == "ok" and np.isfinite(r.get("objective_value", np.nan))
        ]
        if not gen_candidates:
            gen_candidates = gen_rows_screen_ok
        if not gen_candidates:
            raise RuntimeError(f"No valid GenAug/QC tuning trials for generator={gen_target}.")

        gen_best_row = _sort_trials(gen_candidates)[0]
        gen_best_payload = json.loads(gen_best_row["params_json"])
        logger.info(
            "genaug/qc best generator=%s trial=%s phase=%s obj=%.4f tie=%.4f",
            gen_target,
            gen_best_row.get("trial_id"),
            gen_best_row.get("phase"),
            float(gen_best_row.get("objective_value", np.nan)),
            float(gen_best_row.get("tie_break_value", np.nan)),
        )
        gen_best_entries.append(
            {
                "generator": gen_target,
                "row": gen_best_row,
                "payload": gen_best_payload,
            }
        )

    if not gen_best_entries:
        raise RuntimeError("No valid GenAug/QC tuning results.")
    gen_best_row = _sort_trials([e["row"] for e in gen_best_entries])[0]
    gen_best_entry = next(e for e in gen_best_entries if e["row"]["trial_id"] == gen_best_row["trial_id"])
    gen_best_payload = gen_best_entry["payload"]
    gen_best_generator = str(gen_best_entry["generator"])
    logger.info(
        "genaug overall best generator=%s trial=%s obj=%.4f tie=%.4f",
        gen_best_generator,
        gen_best_row.get("trial_id"),
        float(gen_best_row.get("objective_value", np.nan)),
        float(gen_best_row.get("tie_break_value", np.nan)),
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "config_pack_base": str(args.config_pack),
        "objective_metric": metric,
        "alpha_ratio_ref": alpha_ratio_ref,
        "gen_sequence": gen_targets,
        "pilot": {
            "subjects": pilot_subjects,
            "seeds": pilot_seeds,
            "r_list": pilot_r,
            "screen_seeds": screen_seeds,
            "screen_r_list": screen_r,
        },
        "budget": {
            "eegnet_trials": eegnet_trials,
            "genaug_trials": gen_trials,
            "top_k": top_k,
        },
        "eegnet": {
            "trial_id": eeg_best_row["trial_id"],
            "phase": eeg_best_row["phase"],
            "objective_value": float(eeg_best_row["objective_value"]),
            "tie_break_value": float(eeg_best_row["tie_break_value"]),
            "params": eeg_best_params,
        },
        "genaug_qc": {
            "trial_id": gen_best_row["trial_id"],
            "phase": gen_best_row["phase"],
            "objective_value": float(gen_best_row["objective_value"]),
            "tie_break_value": float(gen_best_row["tie_break_value"]),
            "generator": gen_best_generator,
            "params": {
                "generator": gen_best_payload.get("generator", {}),
                "qc": gen_best_payload.get("qc", {}),
            },
        },
        "genaug_qc_all": [
            {
                "generator": str(e["generator"]),
                "trial_id": e["row"]["trial_id"],
                "phase": e["row"]["phase"],
                "objective_value": float(e["row"]["objective_value"]),
                "tie_break_value": float(e["row"]["tie_break_value"]),
                "params": {
                    "generator": e["payload"].get("generator", {}),
                    "qc": e["payload"].get("qc", {}),
                },
            }
            for e in gen_best_entries
        ],
    }
    write_json(best_json, summary)
    logger.info("tuning end best_params=%s trials_table=%s", best_json, trials_csv)


if __name__ == "__main__":
    main()
