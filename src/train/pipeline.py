from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.augment.generate import compute_target_counts, build_synthetic_with_qc
from src.data.dataset import load_index, load_samples
from src.data.normalize import ZScoreNormalizer
from src.data.subsample import load_split_indices
from src.eval.distance import classwise_distance_summary
from src.eval.embedding import FrozenEEGNetEmbedder, train_embedding_eegnet
from src.qc.qc_pipeline import fit_qc
from src.train.train_classifier import train_classifier
from src.train.train_generator import train_generator, sample_from_generator
from src.train.proxy_select_ckpt import select_best_checkpoint
from src.utils.alpha import alpha_ratio_to_mix
from src.utils.config import config_hash, save_yaml
from src.utils.io import ensure_dir, write_json
from src.utils.resource import get_git_commit
from src.utils.results import make_run_id
from src.utils.seed import set_global_seed, stable_hash_seed


def _resolve_device(train_cfg: dict) -> str:
    req = str(train_cfg.get("device", "auto"))
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return req


def _resolve_batch_size(train_cfg: dict, r: float) -> int:
    """Resolve batch size with optional r-dependent logic.

    Supported (optional) keys in train_cfg:
    - batch_size_by_r: mapping of {r_value: batch_size}, chooses max r_value <= r.
    - batch_size_scale_with_r: bool, scales base batch_size by r / r_ref.
      r_ref defaults to 0.1, min_scale defaults to 1.0, max_scale optional.
    """
    base = int(train_cfg.get("batch_size", 32))
    by_r = train_cfg.get("batch_size_by_r")
    if isinstance(by_r, dict) and by_r:
        items = sorted((float(k), int(v)) for k, v in by_r.items())
        chosen = items[0][1]
        for rk, bs in items:
            if r >= rk:
                chosen = bs
        return max(1, int(chosen))
    if bool(train_cfg.get("batch_size_scale_with_r", False)):
        r_ref = float(train_cfg.get("batch_size_r_ref", 0.1))
        min_scale = float(train_cfg.get("batch_size_min_scale", 1.0))
        max_scale = train_cfg.get("batch_size_max_scale")
        scale = max(min_scale, float(r) / max(r_ref, 1e-6))
        if max_scale is not None:
            scale = min(scale, float(max_scale))
        return max(1, int(round(base * scale)))
    return base


def _generator_run_id(
    subject: int,
    seed: int,
    r: float,
    generator: str,
    dataset_cfg: dict,
    preprocess_cfg: dict,
    gen_cfg: dict,
) -> str:
    payload = {
        "dataset": dataset_cfg.get("name", "bci2a"),
        "subject": subject,
        "seed": seed,
        "r": r,
        "generator": generator,
        "gen_model": gen_cfg.get("model", {}),
        "gen_train": gen_cfg.get("train", {}),
        "preprocess": preprocess_cfg,
    }
    return config_hash(payload)


def _list_ckpts(gen_run_dir: Path) -> list[str]:
    ckpts = sorted(gen_run_dir.glob("ckpt_epoch_*.pt"))
    return [str(p) for p in ckpts]


def run_experiment(
    subject: int,
    seed: int,
    r: float,
    method: str,
    classifier: str,
    generator: str,
    alpha_ratio: float,
    qc_on: bool,
    dataset_cfg: dict,
    preprocess_cfg: dict,
    split_cfg: dict,
    model_cfgs: dict,
    gen_cfgs: dict,
    qc_cfg: dict,
    results_path: str | Path,
    run_root: str | Path = "./artifacts/runs",
    stage: str = "full",
    compute_distance: bool = True,
) -> dict[str, Any]:
    start_time = time.time()
    set_global_seed(seed)

    index_df = load_index(dataset_cfg["index_path"])
    splits = load_split_indices(dataset_cfg["name"], subject, seed, r, root="./artifacts/splits")

    x_train, y_train = load_samples(index_df, splits["train_sub"])
    x_val, y_val = load_samples(index_df, splits["val"])
    x_test, y_test = load_samples(index_df, splits["test"])

    normalizer = ZScoreNormalizer(eps=float(preprocess_cfg.get("normalization", {}).get("eps", 1.0e-6)))
    normalizer.fit(x_train)
    x_train = normalizer.transform(x_train)
    x_val = normalizer.transform(x_val)
    x_test = normalizer.transform(x_test)
    normalizer_state = normalizer.state_dict()

    num_classes = int(np.max(y_train)) + 1
    run_key = {
        "subject": subject,
        "seed": seed,
        "r": r,
        "classifier": classifier,
        "method": method,
        "generator": generator,
        "qc_on": bool(qc_on),
        "alpha_ratio": float(alpha_ratio),
    }

    run_id = make_run_id(run_key)
    run_dir = ensure_dir(Path(run_root) / run_id)
    save_yaml(run_dir / "config_snapshot.yaml", {
        "dataset": dataset_cfg,
        "preprocess": preprocess_cfg,
        "split": split_cfg,
        "models": model_cfgs,
        "generators": gen_cfgs,
        "qc": qc_cfg,
        "run": run_key,
        "stage": stage,
    })

    synth_data = None
    qc_report = {"pass_rate": np.nan, "oversample_factor": np.nan}
    distance_rows: dict[str, Any] = {}
    ratio_effective = 0.0
    alpha_mix_effective = 0.0

    if method == "GenAug":
        if alpha_ratio <= 0.0:
            synth_data = (np.empty((0,) + x_train.shape[1:], dtype=np.float32), np.empty((0,), dtype=np.int64))
            ratio_effective = 0.0
            alpha_mix_effective = 0.0
        else:
            gen_cfg = gen_cfgs[generator]
            gen_id = _generator_run_id(
                subject=subject,
                seed=seed,
                r=r,
                generator=generator,
                dataset_cfg=dataset_cfg,
                preprocess_cfg=preprocess_cfg,
                gen_cfg=gen_cfg,
            )
            gen_run_dir = ensure_dir(Path("./artifacts/checkpoints") / f"gen_{gen_id}")
            gen_seed = stable_hash_seed(seed, {"generator": generator, "subject": subject, "r": r})

            ckpts = _list_ckpts(gen_run_dir)
            if not ckpts:
                gen_train_cfg = dict(gen_cfg["train"])
                gen_train_cfg["batch_size"] = _resolve_batch_size(gen_train_cfg, r)
                gen_train = train_generator(
                    x_train=x_train,
                    y_train=y_train,
                    model_type=generator,
                    model_cfg=gen_cfg["model"],
                    train_cfg=gen_train_cfg,
                    run_dir=gen_run_dir,
                    seed=gen_seed,
                )
                ckpts = gen_train.get("ckpts", [])
                if not ckpts:
                    ckpts = _list_ckpts(gen_run_dir)

            if not ckpts:
                raise RuntimeError("No generator checkpoints produced.")

            ckpt_sel_cfg = gen_cfg.get("checkpoint_selection", {})
            if ckpt_sel_cfg.get("enabled", True):
                best_ckpt_path = gen_run_dir / "best_ckpt.json"
                if best_ckpt_path.exists():
                    try:
                        import json

                        best_ckpt = json.loads(best_ckpt_path.read_text()).get("best_ckpt")
                    except Exception:
                        best_ckpt = None
                    if not best_ckpt:
                        best_ckpt = ckpts[-1]
                else:
                    sel = select_best_checkpoint(
                        ckpt_paths=ckpts,
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        gen_cfg=gen_cfg,
                        proxy_model_cfg=model_cfgs["eegnet"]["model"],
                        qc_cfg=qc_cfg,
                        num_classes=num_classes,
                        run_dir=gen_run_dir,
                        seed=gen_seed,
                    )
                    best_ckpt = sel.get("best_ckpt") or ckpts[-1]
            else:
                best_ckpt = ckpts[-1]

            target_counts = compute_target_counts(y_train, alpha_ratio)
            qc_state = None
            if qc_on:
                qc_state = fit_qc(x_train, y_train, sfreq=int(dataset_cfg.get("sfreq", 250)), cfg=qc_cfg)

            sample_cfg = gen_cfg.get("sample", {})
            buffer = float(sample_cfg.get("dynamic_buffer", 1.2))
            ddpm_steps = sample_cfg.get("ddpm_steps")

            def _sample_fn(cls: int, n: int) -> np.ndarray:
                y = np.full((n,), int(cls), dtype=np.int64)
                return sample_from_generator(
                    best_ckpt,
                    y,
                    device=_resolve_device(gen_cfg.get("train", {})),
                    ddpm_steps=ddpm_steps,
                )

            x_syn, y_syn, qc_report = build_synthetic_with_qc(
                sample_fn=_sample_fn,
                target_counts=target_counts,
                qc_state=qc_state if qc_on else None,
                qc_cfg=qc_cfg,
                sfreq=int(dataset_cfg.get("sfreq", 250)),
                buffer=buffer,
            )
            synth_data = (x_syn, y_syn)
            ratio_effective = float(len(x_syn) / max(1, len(x_train)))
            alpha_mix_effective = alpha_ratio_to_mix(ratio_effective)

            # Distance analysis
            if compute_distance and len(x_syn) > 0:
                embed_dir = ensure_dir(run_dir / "embedding")
                embed_ckpt = embed_dir / "ckpt.pt"
                if not embed_ckpt.exists():
                    embed_ckpt = train_embedding_eegnet(
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        model_cfg=model_cfgs["eegnet"]["model"],
                        train_cfg=model_cfgs["eegnet"]["train"],
                        run_dir=embed_dir,
                        normalizer_state=normalizer_state,
                        num_classes=num_classes,
                    )
                embedder = FrozenEEGNetEmbedder(embed_ckpt, device=_resolve_device(model_cfgs["eegnet"]["train"]))
                real_emb = embedder.transform(x_train)
                syn_emb = embedder.transform(x_syn)
                distance_rows = classwise_distance_summary(
                    real_emb,
                    y_train,
                    syn_emb,
                    y_syn,
                    num_classes=num_classes,
                    n_projections=int(gen_cfg.get("distance", {}).get("n_projections", 64)),
                    seed=seed,
                )

    model_cfg = model_cfgs[classifier]["model"]
    train_cfg = dict(model_cfgs[classifier]["train"])
    train_cfg["batch_size"] = _resolve_batch_size(train_cfg, r)
    eval_cfg = model_cfgs[classifier]["evaluation"]

    evaluate_test = stage in {"final_eval", "full"}
    metrics = train_classifier(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        model_type=classifier,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        method=method,
        alpha_ratio=alpha_ratio,
        num_classes=num_classes,
        run_dir=run_dir,
        normalizer_state=normalizer_state,
        synth_data=synth_data,
        evaluate_test=evaluate_test,
        aug_cfg=model_cfgs.get("augment", {}),
    )

    runtime_sec = float(time.time() - start_time)
    row = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "config_hash": config_hash({
            "dataset": dataset_cfg,
            "preprocess": preprocess_cfg,
            "split": split_cfg,
            "models": model_cfgs,
            "generators": gen_cfgs,
            "qc": qc_cfg,
        }),
        "dataset": dataset_cfg["name"],
        "subject": subject,
        "seed": seed,
        "r": r,
        "method": method,
        "classifier": classifier,
        "generator": generator,
        "alpha_ratio": float(alpha_ratio),
        "qc_on": bool(qc_on),
        "acc": metrics.get("acc", np.nan),
        "kappa": metrics.get("kappa", np.nan),
        "macro_f1": metrics.get("macro_f1", np.nan),
        "val_acc": metrics.get("val_acc", np.nan),
        "val_kappa": metrics.get("val_kappa", np.nan),
        "val_macro_f1": metrics.get("val_macro_f1", np.nan),
        "pass_rate": qc_report.get("pass_rate", np.nan),
        "oversample_factor": qc_report.get("oversample_factor", np.nan),
        "ratio_effective": ratio_effective,
        "alpha_mix_effective": alpha_mix_effective,
        "runtime_sec": runtime_sec,
        "device": _resolve_device(train_cfg),
    }
    row.update(distance_rows)

    write_json(run_dir / "metrics_row.json", row)
    return row
