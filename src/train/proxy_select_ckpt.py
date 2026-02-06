from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.augment.generate import build_synthetic_with_qc
from src.qc.qc_pipeline import fit_qc
from src.train.train_classifier import train_classifier
from src.train.train_generator import LoadedGeneratorSampler
from src.utils.io import ensure_dir, write_json
from src.utils.seed import stable_hash_seed, set_global_seed


def _resolve_device(req: str) -> str:
    """Resolve device string, supporting 'auto'."""
    if str(req) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(req)


def _sample_by_class(
    sampler: LoadedGeneratorSampler,
    n_per_class: int,
    num_classes: int,
    ddpm_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a fixed number of synthetic trials per class from a checkpoint.

    Inputs:
    - sampler: loaded generator sampler.
    - n_per_class: number of samples per class.
    - num_classes: total number of classes.
    - ddpm_steps: optional DDPM steps.

    Outputs:
    - x: ndarray [N, C, T] synthetic samples.
    - y: ndarray [N] class labels.

    Internal logic:
    - Builds a repeated label vector and samples from the loaded generator.
    """
    y = np.repeat(np.arange(num_classes, dtype=np.int64), int(n_per_class))
    x = sampler.sample(y, ddpm_steps=ddpm_steps)
    return x, y


def select_best_checkpoint(
    ckpt_paths: list[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    gen_cfg: dict,
    proxy_model_cfg: dict,
    qc_cfg: dict,
    num_classes: int,
    run_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Select the best generator checkpoint using a proxy classifier on T_val.

    Inputs:
    - ckpt_paths: list of generator checkpoint paths.
    - x_train/y_train: real train data [N, C, T] / [N].
    - x_val/y_val: validation data [M, C, T] / [M].
    - gen_cfg: generator config (includes selection settings).
    - proxy_model_cfg: EEGNet config for proxy training.
    - qc_cfg: QC config.
    - num_classes: number of classes.
    - run_dir: directory to save proxy artifacts.
    - seed: RNG seed.

    Outputs:
    - dict with best_ckpt, per-ckpt scores, and metric name.

    Internal logic:
    - For each checkpoint: sample, optionally QC filter, quick-train EEGNet, score on val.
    """
    ensure_dir(run_dir)
    ckpt_scores = []

    sel_cfg = gen_cfg.get("checkpoint_selection", {})
    metric = str(sel_cfg.get("metric", "kappa"))  # validation metric to maximize
    alpha_ratio_ref = float(sel_cfg.get("alpha_ratio_ref", 1.0))  # proxy alpha ratio
    sample_n_per_class = int(sel_cfg.get("sample_n_per_class", 50))  # proxy samples per class
    qc_enabled = bool(sel_cfg.get("qc_enabled", True))  # apply QC during proxy selection
    overgen_buffer = float(sel_cfg.get("overgen_buffer", 1.2))  # oversampling buffer

    proxy_cfg = sel_cfg.get("proxy_classifier", {})
    proxy_steps = int(proxy_cfg.get("steps", 120))  # total proxy training steps
    proxy_batch = int(proxy_cfg.get("batch_size", 64))  # proxy batch size
    proxy_lr = float(proxy_cfg.get("lr", 1e-3))  # proxy learning rate
    ddpm_steps = gen_cfg.get("sample", {}).get("ddpm_steps")
    sample_device = _resolve_device(str(gen_cfg.get("train", {}).get("device", "auto")))
    proxy_device = _resolve_device(str(proxy_cfg.get("device", "auto")))

    qc_state = None
    if qc_enabled:
        qc_state = fit_qc(x_train, y_train, sfreq=int(gen_cfg.get("sfreq", 250)), cfg=qc_cfg)

    best_ckpt = None
    best_score = None

    for ckpt_path in ckpt_paths:
        ckpt_seed = stable_hash_seed(seed, {"ckpt": str(ckpt_path)})
        set_global_seed(ckpt_seed)
        sampler = LoadedGeneratorSampler(ckpt_path=ckpt_path, device=sample_device)

        x_syn, y_syn = _sample_by_class(
            sampler,
            n_per_class=sample_n_per_class,
            num_classes=num_classes,
            ddpm_steps=ddpm_steps,
        )

        if qc_enabled and qc_state is not None:
            target_counts = {int(c): int(sample_n_per_class) for c in range(num_classes)}
            x_syn, y_syn, _ = build_synthetic_with_qc(
                sample_fn=lambda cls, n: sampler.sample(np.full((n,), cls, dtype=np.int64), ddpm_steps=ddpm_steps),
                target_counts=target_counts,
                qc_state=qc_state,
                qc_cfg=qc_cfg,
                sfreq=int(gen_cfg.get("sfreq", 250)),
                buffer=overgen_buffer,
            )

        train_cfg = {
            "device": proxy_device,
            "batch_size": proxy_batch,
            "lr": proxy_lr,
            "epochs": 1,
            "num_workers": 0,
            "step_control": {
                "enabled": True,
                "total_steps": proxy_steps,
                "steps_per_eval": max(1, proxy_steps // 3),
            },
        }
        eval_cfg = {"best_metric": metric, "best_direction": "max"}

        proxy_dir = ensure_dir(run_dir / f"proxy_{Path(ckpt_path).stem}")
        metrics = train_classifier(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_val,
            y_test=y_val,
            model_type="eegnet",
            model_cfg=proxy_model_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            method="GenAug",
            alpha_ratio=alpha_ratio_ref,
            num_classes=num_classes,
            run_dir=proxy_dir,
            normalizer_state={},
            synth_data=(x_syn, y_syn) if len(x_syn) > 0 else None,
            evaluate_test=False,
        )

        score = float(metrics.get(f"val_{metric}", metrics.get(metric, np.nan)))
        ckpt_scores.append({"ckpt": str(ckpt_path), "score": score})
        if best_score is None or score > best_score:
            best_score = score
            best_ckpt = str(ckpt_path)

    out = {"best_ckpt": best_ckpt, "scores": ckpt_scores, "metric": metric}
    write_json(run_dir / "best_ckpt.json", out)
    return out
