from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataio import load_samples_by_ids
from src.eval import compute_metrics
from src.models_clf import (
    build_svm_classifier,
    build_torch_classifier,
    is_sklearn_model,
    normalize_classifier_type,
)
from src.preprocess import ZScoreNormalizer
from src.utils import append_jsonl, ensure_dir


def _classical_augment_numpy(
    x: np.ndarray,
    noise_std: float,
    max_shift: int,
    channel_dropout_prob: float,
) -> np.ndarray:
    out = x.copy().astype(np.float32)

    if noise_std > 0:
        out += np.random.randn(*out.shape).astype(np.float32) * float(noise_std)

    if max_shift > 0:
        shifts = np.random.randint(-max_shift, max_shift + 1, size=out.shape[0])
        for i, s in enumerate(shifts):
            out[i] = np.roll(out[i], shift=int(s), axis=-1)

    if channel_dropout_prob > 0:
        drop_mask = np.random.rand(out.shape[0], out.shape[1]) < channel_dropout_prob
        out[drop_mask, :] = 0.0

    return out


def _mixup_numpy_hard(
    xa: np.ndarray,
    ya: np.ndarray,
    xb: np.ndarray,
    yb: np.ndarray,
    beta_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    beta_alpha = max(float(beta_alpha), 1e-3)
    lam = np.random.beta(beta_alpha, beta_alpha, size=xa.shape[0]).astype(np.float32)
    lam_x = lam[:, None, None]
    xm = lam_x * xa + (1.0 - lam_x) * xb
    ym = np.where(lam >= 0.5, ya, yb)
    return xm.astype(np.float32), ym.astype(np.int64)


def _seg_reconstruct_augment_numpy(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    n_classes: int,
    n_per_class: int,
    n_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_per_class <= 0 or n_segments <= 1:
        return np.empty((0, x_pool.shape[1], x_pool.shape[2]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    x_pool = x_pool.astype(np.float32)
    y_pool = y_pool.astype(np.int64)
    seg_points = np.linspace(0, x_pool.shape[-1], n_segments + 1, dtype=int)

    aug_x = []
    aug_y = []
    for cls in range(n_classes):
        cls_idx = np.where(y_pool == cls)[0]
        if len(cls_idx) == 0:
            continue

        out = np.zeros((n_per_class, x_pool.shape[1], x_pool.shape[2]), dtype=np.float32)
        for i in range(n_per_class):
            for s in range(n_segments):
                a = int(seg_points[s])
                b = int(seg_points[s + 1])
                src = int(np.random.choice(cls_idx))
                out[i, :, a:b] = x_pool[src, :, a:b]
        aug_x.append(out)
        aug_y.append(np.full((n_per_class,), cls, dtype=np.int64))

    if not aug_x:
        return np.empty((0, x_pool.shape[1], x_pool.shape[2]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    x_aug = np.concatenate(aug_x, axis=0)
    y_aug = np.concatenate(aug_y, axis=0)
    perm = np.random.permutation(len(x_aug))
    return x_aug[perm], y_aug[perm]


def _build_offline_augmented_bank(
    mode: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    ratio: float,
    clf_cfg: Dict,
) -> tuple[np.ndarray, np.ndarray]:
    if ratio <= 0:
        return np.empty((0, x_train.shape[1], x_train.shape[2]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    n_add = int(round(len(x_train) * float(ratio)))
    if n_add <= 0:
        return np.empty((0, x_train.shape[1], x_train.shape[2]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    if mode == "classical":
        aug_cfg = clf_cfg["augmentation"].get("classical", {})
        idx = np.random.choice(len(x_train), size=n_add, replace=True)
        base_x = x_train[idx]
        base_y = y_train[idx]
        x_aug = _classical_augment_numpy(
            base_x,
            noise_std=float(aug_cfg.get("noise_std", 0.01)),
            max_shift=int(aug_cfg.get("max_time_shift", 20)),
            channel_dropout_prob=float(aug_cfg.get("channel_dropout_prob", 0.1)),
        )
        return x_aug, base_y.astype(np.int64)

    if mode == "mixup":
        mix_cfg = clf_cfg["augmentation"].get("mixup", {})
        beta_alpha = float(mix_cfg.get("alpha", 0.2))
        idx_a = np.random.choice(len(x_train), size=n_add, replace=True)
        idx_b = np.random.choice(len(x_train), size=n_add, replace=True)
        x_aug, y_aug = _mixup_numpy_hard(
            x_train[idx_a],
            y_train[idx_a],
            x_train[idx_b],
            y_train[idx_b],
            beta_alpha=beta_alpha,
        )
        return x_aug, y_aug

    if mode == "paper_sr":
        sr_cfg = clf_cfg["augmentation"].get("paper_sr", {})
        n_classes = int(np.max(y_train)) + 1
        n_per_class = max(1, int(np.ceil((n_add / max(1, n_classes)))))
        x_aug, y_aug = _seg_reconstruct_augment_numpy(
            x_pool=x_train,
            y_pool=y_train,
            n_classes=n_classes,
            n_per_class=n_per_class,
            n_segments=int(sr_cfg.get("n_segments", 8)),
        )
        if len(x_aug) > n_add:
            keep = np.random.choice(len(x_aug), size=n_add, replace=False)
            x_aug = x_aug[keep]
            y_aug = y_aug[keep]
        return x_aug, y_aug

    return np.empty((0, x_train.shape[1], x_train.shape[2]), dtype=np.float32), np.empty((0,), dtype=np.int64)


def _proportional_allocation(counts: np.ndarray, total: int) -> np.ndarray:
    counts = counts.astype(np.float64)
    out = np.zeros_like(counts, dtype=np.int64)
    if total <= 0 or float(counts.sum()) <= 0:
        return out
    raw = counts / counts.sum() * float(total)
    base = np.floor(raw).astype(np.int64)
    remain = int(total - int(base.sum()))
    if remain > 0:
        frac = raw - base
        order = np.argsort(-frac)
        base[order[:remain]] += 1
    return base.astype(np.int64)


def _sample_gen_aug_class_conditional(
    sx_raw: np.ndarray,
    sy: np.ndarray,
    y_real: np.ndarray,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_real = int(len(y_real))
    n_add = int(round(float(n_real) * float(ratio)))
    if n_add <= 0 or len(sx_raw) == 0:
        return (
            np.empty((0, sx_raw.shape[1], sx_raw.shape[2]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {
                "aug_sampling_strategy": "gen_aug_class_conditional",
                "n_add_requested": int(max(0, n_add)),
                "n_add_actual": 0,
                "class_counts_target": {},
                "class_counts_actual": {},
                "synth_pool_class_counts": {},
                "missing_classes_in_synth_pool": [],
                "replacement_used_classes": [],
                "global_replace_used": False,
            },
        )

    y_real = y_real.astype(np.int64)
    sy = sy.astype(np.int64)
    n_classes = int(max(np.max(y_real), np.max(sy))) + 1

    real_counts = np.bincount(y_real, minlength=n_classes)
    target_counts = _proportional_allocation(real_counts, n_add)
    pool_counts = np.bincount(sy, minlength=n_classes)

    missing = [int(c) for c in range(n_classes) if target_counts[c] > 0 and pool_counts[c] <= 0]

    if int(target_counts.sum()) <= 0:
        return (
            np.empty((0, sx_raw.shape[1], sx_raw.shape[2]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {
                "aug_sampling_strategy": "gen_aug_class_conditional",
                "n_add_requested": int(n_add),
                "n_add_actual": 0,
                "class_counts_target": {str(c): int(target_counts[c]) for c in range(n_classes)},
                "class_counts_actual": {},
                "synth_pool_class_counts": {str(c): int(pool_counts[c]) for c in range(n_classes)},
                "missing_classes_in_synth_pool": missing,
                "replacement_used_classes": [],
                "global_replace_used": False,
            },
        )

    xs = []
    ys = []
    replacement_used = []
    for c in range(n_classes):
        k = int(target_counts[c])
        if k <= 0:
            continue
        pool_idx = np.where(sy == c)[0]
        if len(pool_idx) <= 0:
            continue
        use_replace = len(pool_idx) < k
        pick = np.random.choice(pool_idx, size=k, replace=use_replace)
        xs.append(sx_raw[pick].astype(np.float32))
        ys.append(np.full((k,), c, dtype=np.int64))
        if use_replace:
            replacement_used.append(int(c))

    if not xs:
        return (
            np.empty((0, sx_raw.shape[1], sx_raw.shape[2]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {
                "aug_sampling_strategy": "gen_aug_class_conditional",
                "n_add_requested": int(n_add),
                "n_add_actual": 0,
                "class_counts_target": {str(c): int(target_counts[c]) for c in range(n_classes)},
                "class_counts_actual": {},
                "synth_pool_class_counts": {str(c): int(pool_counts[c]) for c in range(n_classes)},
                "missing_classes_in_synth_pool": missing,
                "replacement_used_classes": [],
                "global_replace_used": False,
            },
        )

    x_aug = np.concatenate(xs, axis=0)
    y_aug = np.concatenate(ys, axis=0)
    perm = np.random.permutation(len(y_aug))
    x_aug = x_aug[perm]
    y_aug = y_aug[perm]
    actual_counts = np.bincount(y_aug, minlength=n_classes)

    meta = {
        "aug_sampling_strategy": "gen_aug_class_conditional",
        "n_add_requested": int(n_add),
        "n_add_actual": int(len(y_aug)),
        "class_counts_target": {str(c): int(target_counts[c]) for c in range(n_classes)},
        "class_counts_actual": {str(c): int(actual_counts[c]) for c in range(n_classes)},
        "synth_pool_class_counts": {str(c): int(pool_counts[c]) for c in range(n_classes)},
        "missing_classes_in_synth_pool": missing,
        "replacement_used_classes": replacement_used,
        "global_replace_used": bool(len(replacement_used) > 0),
    }
    return x_aug, y_aug, meta


def _evaluate_torch(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module | None = None,
):
    model.eval()
    ys, ps = [], []
    losses = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb_dev = yb.to(device)
            logits = model(xb)
            if criterion is not None:
                losses.append(float(criterion(logits, yb_dev).item()))
            pred = logits.argmax(dim=1).cpu().numpy()
            ps.append(pred)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    out = compute_metrics(y_true, y_pred)
    if losses:
        out["loss"] = float(np.mean(losses))
    return out


def _evaluate_svm(model, x: np.ndarray, y: np.ndarray):
    pred = model.predict(x)
    return compute_metrics(y, pred)


def _base_clf_ckpt_payload(
    model_type: str,
    mode: str,
    norm: ZScoreNormalizer,
    shape: dict,
    n_classes: int,
) -> dict:
    return {
        "normalizer": norm.state_dict(),
        "shape": shape,
        "n_classes": int(n_classes),
        "mode": mode,
        "model_type": model_type,
    }


def train_classifier(
    split: Dict,
    index_df: pd.DataFrame,
    clf_cfg: Dict,
    preprocess_cfg: Dict,
    out_dir: str | Path,
    mode: str,
    ratio: float = 0.0,
    synth_npz: Optional[str] = None,
    synth_ratio: Optional[float] = None,
    aug_strength: Optional[float] = None,
    evaluate_test: Optional[bool] = None,
) -> Dict[str, float]:
    # Backward-compatible aliases (legacy callers may pass synth_ratio/aug_strength).
    if synth_ratio is not None:
        ratio = float(synth_ratio)
    if aug_strength is not None:
        ratio = float(aug_strength)
    ratio = float(ratio)
    if evaluate_test is None:
        evaluate_test = bool(clf_cfg.get("evaluation", {}).get("evaluate_test", False))

    x_train_real, y_train_real = load_samples_by_ids(index_df, split["train_ids"])
    x_val, y_val = load_samples_by_ids(index_df, split["val_ids"])
    x_test = y_test = None
    if evaluate_test:
        x_test, y_test = load_samples_by_ids(index_df, split["test_ids"])

    # Train-only normalization statistics.
    norm_cfg = preprocess_cfg["normalization"]
    norm = ZScoreNormalizer(
        eps=float(norm_cfg.get("eps", 1e-6)),
        mode=str(norm_cfg.get("mode", "channel_global")),
    ).fit(x_train_real)

    x_train = norm.transform(x_train_real)
    y_train = y_train_real.copy()
    x_val = norm.transform(x_val)
    if evaluate_test and x_test is not None:
        x_test = norm.transform(x_test)

    exp_dir = ensure_dir(out_dir)
    log_path = exp_dir / "log.jsonl"

    # Build augmented-only bank for analysis/logging and append to train set.
    x_added_raw = np.empty((0, x_train_real.shape[1], x_train_real.shape[2]), dtype=np.float32)
    y_added = np.empty((0,), dtype=np.int64)
    aug_sampling_meta = {
        "aug_sampling_strategy": "none",
        "n_add_requested": 0,
        "n_add_actual": 0,
    }

    if mode in {"classical", "mixup", "paper_sr"}:
        x_added_raw, y_added = _build_offline_augmented_bank(
            mode=mode,
            x_train=x_train_real,
            y_train=y_train,
            ratio=ratio,
            clf_cfg=clf_cfg,
        )

    elif mode == "gen_aug" and ratio > 0 and synth_npz is not None:
        synth = np.load(synth_npz)
        sx_raw = synth["X"].astype(np.float32)
        sy = synth["y"].astype(np.int64)
        x_added_raw, y_added, aug_sampling_meta = _sample_gen_aug_class_conditional(
            sx_raw=sx_raw,
            sy=sy,
            y_real=y_train_real,
            ratio=ratio,
        )

    if len(x_added_raw) > 0:
        x_added = norm.transform(x_added_raw)
        x_train = np.concatenate([x_train, x_added], axis=0)
        y_train = np.concatenate([y_train, y_added], axis=0)
        np.savez_compressed(exp_dir / "aug_used.npz", X=x_added_raw, y=y_added)

    model_type = normalize_classifier_type(str(clf_cfg["model"].get("type", "eegnet")))
    shape = {"c": int(x_train.shape[1]), "t": int(x_train.shape[2])}
    n_classes = int(np.max(y_train)) + 1
    alpha_tilde = float(ratio / (1.0 + ratio))
    n_train_real = int(len(x_train_real))
    n_train_aug = int(len(x_added_raw))
    n_train_total = int(len(x_train))
    ratio_effective = float(n_train_aug / max(1, n_train_real))
    alpha_effective = float(n_train_aug / max(1, n_train_real + n_train_aug))
    train_meta = {
        "mode": mode,
        "model_type": model_type,
        "ratio": float(ratio),
        "alpha_tilde": alpha_tilde,
        "ratio_effective": ratio_effective,
        "alpha_effective": alpha_effective,
        "evaluate_test": bool(evaluate_test),
        "n_train_real": n_train_real,
        "n_train_aug": n_train_aug,
        "n_train_total": n_train_total,
        "batch_size": int(clf_cfg["train"].get("batch_size", 64)),
        "sampling_strategy": "real_plus_aug_concat",
        **aug_sampling_meta,
    }

    # Best checkpoint selection should be configurable (BCI2a is often reported with Kappa).
    eval_cfg = clf_cfg.get("evaluation", {})
    best_ckpt_metric = str(eval_cfg.get("best_ckpt_metric", "kappa")).strip()
    best_ckpt_direction = str(eval_cfg.get("best_ckpt_direction", "")).strip().lower()
    if best_ckpt_direction not in {"min", "max"}:
        best_ckpt_direction = "min" if best_ckpt_metric.endswith("loss") else "max"
    train_meta.update(
        {
            "best_ckpt_metric": best_ckpt_metric,
            "best_ckpt_direction": best_ckpt_direction,
        }
    )

    def _score_from_metrics(m: dict) -> float:
        # Fall back gracefully to avoid "no ckpt saved" crashes due to a misconfigured key.
        if best_ckpt_metric in m and np.isfinite(m.get(best_ckpt_metric, np.nan)):
            return float(m[best_ckpt_metric])
        for k in ("kappa", "bal_acc", "acc", "f1_macro", "loss"):
            if k in m and np.isfinite(m.get(k, np.nan)):
                return float(m[k])
        return float("inf") if best_ckpt_direction == "min" else float("-inf")

    def _is_better(cur: float, best: float) -> bool:
        if best_ckpt_direction == "min":
            return cur < best
        return cur > best

    ckpt_base = _base_clf_ckpt_payload(
        model_type=model_type,
        mode=mode,
        norm=norm,
        shape=shape,
        n_classes=n_classes,
    )

    if is_sklearn_model(model_type):
        svm = build_svm_classifier(clf_cfg["model"])
        svm.fit(x_train, y_train)

        val_metrics = _evaluate_svm(svm, x_val, y_val)
        test_metrics = _evaluate_svm(svm, x_test, y_test) if evaluate_test and x_test is not None and y_test is not None else {}

        append_jsonl(log_path, {"epoch": 1, "train_size": int(len(x_train)), **val_metrics})
        joblib.dump({"svm_model": svm, **ckpt_base}, exp_dir / "ckpt.pkl")
        train_meta.update(
            {
                "step_mode": "not_applicable_svm",
                "total_steps_target": 0,
                "total_steps_done": 0,
                "steps_per_eval": 0,
                "best_ckpt_score": float(_score_from_metrics(val_metrics)),
                "best_ckpt_step": None,
                "best_ckpt_epoch": 1,
            }
        )
        with open(exp_dir / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(train_meta, f, ensure_ascii=True, indent=2)

        selected = test_metrics if evaluate_test else val_metrics
        out = {
            "acc": float(selected.get("acc", np.nan)),
            "bal_acc": float(selected.get("bal_acc", np.nan)),
            "kappa": float(selected.get("kappa", np.nan)),
            "f1_macro": float(selected.get("f1_macro", np.nan)),
            "evaluated_on": "test" if evaluate_test else "val",
            "test_acc": float(test_metrics.get("acc", np.nan)),
            "test_bal_acc": float(test_metrics.get("bal_acc", np.nan)),
            "test_kappa": float(test_metrics.get("kappa", np.nan)),
            "test_f1_macro": float(test_metrics.get("f1_macro", np.nan)),
            "val_acc": float(val_metrics["acc"]),
            "val_bal_acc": float(val_metrics.get("bal_acc", np.nan)),
            "val_kappa": float(val_metrics["kappa"]),
            "val_f1_macro": float(val_metrics["f1_macro"]),
            "n_train_real": n_train_real,
            "n_train_aug": n_train_aug,
            "n_train_total": n_train_total,
            "ratio": float(ratio),
            "alpha_tilde": alpha_tilde,
            "ratio_effective": ratio_effective,
            "alpha_effective": alpha_effective,
            "total_steps_target": 0,
            "total_steps_done": 0,
            "best_ckpt_metric": best_ckpt_metric,
            "best_ckpt_direction": best_ckpt_direction,
            "best_ckpt_score": float(_score_from_metrics(val_metrics)),
            "best_ckpt_step": None,
            "best_ckpt_epoch": 1,
        }
        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=True, indent=2)
        return out

    dev = str(clf_cfg["train"].get("device", "auto"))
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    model = build_torch_classifier(
        model_type=model_type,
        n_ch=x_train.shape[1],
        n_t=x_train.shape[2],
        n_classes=n_classes,
        cfg=clf_cfg["model"],
    ).to(device)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_ds = None
    if evaluate_test and x_test is not None and y_test is not None:
        test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    batch_size = int(clf_cfg["train"].get("batch_size", 64))
    num_workers = int(clf_cfg["train"].get("num_workers", 0))
    tr_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    te_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_ds is not None else None

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(clf_cfg["train"].get("lr", 1e-3)),
        weight_decay=float(clf_cfg["train"].get("weight_decay", 1e-4)),
    )
    sched = None
    sched_cfg = clf_cfg["train"].get("scheduler", {})
    if bool(sched_cfg.get("enabled", False)) and str(sched_cfg.get("type", "plateau")).lower() == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=str(sched_cfg.get("mode", "min")),
            factor=float(sched_cfg.get("factor", 0.9)),
            patience=int(sched_cfg.get("patience", 20)),
            min_lr=float(sched_cfg.get("min_lr", 1e-4)),
        )
    ce = torch.nn.CrossEntropyLoss()

    best_score = float("inf") if best_ckpt_direction == "min" else float("-inf")
    best_val_metrics = {"acc": 0.0, "kappa": 0.0, "f1_macro": 0.0}
    best_ckpt_step: int | None = None
    best_ckpt_epoch: int | None = None
    last_val_metrics: dict | None = None
    total_steps_done = 0
    step_cfg = clf_cfg["train"].get("step_control", {})
    use_fixed_steps = bool(step_cfg.get("enabled", False)) and int(step_cfg.get("total_steps", 0)) > 0
    total_steps_target = int(step_cfg.get("total_steps", 0)) if use_fixed_steps else 0
    steps_per_eval = int(step_cfg.get("steps_per_eval", max(1, len(tr_dl))))
    steps_per_eval = max(1, steps_per_eval)

    if use_fixed_steps:
        data_iter = iter(tr_dl)
        while total_steps_done < total_steps_target:
            model.train()
            loss_meter = []
            n_block = min(steps_per_eval, total_steps_target - total_steps_done)
            for _ in range(n_block):
                try:
                    xb, yb = next(data_iter)
                except StopIteration:
                    data_iter = iter(tr_dl)
                    xb, yb = next(data_iter)

                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = ce(logits, yb)

                # Official faithful ATCNet includes explicit L2 regularization terms in the reference code.
                if hasattr(model, "regularization_loss"):
                    loss = loss + model.regularization_loss()

                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_meter.append(float(loss.item()))
                total_steps_done += 1

            val_metrics = _evaluate_torch(model, va_dl, device, criterion=ce)
            last_val_metrics = val_metrics
            if sched is not None:
                sched.step(float(val_metrics.get("loss", np.mean(loss_meter))))
            append_jsonl(
                log_path,
                {
                    "global_step": total_steps_done,
                    "epoch_equiv": float(total_steps_done / max(1, len(tr_dl))),
                    "train_loss": float(np.mean(loss_meter)),
                    **val_metrics,
                },
            )

            score = _score_from_metrics(val_metrics)
            if _is_better(score, best_score):
                best_score = score
                best_val_metrics = val_metrics
                best_ckpt_step = int(total_steps_done)
                best_ckpt_epoch = None
                torch.save(
                    {"state_dict": model.state_dict(), **ckpt_base},
                    exp_dir / "ckpt.pt",
                )
    else:
        for ep in range(1, int(clf_cfg["train"].get("epochs", 80)) + 1):
            model.train()
            loss_meter = []

            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = ce(logits, yb)

                # Official faithful ATCNet includes explicit L2 regularization terms in the reference code.
                if hasattr(model, "regularization_loss"):
                    loss = loss + model.regularization_loss()

                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_meter.append(float(loss.item()))
                total_steps_done += 1

            val_metrics = _evaluate_torch(model, va_dl, device, criterion=ce)
            last_val_metrics = val_metrics
            if sched is not None:
                sched.step(float(val_metrics.get("loss", np.mean(loss_meter))))
            append_jsonl(log_path, {"epoch": ep, "train_loss": float(np.mean(loss_meter)), **val_metrics})

            score = _score_from_metrics(val_metrics)
            if _is_better(score, best_score):
                best_score = score
                best_val_metrics = val_metrics
                best_ckpt_step = None
                best_ckpt_epoch = int(ep)
                torch.save(
                    {"state_dict": model.state_dict(), **ckpt_base},
                    exp_dir / "ckpt.pt",
                )

    ckpt_path = exp_dir / "ckpt.pt"
    if not ckpt_path.exists():
        # Defensive fallback for extreme instability (e.g., all-NaN validation metrics).
        torch.save(
            {"state_dict": model.state_dict(), **ckpt_base},
            ckpt_path,
        )
        if last_val_metrics is not None:
            best_val_metrics = dict(last_val_metrics)
            best_score = float(_score_from_metrics(best_val_metrics))

    # Evaluate best checkpoint on requested split.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = _evaluate_torch(model, te_dl, device) if te_dl is not None else {}
    train_meta.update(
        {
            "step_mode": "fixed_steps" if use_fixed_steps else "epoch_loop",
            "total_steps_target": int(total_steps_target),
            "total_steps_done": int(total_steps_done),
            "steps_per_eval": int(steps_per_eval if use_fixed_steps else max(1, len(tr_dl))),
            "best_ckpt_score": float(best_score),
            "best_ckpt_step": int(best_ckpt_step) if best_ckpt_step is not None else None,
            "best_ckpt_epoch": int(best_ckpt_epoch) if best_ckpt_epoch is not None else None,
        }
    )
    with open(exp_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, ensure_ascii=True, indent=2)

    selected = test_metrics if evaluate_test else best_val_metrics
    out = {
        "acc": float(selected.get("acc", np.nan)),
        "bal_acc": float(selected.get("bal_acc", np.nan)),
        "kappa": float(selected.get("kappa", np.nan)),
        "f1_macro": float(selected.get("f1_macro", np.nan)),
        "evaluated_on": "test" if evaluate_test else "val",
        "test_acc": float(test_metrics.get("acc", np.nan)),
        "test_bal_acc": float(test_metrics.get("bal_acc", np.nan)),
        "test_kappa": float(test_metrics.get("kappa", np.nan)),
        "test_f1_macro": float(test_metrics.get("f1_macro", np.nan)),
        "val_acc": float(best_val_metrics.get("acc", 0.0)),
        "val_bal_acc": float(best_val_metrics.get("bal_acc", np.nan)),
        "val_kappa": float(best_val_metrics.get("kappa", 0.0)),
        "val_f1_macro": float(best_val_metrics.get("f1_macro", 0.0)),
        "n_train_real": n_train_real,
        "n_train_aug": n_train_aug,
        "n_train_total": n_train_total,
        "ratio": float(ratio),
        "alpha_tilde": alpha_tilde,
        "ratio_effective": ratio_effective,
        "alpha_effective": alpha_effective,
        "total_steps_target": int(total_steps_target),
        "total_steps_done": int(total_steps_done),
        "best_ckpt_metric": best_ckpt_metric,
        "best_ckpt_direction": best_ckpt_direction,
        "best_ckpt_score": float(best_score),
        "best_ckpt_step": int(best_ckpt_step) if best_ckpt_step is not None else None,
        "best_ckpt_epoch": int(best_ckpt_epoch) if best_ckpt_epoch is not None else None,
    }
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True, indent=2)
    return out
