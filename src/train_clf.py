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
) -> Dict[str, float]:
    # Backward-compatible aliases (legacy callers may pass synth_ratio/aug_strength).
    if synth_ratio is not None:
        ratio = float(synth_ratio)
    if aug_strength is not None:
        ratio = float(aug_strength)
    ratio = float(ratio)

    x_train_real, y_train_real = load_samples_by_ids(index_df, split["train_ids"])
    x_val, y_val = load_samples_by_ids(index_df, split["val_ids"])
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
    x_test = norm.transform(x_test)

    exp_dir = ensure_dir(out_dir)
    log_path = exp_dir / "log.jsonl"

    # Build augmented-only bank for analysis/logging and append to train set.
    x_added_raw = np.empty((0, x_train_real.shape[1], x_train_real.shape[2]), dtype=np.float32)
    y_added = np.empty((0,), dtype=np.int64)

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
        n_add = int(round(len(x_train_real) * ratio))
        if n_add > 0 and len(sx_raw) > 0:
            replace = len(sx_raw) < n_add
            idx = np.random.choice(len(sx_raw), size=n_add, replace=replace)
            x_added_raw = sx_raw[idx]
            y_added = sy[idx]

    if len(x_added_raw) > 0:
        x_added = norm.transform(x_added_raw)
        x_train = np.concatenate([x_train, x_added], axis=0)
        y_train = np.concatenate([y_train, y_added], axis=0)
        np.savez_compressed(exp_dir / "aug_used.npz", X=x_added_raw, y=y_added)

    model_type = normalize_classifier_type(str(clf_cfg["model"].get("type", "eegnet")))
    alpha_tilde = float(ratio / (1.0 + ratio))
    train_meta = {
        "mode": mode,
        "model_type": model_type,
        "ratio": float(ratio),
        "alpha_tilde": alpha_tilde,
        "n_train_real": int(len(x_train_real)),
        "n_train_aug": int(len(x_added_raw)),
        "n_train_total": int(len(x_train)),
        "batch_size": int(clf_cfg["train"].get("batch_size", 64)),
        "sampling_strategy": "real_plus_aug_concat",
    }

    if is_sklearn_model(model_type):
        svm = build_svm_classifier(clf_cfg["model"])
        svm.fit(x_train, y_train)

        val_metrics = _evaluate_svm(svm, x_val, y_val)
        test_metrics = _evaluate_svm(svm, x_test, y_test)

        append_jsonl(log_path, {"epoch": 1, "train_size": int(len(x_train)), **val_metrics})
        joblib.dump(
            {
                "svm_pipeline": svm.pipeline,
                "normalizer": norm.state_dict(),
                "shape": {"c": int(x_train.shape[1]), "t": int(x_train.shape[2])},
                "n_classes": int(np.max(y_train)) + 1,
                "mode": mode,
                "model_type": model_type,
            },
            exp_dir / "ckpt.pkl",
        )
        train_meta.update(
            {
                "step_mode": "not_applicable_svm",
                "total_steps_target": 0,
                "total_steps_done": 0,
                "steps_per_eval": 0,
            }
        )
        with open(exp_dir / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(train_meta, f, ensure_ascii=True, indent=2)

        out = {
            **test_metrics,
            "val_acc": float(val_metrics["acc"]),
            "val_kappa": float(val_metrics["kappa"]),
            "val_f1_macro": float(val_metrics["f1_macro"]),
            "n_train_real": int(len(x_train_real)),
            "n_train_aug": int(len(x_added_raw)),
            "n_train_total": int(len(x_train)),
            "ratio": float(ratio),
            "alpha_tilde": alpha_tilde,
            "total_steps_target": 0,
            "total_steps_done": 0,
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
        n_classes=int(np.max(y_train)) + 1,
        cfg=clf_cfg["model"],
    ).to(device)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    batch_size = int(clf_cfg["train"].get("batch_size", 64))
    num_workers = int(clf_cfg["train"].get("num_workers", 0))
    tr_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    te_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

    best_val = -1.0
    best_val_metrics = {"acc": 0.0, "kappa": 0.0, "f1_macro": 0.0}
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

            if val_metrics["acc"] > best_val:
                best_val = val_metrics["acc"]
                best_val_metrics = val_metrics
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "normalizer": norm.state_dict(),
                        "shape": {"c": int(x_train.shape[1]), "t": int(x_train.shape[2])},
                        "n_classes": int(np.max(y_train)) + 1,
                        "mode": mode,
                        "model_type": model_type,
                    },
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
            if sched is not None:
                sched.step(float(val_metrics.get("loss", np.mean(loss_meter))))
            append_jsonl(log_path, {"epoch": ep, "train_loss": float(np.mean(loss_meter)), **val_metrics})

            if val_metrics["acc"] > best_val:
                best_val = val_metrics["acc"]
                best_val_metrics = val_metrics
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "normalizer": norm.state_dict(),
                        "shape": {"c": int(x_train.shape[1]), "t": int(x_train.shape[2])},
                        "n_classes": int(np.max(y_train)) + 1,
                        "mode": mode,
                        "model_type": model_type,
                    },
                    exp_dir / "ckpt.pt",
                )

    # Evaluate best checkpoint on fixed test split.
    ckpt = torch.load(exp_dir / "ckpt.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = _evaluate_torch(model, te_dl, device)
    train_meta.update(
        {
            "step_mode": "fixed_steps" if use_fixed_steps else "epoch_loop",
            "total_steps_target": int(total_steps_target),
            "total_steps_done": int(total_steps_done),
            "steps_per_eval": int(steps_per_eval if use_fixed_steps else max(1, len(tr_dl))),
        }
    )
    with open(exp_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, ensure_ascii=True, indent=2)

    out = {
        **test_metrics,
        "val_acc": float(best_val_metrics.get("acc", 0.0)),
        "val_kappa": float(best_val_metrics.get("kappa", 0.0)),
        "val_f1_macro": float(best_val_metrics.get("f1_macro", 0.0)),
        "n_train_real": int(len(x_train_real)),
        "n_train_aug": int(len(x_added_raw)),
        "n_train_total": int(len(x_train)),
        "ratio": float(ratio),
        "alpha_tilde": alpha_tilde,
        "total_steps_target": int(total_steps_target),
        "total_steps_done": int(total_steps_done),
    }
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True, indent=2)
    return out
