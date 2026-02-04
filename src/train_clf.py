from __future__ import annotations

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


def _classical_augment_torch(x: torch.Tensor, noise_std: float, max_shift: int) -> torch.Tensor:
    x = x + torch.randn_like(x) * noise_std
    if max_shift > 0:
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        x = torch.roll(x, shifts=shift, dims=-1)
    return x


def _classical_augment_numpy(x: np.ndarray, noise_std: float, max_shift: int) -> np.ndarray:
    noise = np.random.randn(*x.shape).astype(np.float32) * noise_std
    out = x + noise
    if max_shift <= 0:
        return out
    shifts = np.random.randint(-max_shift, max_shift + 1, size=x.shape[0])
    for i, s in enumerate(shifts):
        out[i] = np.roll(out[i], shift=int(s), axis=-1)
    return out


def _mixup_torch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.shape[0], device=x.device)
    xm = lam * x + (1 - lam) * x[idx]
    ya, yb = y, y[idx]
    return xm, ya, yb, lam


def _mixup_numpy_hard(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.permutation(x.shape[0])
    lam = np.random.beta(alpha, alpha, size=x.shape[0]).astype(np.float32)
    lm = lam[:, None, None]
    xm = lm * x + (1.0 - lm) * x[idx]
    ym = np.where(lam >= 0.5, y, y[idx])
    return xm.astype(np.float32), ym.astype(np.int64)


def _seg_reconstruct_augment_numpy(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    n_classes: int,
    batch_size: int,
    n_aug: int,
    n_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segmentation-and-reconstruction augmentation used in several BCI papers.
    Build synthetic trials by concatenating temporal segments from same-class trials.
    """
    if n_aug <= 0 or n_segments <= 1:
        return np.empty((0, x_pool.shape[1], x_pool.shape[2]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    x_pool = x_pool.astype(np.float32)
    y_pool = y_pool.astype(np.int64)
    n_per_class = max(1, int(batch_size // max(1, n_classes)) * int(n_aug))
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
    synth_ratio: float = 0.0,
    synth_npz: Optional[str] = None,
) -> Dict[str, float]:
    x_train, y_train = load_samples_by_ids(index_df, split["train_ids"])
    x_val, y_val = load_samples_by_ids(index_df, split["val_ids"])
    x_test, y_test = load_samples_by_ids(index_df, split["test_ids"])

    # Train-only normalization statistics.
    norm_cfg = preprocess_cfg["normalization"]
    norm = ZScoreNormalizer(
        eps=float(norm_cfg.get("eps", 1e-6)),
        mode=str(norm_cfg.get("mode", "channel_global")),
    ).fit(x_train)
    x_train = norm.transform(x_train)
    x_val = norm.transform(x_val)
    x_test = norm.transform(x_test)

    if mode == "gen_aug" and synth_ratio > 0 and synth_npz is not None:
        synth = np.load(synth_npz)
        sx = norm.transform(synth["X"].astype(np.float32))
        sy = synth["y"].astype(np.int64)
        n_add = int(len(x_train) * synth_ratio)
        if n_add > 0 and len(sx) > 0:
            idx = np.random.choice(len(sx), size=min(n_add, len(sx)), replace=False)
            x_train = np.concatenate([x_train, sx[idx]], axis=0)
            y_train = np.concatenate([y_train, sy[idx]], axis=0)

    model_type = normalize_classifier_type(str(clf_cfg["model"].get("type", "eegnet")))
    exp_dir = ensure_dir(out_dir)
    log_path = exp_dir / "log.jsonl"

    if is_sklearn_model(model_type):
        # SVM uses offline variants for augmentation modes.
        if mode == "classical":
            x_aug = _classical_augment_numpy(
                x_train,
                noise_std=float(clf_cfg["augmentation"]["classical"].get("noise_std", 0.01)),
                max_shift=int(clf_cfg["augmentation"]["classical"].get("max_time_shift", 20)),
            )
            x_train = np.concatenate([x_train, x_aug], axis=0)
            y_train = np.concatenate([y_train, y_train], axis=0)
        elif mode == "mixup":
            alpha = float(clf_cfg["augmentation"]["mixup"].get("alpha", 0.2))
            x_mix, y_mix = _mixup_numpy_hard(x_train, y_train, alpha=alpha)
            x_train = np.concatenate([x_train, x_mix], axis=0)
            y_train = np.concatenate([y_train, y_mix], axis=0)
        elif mode == "paper_sr":
            sr_cfg = clf_cfg["augmentation"].get("paper_sr", {})
            x_aug, y_aug = _seg_reconstruct_augment_numpy(
                x_pool=x_train,
                y_pool=y_train,
                n_classes=int(np.max(y_train)) + 1,
                batch_size=int(clf_cfg["train"].get("batch_size", 64)),
                n_aug=int(sr_cfg.get("n_aug", 1)),
                n_segments=int(sr_cfg.get("n_segments", 8)),
            )
            if len(x_aug) > 0:
                x_train = np.concatenate([x_train, x_aug], axis=0)
                y_train = np.concatenate([y_train, y_aug], axis=0)

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
        return test_metrics

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

    for ep in range(1, int(clf_cfg["train"].get("epochs", 80)) + 1):
        model.train()
        loss_meter = []

        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)

            if mode == "classical":
                xb = _classical_augment_torch(
                    xb,
                    noise_std=float(clf_cfg["augmentation"]["classical"].get("noise_std", 0.01)),
                    max_shift=int(clf_cfg["augmentation"]["classical"].get("max_time_shift", 20)),
                )

            if mode == "mixup":
                alpha = float(clf_cfg["augmentation"]["mixup"].get("alpha", 0.2))
                xb_m, ya, yb_m, lam = _mixup_torch(xb, yb, alpha=alpha)
                logits = model(xb_m)
                loss = lam * ce(logits, ya) + (1 - lam) * ce(logits, yb_m)
            elif mode == "paper_sr":
                sr_cfg = clf_cfg["augmentation"].get("paper_sr", {})
                x_aug_np, y_aug_np = _seg_reconstruct_augment_numpy(
                    x_pool=x_train,
                    y_pool=y_train,
                    n_classes=int(np.max(y_train)) + 1,
                    batch_size=int(clf_cfg["train"].get("batch_size", 64)),
                    n_aug=int(sr_cfg.get("n_aug", 1)),
                    n_segments=int(sr_cfg.get("n_segments", 8)),
                )
                if len(x_aug_np) > 0:
                    x_aug = torch.from_numpy(x_aug_np).to(device)
                    y_aug = torch.from_numpy(y_aug_np).to(device)
                    xb = torch.cat([xb, x_aug], dim=0)
                    yb = torch.cat([yb, y_aug], dim=0)
                logits = model(xb)
                loss = ce(logits, yb)
            else:
                logits = model(xb)
                loss = ce(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_meter.append(float(loss.item()))

        val_metrics = _evaluate_torch(model, va_dl, device, criterion=ce)
        if sched is not None:
            sched.step(float(val_metrics.get("loss", np.mean(loss_meter))))
        append_jsonl(log_path, {"epoch": ep, "train_loss": float(np.mean(loss_meter)), **val_metrics})

        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "normalizer": norm.state_dict(),
                    "shape": {"c": x_train.shape[1], "t": x_train.shape[2]},
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
    return test_metrics
