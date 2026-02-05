from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.augment.mixup import mixup_batch
from src.augment.traditional import apply_traditional_augment
from src.eval.metrics import compute_metrics
from src.models.classifiers.base import build_classifier, is_sklearn_model, normalize_classifier_type
from src.utils.alpha import alpha_ratio_to_mix
from src.utils.io import ensure_dir, write_json


def _build_optimizer(tcfg: Dict, params) -> torch.optim.Optimizer:
    """Build an optimizer from train config.

    Inputs:
    - tcfg: train config dict (optimizer, lr, weight_decay, etc.).
    - params: model parameters.

    Outputs:
    - torch optimizer instance.

    Internal logic:
    - Selects optimizer type (Adam/AdamW/SGD) and applies hyperparameters.
    """
    opt_name = str(tcfg.get("optimizer", "adam")).lower()
    lr = float(tcfg.get("lr", 1e-3))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = float(tcfg.get("momentum", 0.0))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _apply_cuda_speed(train_cfg: Dict, device: torch.device) -> None:
    """Apply CUDA speed settings (TF32, matmul precision).

    Inputs:
    - train_cfg: training config dict.
    - device: torch.device.

    Outputs:
    - None (modifies torch backend flags).

    Internal logic:
    - Enables TF32 and matmul precision hints when running on CUDA.
    """
    if device.type != "cuda":
        return
    cuda_cfg = train_cfg.get("cuda", {}) if isinstance(train_cfg, dict) else {}
    if bool(cuda_cfg.get("tf32", False)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    matmul_precision = cuda_cfg.get("matmul_precision")
    if matmul_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(matmul_precision))


def _soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy with soft labels.

    Inputs:
    - logits: torch.Tensor [B, K]
    - target: torch.Tensor [B, K] soft labels

    Outputs:
    - scalar loss tensor.

    Internal logic:
    - Applies log-softmax then computes mean negative log-likelihood.
    """
    log_probs = torch.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def _evaluate_torch(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> dict[str, float]:
    """Evaluate a torch model on numpy data.

    Inputs:
    - model: torch.nn.Module
    - x: ndarray [N, C, T]
    - y: ndarray [N]
    - device: torch.device

    Outputs:
    - metrics dict (acc/kappa/macro_f1).

    Internal logic:
    - Runs batched inference on device and computes metrics from predictions.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for st in range(0, len(x), 256):
            xb = torch.from_numpy(x[st : st + 256]).to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)
    pred = np.concatenate(preds, axis=0)
    return compute_metrics(y, pred)


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    model_cfg: dict,
    train_cfg: dict,
    eval_cfg: dict,
    method: str,
    alpha_ratio: float,
    num_classes: int,
    run_dir: Path,
    normalizer_state: dict,
    synth_data: tuple[np.ndarray, np.ndarray] | None = None,
    evaluate_test: bool = False,
    aug_cfg: dict | None = None,
) -> dict:
    """Train a classifier (torch or sklearn) with optional augmentation.

    Inputs:
    - x_train/y_train: training data [N, C, T] / [N]
    - x_val/y_val: validation data [M, C, T] / [M]
    - x_test/y_test: test data [Q, C, T] / [Q]
    - model_type/model_cfg: classifier selection and hyperparams
    - train_cfg/eval_cfg: training and selection configs
    - method: C0/C1/C2/GenAug
    - alpha_ratio: synth:real ratio (logged)
    - num_classes: number of classes
    - run_dir: output directory for metrics/ckpt
    - normalizer_state: fitted normalizer state
    - synth_data: optional (X_syn, y_syn)
    - evaluate_test: whether to compute test metrics
    - aug_cfg: traditional augmentation params

    Outputs:
    - metrics dict with validation/test scores and effective ratios.

    Internal logic:
    - Builds augmented dataset (GenAug or traditional or mixup) and trains.
    - Uses early stopping on validation and saves best checkpoint.
    """
    ensure_dir(run_dir)
    model_type = normalize_classifier_type(model_type)

    # Augmentation strategy
    x_aug = x_train  # augmented training data (initialized to real)
    y_aug = y_train  # augmented labels
    n_real = int(len(x_train))  # number of real samples
    n_syn = 0  # number of synthetic samples

    if method == "GenAug" and synth_data is not None:
        x_syn, y_syn = synth_data
        if len(x_syn) > 0:
            x_aug = np.concatenate([x_train, x_syn], axis=0)
            y_aug = np.concatenate([y_train, y_syn], axis=0)
            n_syn = int(len(x_syn))

    ratio_effective = float(n_syn / max(1, n_real))  # actual synth:real ratio
    alpha_mix_effective = alpha_ratio_to_mix(ratio_effective)  # mixture weight

    if is_sklearn_model(model_type):
        model = build_classifier(model_type, x_train.shape[1], x_train.shape[2], num_classes, model_cfg)
        if method == "C1":
            x_aug = apply_traditional_augment(
                x_aug,
                noise_std=float(aug_cfg.get("noise_std", 0.01)),
                max_time_shift=int(aug_cfg.get("max_time_shift", 20)),
                channel_dropout_prob=float(aug_cfg.get("channel_dropout_prob", 0.1)),
            )
        if method == "C2":
            x_mix, y_mix = mixup_batch(x_aug, y_aug, alpha=mixup_alpha, num_classes=num_classes)
            y_aug = np.argmax(y_mix, axis=1).astype(np.int64)
            x_aug = x_mix.astype(np.float32)
        model.fit(x_aug, y_aug)
        val_metrics = compute_metrics(y_val, model.predict(x_val))
        test_metrics = compute_metrics(y_test, model.predict(x_test)) if evaluate_test else {}
        metrics = {
            "val_acc": val_metrics["acc"],
            "val_kappa": val_metrics["kappa"],
            "val_macro_f1": val_metrics["macro_f1"],
            "acc": test_metrics.get("acc", np.nan),
            "kappa": test_metrics.get("kappa", np.nan),
            "macro_f1": test_metrics.get("macro_f1", np.nan),
            "ratio_effective": ratio_effective,
            "alpha_mix_effective": alpha_mix_effective,
        }
        write_json(run_dir / "metrics.json", metrics)
        return metrics

    device = torch.device("cuda" if (train_cfg.get("device", "auto") == "auto" and torch.cuda.is_available()) else train_cfg.get("device", "cpu"))
    _apply_cuda_speed(train_cfg, device)

    model = build_classifier(model_type, x_train.shape[1], x_train.shape[2], num_classes, model_cfg).to(device)

    batch_size = int(train_cfg.get("batch_size", 64))  # training batch size
    num_workers = int(train_cfg.get("num_workers", 0))  # dataloader workers

    def _seed_worker(worker_id: int) -> None:
        """Seed a DataLoader worker for deterministic behavior.

        Inputs:
        - worker_id: integer worker index.

        Outputs:
        - None (sets NumPy seed for worker).

        Internal logic:
        - Uses torch.initial_seed to derive a per-worker NumPy seed.
        """
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed + worker_id)

    ds = TensorDataset(torch.from_numpy(x_aug), torch.from_numpy(y_aug))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    opt = _build_optimizer(train_cfg, model.parameters())
    best_metric = float("-inf")  # track best validation score
    best_state = None  # model state for best validation
    best_metrics = {}  # cached best validation metrics

    step_cfg = train_cfg.get("step_control", {})  # optional step-based training
    use_step_control = bool(step_cfg.get("enabled", False))

    total_steps = int(step_cfg.get("total_steps", 0))  # total training steps (if enabled)
    steps_per_eval = int(step_cfg.get("steps_per_eval", 100))  # eval cadence

    loss_fn = nn.CrossEntropyLoss()

    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.2))  # Beta alpha for mixup
    aug_cfg = aug_cfg or {}

    def _eval_and_update() -> None:
        """Evaluate on validation data and update best checkpoint state.

        Inputs:
        - None (uses closure variables for model and eval_cfg).

        Outputs:
        - None (updates nonlocal best_metric/state/metrics).

        Internal logic:
        - Computes validation metrics and saves state if improved.
        """
        nonlocal best_metric, best_state, best_metrics
        val_metrics = _evaluate_torch(model, x_val, y_val, device)
        metric_key = str(eval_cfg.get("best_metric", "kappa"))
        direction = str(eval_cfg.get("best_direction", "max"))
        score = float(val_metrics.get(metric_key, val_metrics.get("kappa", 0.0)))
        improved = score > best_metric if direction == "max" else score < best_metric
        if improved:
            best_metric = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = val_metrics

    if use_step_control and total_steps > 0:
        step = 0
        data_iter = iter(dl)
        while step < total_steps:
            try:
                xb, yb = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                xb, yb = next(data_iter)

            xb = xb.to(device)
            yb = yb.to(device)

            if method == "C1":
                xb = torch.from_numpy(
                    apply_traditional_augment(
                        xb.cpu().numpy(),
                        noise_std=float(aug_cfg.get("noise_std", 0.01)),
                        max_time_shift=int(aug_cfg.get("max_time_shift", 20)),
                        channel_dropout_prob=float(aug_cfg.get("channel_dropout_prob", 0.1)),
                    )
                ).to(device)

            if method == "C2":
                x_mix, y_mix = mixup_batch(
                    xb.cpu().numpy(),
                    yb.cpu().numpy(),
                    alpha=mixup_alpha,
                    num_classes=num_classes,
                )
                xb = torch.from_numpy(x_mix).to(device)
                y_soft = torch.from_numpy(y_mix).to(device)
                logits = model(xb)
                loss = _soft_cross_entropy(logits, y_soft)
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if step % steps_per_eval == 0:
                _eval_and_update()
    else:
        epochs = int(train_cfg.get("epochs", 80))
        for _ in range(epochs):
            model.train()
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)

                if method == "C1":
                    xb = torch.from_numpy(
                        apply_traditional_augment(
                            xb.cpu().numpy(),
                            noise_std=float(aug_cfg.get("noise_std", 0.01)),
                            max_time_shift=int(aug_cfg.get("max_time_shift", 20)),
                            channel_dropout_prob=float(aug_cfg.get("channel_dropout_prob", 0.1)),
                        )
                    ).to(device)

                if method == "C2":
                    x_mix, y_mix = mixup_batch(
                        xb.cpu().numpy(),
                        yb.cpu().numpy(),
                        alpha=mixup_alpha,
                        num_classes=num_classes,
                    )
                    xb = torch.from_numpy(x_mix).to(device)
                    y_soft = torch.from_numpy(y_mix).to(device)
                    logits = model(xb)
                    loss = _soft_cross_entropy(logits, y_soft)
                else:
                    logits = model(xb)
                    loss = loss_fn(logits, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()

            _eval_and_update()

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = _evaluate_torch(model, x_val, y_val, device)
    test_metrics = _evaluate_torch(model, x_test, y_test, device) if evaluate_test else {}

    ckpt = {
        "model_type": model_type,
        "state_dict": model.state_dict(),
        "normalizer": normalizer_state,
        "shape": {"c": int(x_train.shape[1]), "t": int(x_train.shape[2])},
        "n_classes": int(num_classes),
        "model_cfg": model_cfg,
    }
    torch.save(ckpt, run_dir / "ckpt.pt")

    metrics = {
        "val_acc": val_metrics["acc"],
        "val_kappa": val_metrics["kappa"],
        "val_macro_f1": val_metrics["macro_f1"],
        "acc": test_metrics.get("acc", np.nan),
        "kappa": test_metrics.get("kappa", np.nan),
        "macro_f1": test_metrics.get("macro_f1", np.nan),
        "ratio_effective": ratio_effective,
        "alpha_mix_effective": alpha_mix_effective,
    }
    write_json(run_dir / "metrics.json", metrics)
    return metrics
