from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataio import load_samples_by_ids
from src.eval import compute_metrics
from src.models_official_faithful import ATCNetOfficialFaithful, build_faithful_model
from src.preprocess import ZScoreNormalizer
from src.utils import append_jsonl, ensure_dir, resolve_device


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    ce: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ys, ps, losses = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            if isinstance(model, ATCNetOfficialFaithful):
                loss = loss + model.regularization_loss()
            losses.append(float(loss.item()))
            ys.append(yb.cpu().numpy())
            ps.append(logits.argmax(dim=1).cpu().numpy())
    out = compute_metrics(np.concatenate(ys), np.concatenate(ps))
    out["loss"] = float(np.mean(losses))
    return out


def train_faithful_classifier(
    split: Dict,
    index_df: pd.DataFrame,
    model_key: str,
    model_cfg: dict,
    train_cfg: dict,
    preprocess_cfg: dict,
    out_dir: str | Path,
) -> Dict[str, float]:
    x_train, y_train = load_samples_by_ids(index_df, split["train_ids"])
    x_val, y_val = load_samples_by_ids(index_df, split["val_ids"])
    x_test, y_test = load_samples_by_ids(index_df, split["test_ids"])

    norm = ZScoreNormalizer(
        eps=float(preprocess_cfg["eps"]),
        mode=str(preprocess_cfg["mode"]),
    ).fit(x_train)
    x_train = norm.transform(x_train)
    x_val = norm.transform(x_val)
    x_test = norm.transform(x_test)

    device = resolve_device(train_cfg.get("device", "auto"))

    model = build_faithful_model(
        model_key=model_key,
        n_ch=int(x_train.shape[1]),
        n_t=int(x_train.shape[2]),
        n_classes=int(np.max(y_train)) + 1,
        model_cfg=model_cfg,
    ).to(device)

    tr_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    va_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    te_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    bs = int(train_cfg["batch_size"])
    nw = int(train_cfg.get("num_workers", 0))
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=nw)
    va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=nw)
    te_dl = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=nw)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    sch_cfg = train_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode=str(sch_cfg.get("mode", "min")),
        factor=float(sch_cfg.get("factor", 0.9)),
        patience=int(sch_cfg.get("patience", 20)),
        min_lr=float(sch_cfg.get("min_lr", 1e-4)),
    )
    ce = torch.nn.CrossEntropyLoss()

    exp_dir = ensure_dir(out_dir)
    log_path = exp_dir / "log.jsonl"
    best_val = float("inf")
    best_path = exp_dir / "ckpt.pt"

    for ep in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        train_losses = []
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            if isinstance(model, ATCNetOfficialFaithful):
                loss = loss + model.regularization_loss()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        val = _evaluate(model, va_dl, ce, device)
        scheduler.step(float(val["loss"]))
        append_jsonl(
            log_path,
            {
                "epoch": ep,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(val["loss"]),
                "val_acc": float(val["acc"]),
                "val_kappa": float(val["kappa"]),
            },
        )
        if float(val["loss"]) < best_val:
            best_val = float(val["loss"])
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "normalizer": norm.state_dict(),
                    "shape": {"c": int(x_train.shape[1]), "t": int(x_train.shape[2])},
                    "n_classes": int(np.max(y_train)) + 1,
                    "model_key": model_key,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = _evaluate(model, te_dl, ce, device)
    return {k: float(v) for k, v in test_metrics.items() if k in {"acc", "kappa", "f1_macro"}}
