from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataio import load_samples_by_ids
from src.eval import compute_metrics
from src.models_clf import build_torch_classifier, normalize_classifier_type
from src.models_gen import (
    CVAE1D,
    CWGANCritic,
    CWGANGenerator,
    ConditionalDDPM1D,
    EEGGANNetDiscriminator,
    EEGGANNetGenerator,
    normalize_generator_type,
)
from src.preprocess import ZScoreNormalizer
from src.qc import run_qc
from src.sample_gen import sample_by_class
from src.utils import append_jsonl, ensure_dir, set_seed


SaveEpochCallback = Callable[[int, dict[str, Any], dict[str, float]], None]


def _resolve_device(cfg: Dict) -> torch.device:
    dev = str(cfg["train"].get("device", "auto"))
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def _make_loader(x_train: np.ndarray, y_train: np.ndarray, batch_size: int, num_workers: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def _model_section(cfg: Dict, model_type: str) -> Dict:
    sec = cfg.get("model", {}).get(model_type, {})
    if isinstance(sec, dict) and sec:
        return sec
    # Backward compatibility for flat keys.
    return cfg.get("model", {})


def _train_cvae(
    dl: DataLoader,
    cfg: Dict,
    in_channels: int,
    time_steps: int,
    num_classes: int,
    device: torch.device,
    log_path: Path,
    save_epoch_cb: SaveEpochCallback | None = None,
) -> tuple[dict, dict]:
    mcfg = _model_section(cfg, "cvae")
    model = CVAE1D(
        in_channels=in_channels,
        time_steps=time_steps,
        num_classes=num_classes,
        latent_dim=int(mcfg.get("latent_dim", 64)),
        hidden_dim=int(mcfg.get("hidden_dim", 128)),
        cond_dim=int(mcfg.get("cond_dim", 16)),
    ).to(device)

    tcfg = cfg["train"]
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(tcfg.get("cvae_lr", tcfg.get("lr", 1e-3))),
        weight_decay=float(tcfg.get("weight_decay", 1e-5)),
    )

    epochs = int(tcfg.get("epochs", 50))
    beta_kl = float(tcfg.get("beta_kl", 1e-3))

    for ep in range(1, epochs + 1):
        model.train()
        loss_meter = []
        recon_v = 0.0
        kl_v = 0.0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model.loss(xb, yb, beta_kl=beta_kl)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()

            loss_meter.append(float(out["loss"].item()))
            recon_v = float(out["recon"].item())
            kl_v = float(out["kl"].item())

        log_rec = {"epoch": ep, "loss": float(np.mean(loss_meter)), "recon": recon_v, "kl": kl_v}
        append_jsonl(log_path, log_rec)
        if save_epoch_cb is not None:
            save_epoch_cb(ep, {"state_dict": model.state_dict()}, log_rec)

    model_kwargs = {
        "in_channels": in_channels,
        "time_steps": time_steps,
        "num_classes": num_classes,
        "latent_dim": int(mcfg.get("latent_dim", 64)),
        "hidden_dim": int(mcfg.get("hidden_dim", 128)),
        "cond_dim": int(mcfg.get("cond_dim", 16)),
    }
    return {"state_dict": model.state_dict()}, model_kwargs


def _train_eeggan(
    dl: DataLoader,
    cfg: Dict,
    in_channels: int,
    time_steps: int,
    num_classes: int,
    device: torch.device,
    log_path: Path,
    save_epoch_cb: SaveEpochCallback | None = None,
) -> tuple[dict, dict]:
    mcfg = _model_section(cfg, "eeggan_net")
    latent_dim = int(mcfg.get("latent_dim", 128))
    base_channels = int(mcfg.get("base_channels", 64))
    cond_dim = int(mcfg.get("cond_dim", 16))

    gen = EEGGANNetGenerator(
        in_channels=in_channels,
        time_steps=time_steps,
        num_classes=num_classes,
        latent_dim=latent_dim,
        base_channels=base_channels,
        cond_dim=cond_dim,
    ).to(device)
    dis = EEGGANNetDiscriminator(
        in_channels=in_channels,
        time_steps=time_steps,
        num_classes=num_classes,
        base_channels=base_channels,
        cond_dim=cond_dim,
    ).to(device)

    tcfg = cfg["train"]
    lr = float(tcfg.get("gan_lr", tcfg.get("lr", 2e-4)))
    wd = float(tcfg.get("weight_decay", 0.0))
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    opt_d = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    bce = torch.nn.BCEWithLogitsLoss()

    epochs = int(tcfg.get("epochs", 50))
    d_steps = int(tcfg.get("d_steps", 1))

    for ep in range(1, epochs + 1):
        gen.train()
        dis.train()
        g_losses = []
        d_losses = []

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            bs = xb.shape[0]

            for _ in range(d_steps):
                z = torch.randn(bs, latent_dim, device=device)
                fake = gen(z, yb).detach()

                real_logits = dis(xb, yb)
                fake_logits = dis(fake, yb)

                real_target = torch.ones_like(real_logits)
                fake_target = torch.zeros_like(fake_logits)

                d_loss = 0.5 * (bce(real_logits, real_target) + bce(fake_logits, fake_target))
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            z = torch.randn(bs, latent_dim, device=device)
            fake = gen(z, yb)
            g_logits = dis(fake, yb)
            g_target = torch.ones_like(g_logits)
            g_loss = bce(g_logits, g_target)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_losses.append(float(g_loss.item()))
            d_losses.append(float(d_loss.item()))

        log_rec = {"epoch": ep, "g_loss": float(np.mean(g_losses)), "d_loss": float(np.mean(d_losses))}
        append_jsonl(log_path, log_rec)
        if save_epoch_cb is not None:
            save_epoch_cb(
                ep,
                {"generator_state_dict": gen.state_dict(), "discriminator_state_dict": dis.state_dict()},
                log_rec,
            )

    model_kwargs = {
        "in_channels": in_channels,
        "time_steps": time_steps,
        "num_classes": num_classes,
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "cond_dim": cond_dim,
    }
    return {"generator_state_dict": gen.state_dict(), "discriminator_state_dict": dis.state_dict()}, model_kwargs


def _gradient_penalty(critic: CWGANCritic, real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor, device: torch.device) -> torch.Tensor:
    eps = torch.rand(real.shape[0], 1, 1, device=device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)
    score = critic(interp, y)
    grad = torch.autograd.grad(outputs=score.sum(), inputs=interp, create_graph=True, retain_graph=True)[0]
    grad = grad.view(real.shape[0], -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def _train_cwgan_gp(
    dl: DataLoader,
    cfg: Dict,
    in_channels: int,
    time_steps: int,
    num_classes: int,
    device: torch.device,
    log_path: Path,
    save_epoch_cb: SaveEpochCallback | None = None,
) -> tuple[dict, dict]:
    mcfg = _model_section(cfg, "cwgan_gp")
    latent_dim = int(mcfg.get("latent_dim", 128))
    base_channels = int(mcfg.get("base_channels", 64))
    cond_dim = int(mcfg.get("cond_dim", 16))

    gen = CWGANGenerator(
        in_channels=in_channels,
        time_steps=time_steps,
        num_classes=num_classes,
        latent_dim=latent_dim,
        base_channels=base_channels,
        cond_dim=cond_dim,
    ).to(device)
    critic = CWGANCritic(
        in_channels=in_channels,
        time_steps=time_steps,
        num_classes=num_classes,
        base_channels=base_channels,
        cond_dim=cond_dim,
    ).to(device)

    tcfg = cfg["train"]
    lr = float(tcfg.get("gan_lr", tcfg.get("lr", 2e-4)))
    wd = float(tcfg.get("weight_decay", 0.0))
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=wd)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=wd)

    epochs = int(tcfg.get("epochs", 50))
    n_critic = int(tcfg.get("n_critic", 3))
    lambda_gp = float(tcfg.get("lambda_gp", 10.0))

    step = 0
    for ep in range(1, epochs + 1):
        gen.train()
        critic.train()
        g_losses = []
        c_losses = []

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            bs = xb.shape[0]

            z = torch.randn(bs, latent_dim, device=device)
            fake = gen(z, yb)

            c_real = critic(xb, yb).mean()
            c_fake = critic(fake.detach(), yb).mean()
            gp = _gradient_penalty(critic, xb, fake.detach(), yb, device)
            c_loss = c_fake - c_real + lambda_gp * gp
            opt_c.zero_grad()
            c_loss.backward()
            opt_c.step()

            step += 1
            g_loss = torch.tensor(0.0, device=device)
            if step % n_critic == 0:
                z = torch.randn(bs, latent_dim, device=device)
                fake = gen(z, yb)
                g_loss = -critic(fake, yb).mean()
                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

            c_losses.append(float(c_loss.item()))
            g_losses.append(float(g_loss.item()))

        log_rec = {"epoch": ep, "g_loss": float(np.mean(g_losses)), "c_loss": float(np.mean(c_losses))}
        append_jsonl(log_path, log_rec)
        if save_epoch_cb is not None:
            save_epoch_cb(ep, {"generator_state_dict": gen.state_dict(), "critic_state_dict": critic.state_dict()}, log_rec)

    model_kwargs = {
        "in_channels": in_channels,
        "time_steps": time_steps,
        "num_classes": num_classes,
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "cond_dim": cond_dim,
    }
    return {"generator_state_dict": gen.state_dict(), "critic_state_dict": critic.state_dict()}, model_kwargs


def _train_conditional_ddpm(
    dl: DataLoader,
    cfg: Dict,
    in_channels: int,
    time_steps: int,
    num_classes: int,
    device: torch.device,
    log_path: Path,
    save_epoch_cb: SaveEpochCallback | None = None,
) -> tuple[dict, dict]:
    mcfg = _model_section(cfg, "conditional_ddpm")
    base_channels = int(mcfg.get("base_channels", 64))
    time_dim = int(mcfg.get("time_dim", 128))
    diffusion_steps = int(mcfg.get("diffusion_steps", 200))
    beta_start = float(mcfg.get("beta_start", 1e-4))
    beta_end = float(mcfg.get("beta_end", 0.02))

    model = ConditionalDDPM1D(
        in_channels=in_channels,
        time_steps=time_steps,
        num_classes=num_classes,
        base_channels=base_channels,
        time_dim=time_dim,
        diffusion_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    ).to(device)

    tcfg = cfg["train"]
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(tcfg.get("ddpm_lr", tcfg.get("lr", 1e-4))),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    epochs = int(tcfg.get("epochs", 50))

    for ep in range(1, epochs + 1):
        model.train()
        loss_meter = []

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model.loss(xb, yb)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            loss_meter.append(float(out["loss"].item()))

        log_rec = {"epoch": ep, "loss": float(np.mean(loss_meter))}
        append_jsonl(log_path, log_rec)
        if save_epoch_cb is not None:
            save_epoch_cb(ep, {"state_dict": model.state_dict()}, log_rec)

    model_kwargs = {
        "in_channels": in_channels,
        "time_steps": time_steps,
        "num_classes": num_classes,
        "base_channels": base_channels,
        "time_dim": time_dim,
        "diffusion_steps": diffusion_steps,
        "beta_start": beta_start,
        "beta_end": beta_end,
    }
    return {"state_dict": model.state_dict()}, model_kwargs


def _build_ckpt_payload(
    train_state: dict[str, Any],
    model_type: str,
    model_kwargs: dict[str, Any],
    norm: ZScoreNormalizer,
    in_channels: int,
    time_steps: int,
    num_classes: int,
    gen_cfg: Dict,
    epoch: int | None = None,
) -> dict[str, Any]:
    payload = {
        **train_state,
        "model_type": model_type,
        "model_kwargs": model_kwargs,
        "normalizer": norm.state_dict(),
        "shape": {"c": in_channels, "t": time_steps},
        "num_classes": num_classes,
        "gen_cfg": gen_cfg,
    }
    if epoch is not None:
        payload["epoch"] = int(epoch)
    return payload


def _quick_proxy_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    clf_cfg: Dict,
    proxy_cfg: Dict,
    metric_key: str,
    seed: int,
) -> tuple[float, dict[str, float]]:
    set_seed(seed)

    proxy_model = normalize_classifier_type(str(proxy_cfg.get("model_type", "eegnet_tf_faithful")))
    batch_size = int(proxy_cfg.get("batch_size", 64))
    total_steps = int(proxy_cfg.get("steps", 120))
    lr = float(proxy_cfg.get("lr", 1e-3))

    dev = str(clf_cfg.get("train", {}).get("device", "auto"))
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    proxy_model_cfg = clf_cfg.get("proxy_model", clf_cfg.get("model", {}))
    model = build_torch_classifier(
        model_type=proxy_model,
        n_ch=int(x_train.shape[1]),
        n_t=int(x_train.shape[2]),
        n_classes=int(np.max(y_train)) + 1,
        cfg=proxy_model_cfg,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    itr = iter(train_dl)
    steps_done = 0
    while steps_done < total_steps:
        try:
            xb, yb = next(itr)
        except StopIteration:
            itr = iter(train_dl)
            xb, yb = next(itr)

        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps_done += 1

    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(x_val).to(device)
        pred = model(xv).argmax(dim=1).cpu().numpy()
    metrics = compute_metrics(y_val, pred)
    score = float(metrics.get(metric_key, metrics.get("bal_acc", metrics.get("acc", 0.0))))
    return score, metrics


def _select_best_checkpoint(
    ckpt_candidates: list[dict[str, Any]],
    x_train_real: np.ndarray,
    y_train_real: np.ndarray,
    x_train_norm: np.ndarray,
    x_val_norm: np.ndarray,
    y_val: np.ndarray,
    norm: ZScoreNormalizer,
    qc_cfg: Dict | None,
    gen_cfg: Dict,
    clf_cfg: Dict,
    num_classes: int,
    device: str,
    base_seed: int,
    out_dir: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    sel_cfg = gen_cfg.get("checkpoint_selection", {})
    metric_key = str(sel_cfg.get("metric", "bal_acc"))
    n_per_class = int(sel_cfg.get("sample_n_per_class", 80))
    proxy_cfg = sel_cfg.get("proxy_classifier", {})
    qc_on = bool(sel_cfg.get("qc_enabled", True)) and qc_cfg is not None
    seed_offset = int(sel_cfg.get("seed_offset", 10000))

    scores: list[dict[str, Any]] = []
    best_key: tuple[float, float, int] | None = None
    best_path: Path | None = None

    for i, c in enumerate(ckpt_candidates):
        ckpt_path = Path(c["path"])
        eval_seed = base_seed + seed_offset + i
        set_seed(eval_seed)

        synth = sample_by_class(
            ckpt_path=ckpt_path,
            n_per_class=n_per_class,
            num_classes=num_classes,
            device=device,
        )

        if qc_on:
            kept, qc_report = run_qc(
                real_x=x_train_real,
                synth=synth,
                sfreq=int(gen_cfg.get("data", {}).get("sfreq", 250)),
                cfg=qc_cfg,
                real_y=y_train_real,
            )
        else:
            kept = synth
            qc_report = {
                "n_before": int(synth["X"].shape[0]),
                "n_after": int(synth["X"].shape[0]),
                "keep_ratio": 1.0,
                "target_keep_ratio": 1.0,
                "fallback_applied": False,
            }

        n_after = int(kept["X"].shape[0])
        if n_after <= 0:
            rec = {
                "epoch": int(c.get("epoch", -1)),
                "ckpt_path": str(ckpt_path),
                "eval_seed": int(eval_seed),
                "proxy_metric": metric_key,
                "proxy_score": float("-inf"),
                "qc_keep_ratio": float(qc_report.get("keep_ratio", 0.0)),
                "n_synth_after_qc": n_after,
                "selected": False,
            }
            scores.append(rec)
            continue

        x_aug = norm.transform(kept["X"].astype(np.float32))
        y_aug = kept["y"].astype(np.int64)

        x_mix = np.concatenate([x_train_norm, x_aug], axis=0)
        y_mix = np.concatenate([y_train_real.astype(np.int64), y_aug], axis=0)

        score, metrics = _quick_proxy_eval(
            x_train=x_mix,
            y_train=y_mix,
            x_val=x_val_norm,
            y_val=y_val.astype(np.int64),
            clf_cfg=clf_cfg,
            proxy_cfg=proxy_cfg,
            metric_key=metric_key,
            seed=eval_seed,
        )

        rec = {
            "epoch": int(c.get("epoch", -1)),
            "ckpt_path": str(ckpt_path),
            "eval_seed": int(eval_seed),
            "proxy_metric": metric_key,
            "proxy_score": float(score),
            "proxy_acc": float(metrics.get("acc", np.nan)),
            "proxy_bal_acc": float(metrics.get("bal_acc", np.nan)),
            "proxy_kappa": float(metrics.get("kappa", np.nan)),
            "proxy_f1_macro": float(metrics.get("f1_macro", np.nan)),
            "qc_keep_ratio": float(qc_report.get("keep_ratio", 0.0)),
            "n_synth_before_qc": int(qc_report.get("n_before", synth["X"].shape[0])),
            "n_synth_after_qc": n_after,
            "selected": False,
        }
        scores.append(rec)

        key = (float(score), float(rec["qc_keep_ratio"]), int(c.get("epoch", -1)))
        if best_key is None or key > best_key:
            best_key = key
            best_path = ckpt_path

    if best_path is None:
        best_path = Path(ckpt_candidates[-1]["path"])

    for r in scores:
        r["selected"] = Path(r["ckpt_path"]) == best_path

    with open(out_dir / "ckpt_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=True, indent=2)

    return best_path, scores


def train_generative_model(
    split: Dict,
    index_df: pd.DataFrame,
    gen_cfg: Dict,
    preprocess_cfg: Dict,
    out_dir: str | Path,
    qc_cfg: Dict | None = None,
    clf_cfg: Dict | None = None,
    base_seed: int = 0,
) -> Path:
    x_train_real, y_train = load_samples_by_ids(index_df, split["train_ids"])
    x_val_real, y_val = load_samples_by_ids(index_df, split["val_ids"])

    norm = ZScoreNormalizer(
        eps=float(preprocess_cfg["normalization"].get("eps", 1e-6)),
        mode=str(preprocess_cfg["normalization"].get("mode", "channel_global")),
    ).fit(x_train_real)
    x_train = norm.transform(x_train_real)
    x_val = norm.transform(x_val_real)

    device_t = _resolve_device(gen_cfg)
    device = str(device_t)
    model_type = normalize_generator_type(str(gen_cfg["model"].get("type", "cvae")))

    batch_size = int(gen_cfg["train"].get("batch_size", 32))
    num_workers = int(gen_cfg["train"].get("num_workers", 0))
    dl = _make_loader(x_train, y_train, batch_size=batch_size, num_workers=num_workers)

    exp_dir = ensure_dir(out_dir)
    log_path = exp_dir / "log.jsonl"
    ckpt_dir = ensure_dir(exp_dir / "checkpoints")

    in_channels = int(x_train.shape[1])
    time_steps = int(x_train.shape[2])
    num_classes = int(np.max(y_train)) + 1

    epochs = int(gen_cfg["train"].get("epochs", 50))
    checkpoint_every = int(gen_cfg["train"].get("checkpoint_every", max(1, epochs // 5)))
    checkpoint_every = max(1, checkpoint_every)

    ckpt_candidates: list[dict[str, Any]] = []
    model_kwargs_ref: dict[str, Any] = {}

    def save_epoch_cb(epoch: int, train_state: dict[str, Any], train_log: dict[str, float]) -> None:
        if epoch % checkpoint_every != 0 and epoch != epochs:
            return

        payload = _build_ckpt_payload(
            train_state=train_state,
            model_type=model_type,
            model_kwargs=model_kwargs_ref,
            norm=norm,
            in_channels=in_channels,
            time_steps=time_steps,
            num_classes=num_classes,
            gen_cfg=gen_cfg,
            epoch=epoch,
        )
        p = ckpt_dir / f"epoch_{epoch:04d}.pt"
        torch.save(payload, p)
        ckpt_candidates.append(
            {
                "epoch": int(epoch),
                "path": str(p),
                "log": {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in train_log.items()},
            }
        )

    if model_type == "cvae":
        train_state, model_kwargs = _train_cvae(
            dl,
            gen_cfg,
            in_channels,
            time_steps,
            num_classes,
            device_t,
            log_path,
            save_epoch_cb=save_epoch_cb,
        )
    elif model_type == "eeggan_net":
        train_state, model_kwargs = _train_eeggan(
            dl,
            gen_cfg,
            in_channels,
            time_steps,
            num_classes,
            device_t,
            log_path,
            save_epoch_cb=save_epoch_cb,
        )
    elif model_type == "cwgan_gp":
        train_state, model_kwargs = _train_cwgan_gp(
            dl,
            gen_cfg,
            in_channels,
            time_steps,
            num_classes,
            device_t,
            log_path,
            save_epoch_cb=save_epoch_cb,
        )
    elif model_type == "conditional_ddpm":
        train_state, model_kwargs = _train_conditional_ddpm(
            dl,
            gen_cfg,
            in_channels,
            time_steps,
            num_classes,
            device_t,
            log_path,
            save_epoch_cb=save_epoch_cb,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_kwargs_ref.update(model_kwargs)

    # Ensure final epoch checkpoint exists even when callback fired before kwargs were populated.
    final_payload = _build_ckpt_payload(
        train_state=train_state,
        model_type=model_type,
        model_kwargs=model_kwargs,
        norm=norm,
        in_channels=in_channels,
        time_steps=time_steps,
        num_classes=num_classes,
        gen_cfg=gen_cfg,
        epoch=epochs,
    )
    final_epoch_ckpt = ckpt_dir / f"epoch_{epochs:04d}.pt"
    torch.save(final_payload, final_epoch_ckpt)
    if not any(Path(c["path"]) == final_epoch_ckpt for c in ckpt_candidates):
        ckpt_candidates.append({"epoch": epochs, "path": str(final_epoch_ckpt), "log": {"epoch": epochs}})

    # Refresh candidate payloads to include model_kwargs metadata.
    refreshed_candidates: list[dict[str, Any]] = []
    for c in ckpt_candidates:
        cp = Path(c["path"])
        old = torch.load(cp, map_location="cpu", weights_only=False)
        payload = {
            **old,
            "model_type": model_type,
            "model_kwargs": model_kwargs,
            "normalizer": norm.state_dict(),
            "shape": {"c": in_channels, "t": time_steps},
            "num_classes": num_classes,
            "gen_cfg": gen_cfg,
            "epoch": int(c.get("epoch", old.get("epoch", -1))),
        }
        torch.save(payload, cp)
        refreshed_candidates.append({"epoch": int(payload["epoch"]), "path": str(cp), "log": c.get("log", {})})
    ckpt_candidates = sorted(refreshed_candidates, key=lambda x: int(x["epoch"]))

    with open(exp_dir / "ckpt_list.json", "w", encoding="utf-8") as f:
        json.dump(ckpt_candidates, f, ensure_ascii=True, indent=2)

    sel_cfg = gen_cfg.get("checkpoint_selection", {})
    do_select = bool(sel_cfg.get("enabled", True)) and clf_cfg is not None and len(split.get("val_ids", [])) > 0

    if do_select:
        best_ckpt, _ = _select_best_checkpoint(
            ckpt_candidates=ckpt_candidates,
            x_train_real=x_train_real,
            y_train_real=y_train.astype(np.int64),
            x_train_norm=x_train,
            x_val_norm=x_val,
            y_val=y_val,
            norm=norm,
            qc_cfg=qc_cfg,
            gen_cfg=gen_cfg,
            clf_cfg=clf_cfg,
            num_classes=num_classes,
            device=device,
            base_seed=int(base_seed),
            out_dir=exp_dir,
        )
        selected_by = "t_val_proxy"
    else:
        best_ckpt = Path(ckpt_candidates[-1]["path"])
        selected_by = "final_epoch"

    final_ckpt = exp_dir / "ckpt.pt"
    shutil.copy2(best_ckpt, final_ckpt)

    train_meta = {
        "model_type": model_type,
        "selected_by": selected_by,
        "best_ckpt_path": str(best_ckpt),
        "final_ckpt_path": str(final_ckpt),
        "checkpoint_every": int(checkpoint_every),
        "num_candidates": int(len(ckpt_candidates)),
        "base_seed": int(base_seed),
        "proxy_metric": str(sel_cfg.get("metric", "bal_acc")),
        "proxy_sample_n_per_class": int(sel_cfg.get("sample_n_per_class", 80)),
        "proxy_qc_enabled": bool(sel_cfg.get("qc_enabled", True)),
    }
    with open(exp_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, ensure_ascii=True, indent=2)

    return exp_dir
