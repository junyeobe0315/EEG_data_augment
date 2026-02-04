from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataio import load_samples_by_ids
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
from src.utils import append_jsonl, ensure_dir


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

        append_jsonl(log_path, {"epoch": ep, "loss": float(np.mean(loss_meter)), "recon": recon_v, "kl": kl_v})

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

        append_jsonl(log_path, {"epoch": ep, "g_loss": float(np.mean(g_losses)), "d_loss": float(np.mean(d_losses))})

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

        append_jsonl(log_path, {"epoch": ep, "g_loss": float(np.mean(g_losses)), "c_loss": float(np.mean(c_losses))})

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

        append_jsonl(log_path, {"epoch": ep, "loss": float(np.mean(loss_meter))})

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


def train_generative_model(
    split: Dict,
    index_df: pd.DataFrame,
    gen_cfg: Dict,
    preprocess_cfg: Dict,
    out_dir: str | Path,
) -> Path:
    x_train, y_train = load_samples_by_ids(index_df, split["train_ids"])

    norm = ZScoreNormalizer(
        eps=float(preprocess_cfg["normalization"].get("eps", 1e-6)),
        mode=str(preprocess_cfg["normalization"].get("mode", "channel_global")),
    ).fit(x_train)
    x_train = norm.transform(x_train)

    device = _resolve_device(gen_cfg)
    model_type = normalize_generator_type(str(gen_cfg["model"].get("type", "cvae")))

    batch_size = int(gen_cfg["train"].get("batch_size", 32))
    num_workers = int(gen_cfg["train"].get("num_workers", 0))
    dl = _make_loader(x_train, y_train, batch_size=batch_size, num_workers=num_workers)

    exp_dir = ensure_dir(out_dir)
    log_path = exp_dir / "log.jsonl"

    in_channels = int(x_train.shape[1])
    time_steps = int(x_train.shape[2])
    num_classes = int(np.max(y_train)) + 1

    if model_type == "cvae":
        train_state, model_kwargs = _train_cvae(dl, gen_cfg, in_channels, time_steps, num_classes, device, log_path)
    elif model_type == "eeggan_net":
        train_state, model_kwargs = _train_eeggan(dl, gen_cfg, in_channels, time_steps, num_classes, device, log_path)
    elif model_type == "cwgan_gp":
        train_state, model_kwargs = _train_cwgan_gp(dl, gen_cfg, in_channels, time_steps, num_classes, device, log_path)
    elif model_type == "conditional_ddpm":
        train_state, model_kwargs = _train_conditional_ddpm(dl, gen_cfg, in_channels, time_steps, num_classes, device, log_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    ckpt_path = exp_dir / "ckpt.pt"
    payload = {
        **train_state,
        "model_type": model_type,
        "model_kwargs": model_kwargs,
        "normalizer": norm.state_dict(),
        "shape": {"c": in_channels, "t": time_steps},
        "num_classes": num_classes,
        "gen_cfg": gen_cfg,
    }
    torch.save(payload, ckpt_path)
    return exp_dir
