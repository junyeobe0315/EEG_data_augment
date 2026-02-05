from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.generators.factory import build_generator, build_critic
from src.models.generators.base import normalize_generator_type
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_global_seed


def _apply_cuda_speed(train_cfg: Dict, device: torch.device) -> None:
    if device.type != "cuda":
        return
    cuda_cfg = train_cfg.get("cuda", {}) if isinstance(train_cfg, dict) else {}
    if bool(cuda_cfg.get("tf32", False)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    matmul_precision = cuda_cfg.get("matmul_precision")
    if matmul_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(matmul_precision))


def _resolve_device(train_cfg: dict) -> torch.device:
    req = str(train_cfg.get("device", "auto"))
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def _save_ckpt(
    ckpt_path: Path,
    model_type: str,
    gen: nn.Module,
    critic: nn.Module | None,
    epoch: int,
    model_cfg: dict,
    train_cfg: dict,
    shape: dict,
) -> None:
    payload = {
        "model_type": model_type,
        "gen_state": gen.state_dict(),
        "critic_state": critic.state_dict() if critic is not None else None,
        "epoch": int(epoch),
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "shape": shape,
    }
    torch.save(payload, ckpt_path)


def train_generator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    model_cfg: dict,
    train_cfg: dict,
    run_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Train generator on (x_train, y_train). Returns dict with ckpt paths."""
    ensure_dir(run_dir)
    set_global_seed(seed)

    device = _resolve_device(train_cfg)
    _apply_cuda_speed(train_cfg, device)

    model_type = normalize_generator_type(model_type)
    in_channels = int(x_train.shape[1])
    time_steps = int(x_train.shape[2])
    num_classes = int(np.max(y_train)) + 1

    gen = build_generator(model_type, in_channels, time_steps, num_classes, model_cfg).to(device)
    critic = build_critic(model_type, in_channels, time_steps, num_classes, model_cfg)
    if critic is not None:
        critic = critic.to(device)

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))
    def _seed_worker(worker_id: int) -> None:
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed + worker_id)

    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    epochs = int(train_cfg.get("epochs", 200))
    lr = float(train_cfg.get("lr", 2e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    save_every = int(train_cfg.get("save_every", max(1, epochs // 5)))

    log_rows = []
    ckpts = []
    start_time = time.time()

    if model_type == "cwgan_gp":
        n_critic = int(train_cfg.get("n_critic", 3))
        lambda_gp = float(train_cfg.get("lambda_gp", 10.0))
        betas = train_cfg.get("betas", [0.5, 0.9])
        gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=tuple(betas), weight_decay=weight_decay)
        cri_opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=tuple(betas), weight_decay=weight_decay)

        def _gradient_penalty(real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            eps = torch.rand(real.size(0), 1, 1, device=real.device)
            inter = eps * real + (1.0 - eps) * fake
            inter.requires_grad_(True)
            score = critic(inter, y)
            grads = torch.autograd.grad(
                outputs=score.sum(),
                inputs=inter,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grads = grads.view(grads.size(0), -1)
            gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
            return gp

        for epoch in range(1, epochs + 1):
            gen.train()
            critic.train()
            loss_g = 0.0
            loss_d = 0.0
            n_batches = 0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                n_batches += 1

                # critic update(s)
                for _ in range(n_critic):
                    z = torch.randn(xb.size(0), gen.latent_dim, device=device)
                    fake = gen(z, yb)
                    real_score = critic(xb, yb).mean()
                    fake_score = critic(fake.detach(), yb).mean()
                    gp = _gradient_penalty(xb, fake.detach(), yb)
                    d_loss = fake_score - real_score + lambda_gp * gp
                    cri_opt.zero_grad()
                    d_loss.backward()
                    cri_opt.step()

                # generator update
                z = torch.randn(xb.size(0), gen.latent_dim, device=device)
                fake = gen(z, yb)
                g_loss = -critic(fake, yb).mean()
                gen_opt.zero_grad()
                g_loss.backward()
                gen_opt.step()

                loss_g += float(g_loss.item())
                loss_d += float(d_loss.item())

            row = {
                "epoch": epoch,
                "loss_g": loss_g / max(1, n_batches),
                "loss_d": loss_d / max(1, n_batches),
            }
            log_rows.append(row)

            if epoch % save_every == 0 or epoch == epochs:
                ckpt_path = run_dir / f"ckpt_epoch_{epoch:04d}.pt"
                _save_ckpt(
                    ckpt_path=ckpt_path,
                    model_type=model_type,
                    gen=gen,
                    critic=critic,
                    epoch=epoch,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    shape={"c": in_channels, "t": time_steps, "n_classes": num_classes},
                )
                ckpts.append(str(ckpt_path))

    elif model_type == "ddpm":
        opt = torch.optim.Adam(gen.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(1, epochs + 1):
            gen.train()
            loss_sum = 0.0
            n_batches = 0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                out = gen.loss(xb, yb)
                loss = out["loss"]
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_sum += float(loss.item())
                n_batches += 1
            row = {"epoch": epoch, "loss": loss_sum / max(1, n_batches)}
            log_rows.append(row)

            if epoch % save_every == 0 or epoch == epochs:
                ckpt_path = run_dir / f"ckpt_epoch_{epoch:04d}.pt"
                _save_ckpt(
                    ckpt_path=ckpt_path,
                    model_type=model_type,
                    gen=gen,
                    critic=None,
                    epoch=epoch,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    shape={"c": in_channels, "t": time_steps, "n_classes": num_classes},
                )
                ckpts.append(str(ckpt_path))
    else:
        raise ValueError(f"Unsupported generator type: {model_type}")

    runtime = float(time.time() - start_time)
    write_json(run_dir / "train_log.json", {"rows": log_rows, "runtime_sec": runtime})
    return {
        "ckpts": ckpts,
        "runtime_sec": runtime,
        "model_type": model_type,
        "shape": {"c": in_channels, "t": time_steps, "n_classes": num_classes},
    }


@torch.no_grad()
def sample_from_generator(
    ckpt_path: str | Path,
    y: np.ndarray,
    device: str = "cpu",
    ddpm_steps: int | None = None,
) -> np.ndarray:
    """Sample synthetic data given a generator checkpoint and label array."""
    try:
        ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(Path(ckpt_path), map_location=device)
    model_type = normalize_generator_type(str(ckpt.get("model_type", "cwgan_gp")))
    shape = ckpt.get("shape", {})
    in_channels = int(shape.get("c"))
    time_steps = int(shape.get("t"))
    num_classes = int(shape.get("n_classes"))
    model_cfg = ckpt.get("model_cfg", {})

    gen = build_generator(model_type, in_channels, time_steps, num_classes, model_cfg).to(device)
    gen.load_state_dict(ckpt["gen_state"])
    gen.eval()

    y_t = torch.from_numpy(y.astype(np.int64)).to(device)
    if model_type == "cwgan_gp":
        z = torch.randn(y_t.size(0), gen.latent_dim, device=device)
        out = gen(z, y_t)
    elif model_type == "ddpm":
        steps = ddpm_steps
        out = gen.sample(y_t, num_steps=steps)
    else:
        raise ValueError(f"Unsupported generator type: {model_type}")

    return out.detach().cpu().numpy().astype(np.float32)
