from __future__ import annotations

import time
from dataclasses import dataclass
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
    """Apply CUDA speed settings for generator training.

    Inputs:
    - train_cfg: generator training config dict.
    - device: torch.device for training.

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


def _resolve_device(train_cfg: dict) -> torch.device:
    """Resolve training device from config.

    Inputs:
    - train_cfg: generator training config dict (device key).

    Outputs:
    - torch.device for training.

    Internal logic:
    - Returns CUDA when available and device is "auto", otherwise uses config.
    """
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
    """Serialize generator/critic checkpoint to disk.

    Inputs:
    - ckpt_path: output checkpoint path.
    - model_type: generator type string.
    - gen: generator module.
    - critic: optional critic module.
    - epoch: current epoch index.
    - model_cfg/train_cfg: config dicts saved for reproducibility.
    - shape: dict with data shape metadata.

    Outputs:
    - None (writes torch checkpoint).

    Internal logic:
    - Packages states and metadata into a dict and calls torch.save.
    """
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


def _build_early_stop_state(train_cfg: dict, model_type: str, run_dir: Path) -> dict[str, Any]:
    """Create early-stopping state from train config."""
    es_cfg = train_cfg.get("early_stopping", {}) if isinstance(train_cfg, dict) else {}
    enabled = bool(es_cfg.get("enabled", False))
    monitor_cfg = es_cfg.get("monitor", "auto")
    if isinstance(monitor_cfg, dict):
        monitor = str(monitor_cfg.get(model_type, "auto"))
    else:
        monitor = str(monitor_cfg)
    if monitor == "auto":
        monitor = {"cwgan_gp": "loss_g", "cvae": "loss", "ddpm": "loss"}.get(model_type, "loss")

    mode = str(es_cfg.get("mode", "min")).lower()
    if mode not in {"min", "max"}:
        mode = "min"

    patience = max(1, int(es_cfg.get("patience_epochs", es_cfg.get("patience", 500))))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    start_epoch = max(1, int(es_cfg.get("start_epoch", 1)))
    best_init = float("inf") if mode == "min" else float("-inf")
    return {
        "enabled": enabled,
        "monitor": monitor,
        "mode": mode,
        "patience_epochs": patience,
        "min_delta": min_delta,
        "start_epoch": start_epoch,
        "best_value": best_init,
        "best_epoch": 0,
        "no_improve_epochs": 0,
        "stopped_epoch": 0,
        "best_ckpt_path": run_dir / "ckpt_best_by_loss.pt",
    }


def _update_early_stop(
    state: dict[str, Any],
    row: dict[str, float],
    epoch: int,
    model_type: str,
    gen: nn.Module,
    critic: nn.Module | None,
    model_cfg: dict,
    train_cfg: dict,
    shape: dict,
) -> bool:
    """Update early-stopping state and return whether training should stop."""
    if not bool(state.get("enabled", False)):
        return False
    monitor = str(state.get("monitor", "loss"))
    raw = row.get(monitor)
    if raw is None:
        return False
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value):
        return False

    best = float(state.get("best_value", float("inf")))
    min_delta = float(state.get("min_delta", 0.0))
    mode = str(state.get("mode", "min"))
    improved = (value < (best - min_delta)) if mode == "min" else (value > (best + min_delta))

    if improved:
        state["best_value"] = value
        state["best_epoch"] = int(epoch)
        state["no_improve_epochs"] = 0
        _save_ckpt(
            ckpt_path=state["best_ckpt_path"],
            model_type=model_type,
            gen=gen,
            critic=critic,
            epoch=epoch,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            shape=shape,
        )
    else:
        state["no_improve_epochs"] = int(state.get("no_improve_epochs", 0)) + 1

    if int(epoch) < int(state.get("start_epoch", 1)):
        return False
    if int(state.get("no_improve_epochs", 0)) >= int(state.get("patience_epochs", 1)):
        state["stopped_epoch"] = int(epoch)
        return True
    return False


def train_generator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    model_cfg: dict,
    train_cfg: dict,
    run_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Train generator on (x_train, y_train).

    Inputs:
    - x_train/y_train: real training data [N, C, T] / [N].
    - model_type/model_cfg: generator selection and hyperparameters.
    - train_cfg: training config (epochs, lr, batch_size, etc.).
    - run_dir: output directory for checkpoints and logs.
    - seed: RNG seed for reproducibility.

    Outputs:
    - dict with ckpt paths, runtime, model_type, and shape metadata.

    Internal logic:
    - Builds generator/critic, trains by type (cwgan_gp/cvae/ddpm), and saves checkpoints.
    """
    ensure_dir(run_dir)
    set_global_seed(seed)

    device = _resolve_device(train_cfg)
    _apply_cuda_speed(train_cfg, device)

    model_type = normalize_generator_type(model_type)
    in_channels = int(x_train.shape[1])  # EEG channels
    time_steps = int(x_train.shape[2])  # time samples per trial
    num_classes = int(np.max(y_train)) + 1  # number of classes

    gen = build_generator(model_type, in_channels, time_steps, num_classes, model_cfg).to(device)
    critic = build_critic(model_type, in_channels, time_steps, num_classes, model_cfg)
    if critic is not None:
        critic = critic.to(device)

    batch_size = int(train_cfg.get("batch_size", 32))  # training batch size
    num_workers = int(train_cfg.get("num_workers", 0))  # dataloader workers
    def _seed_worker(worker_id: int) -> None:
        """Seed a DataLoader worker for deterministic shuffling.

        Inputs:
        - worker_id: integer worker index.

        Outputs:
        - None (sets NumPy seed for worker).

        Internal logic:
        - Uses torch.initial_seed to derive per-worker NumPy seed.
        """
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

    epochs = int(train_cfg.get("epochs", 200))  # training epochs
    lr = float(train_cfg.get("lr", 2e-4))  # learning rate
    weight_decay = float(train_cfg.get("weight_decay", 0.0))  # L2 regularization

    save_every = int(train_cfg.get("save_every", max(1, epochs // 5)))  # ckpt interval
    shape_meta = {"c": in_channels, "t": time_steps, "n_classes": num_classes}
    es_state = _build_early_stop_state(train_cfg, model_type, run_dir)

    log_rows = []
    ckpts = []
    start_time = time.time()

    if model_type == "cwgan_gp":
        n_critic = int(train_cfg.get("n_critic", 3))  # critic updates per step
        lambda_gp = float(train_cfg.get("lambda_gp", 10.0))  # gradient penalty weight
        betas = train_cfg.get("betas", [0.5, 0.9])
        gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=tuple(betas), weight_decay=weight_decay)
        cri_opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=tuple(betas), weight_decay=weight_decay)

        def _gradient_penalty(real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Compute WGAN-GP gradient penalty.

            Inputs:
            - real: torch.Tensor [B, C, T] real samples.
            - fake: torch.Tensor [B, C, T] generated samples.
            - y: torch.Tensor [B] labels.

            Outputs:
            - scalar penalty tensor.

            Internal logic:
            - Interpolates real/fake, computes critic gradients, and penalizes norm.
            """
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
            stop_now = _update_early_stop(
                state=es_state,
                row=row,
                epoch=epoch,
                model_type=model_type,
                gen=gen,
                critic=critic,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                shape=shape_meta,
            )

            if epoch % save_every == 0 or epoch == epochs or stop_now:
                ckpt_path = run_dir / f"ckpt_epoch_{epoch:04d}.pt"
                _save_ckpt(
                    ckpt_path=ckpt_path,
                    model_type=model_type,
                    gen=gen,
                    critic=critic,
                    epoch=epoch,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    shape=shape_meta,
                )
                ckpts.append(str(ckpt_path))
            if stop_now:
                break

    elif model_type == "cvae":
        opt = torch.optim.Adam(gen.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(1, epochs + 1):
            gen.train()
            loss_sum = 0.0
            rec_sum = 0.0
            kl_sum = 0.0
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
                rec_sum += float(out.get("recon_loss", torch.tensor(0.0, device=device)).item())
                kl_sum += float(out.get("kl_loss", torch.tensor(0.0, device=device)).item())
                n_batches += 1
            row = {
                "epoch": epoch,
                "loss": loss_sum / max(1, n_batches),
                "recon_loss": rec_sum / max(1, n_batches),
                "kl_loss": kl_sum / max(1, n_batches),
            }
            log_rows.append(row)
            stop_now = _update_early_stop(
                state=es_state,
                row=row,
                epoch=epoch,
                model_type=model_type,
                gen=gen,
                critic=None,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                shape=shape_meta,
            )

            if epoch % save_every == 0 or epoch == epochs or stop_now:
                ckpt_path = run_dir / f"ckpt_epoch_{epoch:04d}.pt"
                _save_ckpt(
                    ckpt_path=ckpt_path,
                    model_type=model_type,
                    gen=gen,
                    critic=None,
                    epoch=epoch,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    shape=shape_meta,
                )
                ckpts.append(str(ckpt_path))
            if stop_now:
                break

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
            stop_now = _update_early_stop(
                state=es_state,
                row=row,
                epoch=epoch,
                model_type=model_type,
                gen=gen,
                critic=None,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                shape=shape_meta,
            )

            if epoch % save_every == 0 or epoch == epochs or stop_now:
                ckpt_path = run_dir / f"ckpt_epoch_{epoch:04d}.pt"
                _save_ckpt(
                    ckpt_path=ckpt_path,
                    model_type=model_type,
                    gen=gen,
                    critic=None,
                    epoch=epoch,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    shape=shape_meta,
                )
                ckpts.append(str(ckpt_path))
            if stop_now:
                break
    else:
        raise ValueError(f"Unsupported generator type: {model_type}")

    best_ckpt_path = Path(es_state.get("best_ckpt_path", ""))
    if best_ckpt_path.exists():
        best_ckpt_str = str(best_ckpt_path)
        if best_ckpt_str not in ckpts:
            ckpts.append(best_ckpt_str)

    runtime = float(time.time() - start_time)
    best_value = es_state.get("best_value", np.nan)
    if not np.isfinite(float(best_value)):
        best_value = None
    write_json(
        run_dir / "train_log.json",
        {
            "rows": log_rows,
            "runtime_sec": runtime,
            "early_stopping": {
                "enabled": bool(es_state.get("enabled", False)),
                "monitor": es_state.get("monitor"),
                "mode": es_state.get("mode"),
                "patience_epochs": int(es_state.get("patience_epochs", 0)),
                "min_delta": float(es_state.get("min_delta", 0.0)),
                "start_epoch": int(es_state.get("start_epoch", 1)),
                "best_epoch": int(es_state.get("best_epoch", 0)),
                "best_value": best_value,
                "stopped_epoch": int(es_state.get("stopped_epoch", 0)),
                "best_ckpt_path": str(best_ckpt_path) if best_ckpt_path.exists() else "",
            },
        },
    )
    return {
        "ckpts": ckpts,
        "runtime_sec": runtime,
        "model_type": model_type,
        "shape": {"c": in_channels, "t": time_steps, "n_classes": num_classes},
    }


@dataclass
class LoadedGeneratorSampler:
    """Reusable checkpoint-backed sampler to avoid repeated model reloads."""

    ckpt_path: str | Path
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Load checkpoint and build generator once for repeated sampling."""
        try:
            ckpt = torch.load(Path(self.ckpt_path), map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(Path(self.ckpt_path), map_location=self.device)

        self.model_type = normalize_generator_type(str(ckpt.get("model_type", "cwgan_gp")))
        shape = ckpt.get("shape", {})
        in_channels = int(shape.get("c"))
        time_steps = int(shape.get("t"))
        num_classes = int(shape.get("n_classes"))
        model_cfg = ckpt.get("model_cfg", {})

        self.gen = build_generator(self.model_type, in_channels, time_steps, num_classes, model_cfg).to(self.device)
        self.gen.load_state_dict(ckpt["gen_state"])
        self.gen.eval()

    @torch.no_grad()
    def sample(self, y: np.ndarray, ddpm_steps: int | None = None) -> np.ndarray:
        """Sample synthetic trials for a label vector using the loaded model."""
        y_t = torch.from_numpy(y.astype(np.int64)).to(self.device)
        if self.model_type in {"cwgan_gp", "cvae"}:
            z = torch.randn(y_t.size(0), self.gen.latent_dim, device=self.device)
            if self.model_type == "cwgan_gp":
                out = self.gen(z, y_t)
            else:
                out = self.gen.decode(z, y_t)
        elif self.model_type == "ddpm":
            out = self.gen.sample(y_t, num_steps=ddpm_steps)
        else:
            raise ValueError(f"Unsupported generator type: {self.model_type}")
        return out.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def sample_from_generator(
    ckpt_path: str | Path,
    y: np.ndarray,
    device: str = "cpu",
    ddpm_steps: int | None = None,
) -> np.ndarray:
    """Sample synthetic data given a generator checkpoint and label array.

    Inputs:
    - ckpt_path: path to generator checkpoint.
    - y: ndarray [N] class labels to condition on.
    - device: torch device string for sampling.
    - ddpm_steps: optional number of DDPM sampling steps.

    Outputs:
    - ndarray [N, C, T] synthetic samples.

    Internal logic:
    - Uses a reusable sampler abstraction (single-use wrapper for compatibility).
    """
    sampler = LoadedGeneratorSampler(ckpt_path=ckpt_path, device=device)
    return sampler.sample(y, ddpm_steps=ddpm_steps)
