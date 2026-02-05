from __future__ import annotations

import math
import torch
from torch import nn


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / half)
    ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=1)
    return emb


class _ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.cond_proj(cond).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class _CondUNet1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.cls_embed = nn.Embedding(num_classes, time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = _ResidualBlock1D(base_channels, base_channels, time_dim)
        self.ds1 = nn.Conv1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)

        self.down2 = _ResidualBlock1D(base_channels, base_channels * 2, time_dim)
        self.ds2 = nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.mid = _ResidualBlock1D(base_channels * 2, base_channels * 4, time_dim)

        self.us1 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up1 = _ResidualBlock1D(base_channels * 4, base_channels * 2, time_dim)

        self.us2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up2 = _ResidualBlock1D(base_channels * 2, base_channels, time_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(_sinusoidal_time_embedding(t, self.time_dim))
        cond = temb + self.cls_embed(y)

        x0 = self.in_conv(x)
        d1 = self.down1(x0, cond)
        x = self.ds1(d1)

        d2 = self.down2(x, cond)
        x = self.ds2(d2)

        x = self.mid(x, cond)

        x = self.us1(x)
        if x.shape[-1] != d2.shape[-1]:
            x = torch.nn.functional.interpolate(x, size=d2.shape[-1], mode="linear", align_corners=False)
        x = self.up1(torch.cat([x, d2], dim=1), cond)

        x = self.us2(x)
        if x.shape[-1] != d1.shape[-1]:
            x = torch.nn.functional.interpolate(x, size=d1.shape[-1], mode="linear", align_corners=False)
        x = self.up2(torch.cat([x, d1], dim=1), cond)

        return self.out(x)


class ConditionalDDPM1D(nn.Module):
    """Conditional DDPM with a lightweight 1D U-Net noise predictor."""

    def __init__(
        self,
        in_channels: int,
        time_steps: int,
        num_classes: int,
        base_channels: int = 64,
        time_dim: int = 128,
        diffusion_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.diffusion_steps = diffusion_steps

        self.eps_model = _CondUNet1D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            time_dim=time_dim,
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        c1 = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        c2 = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return c1 * x0 + c2 * noise

    def loss(self, x0: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x0.shape[0]
        t = torch.randint(0, self.diffusion_steps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self._q_sample(x0, t, noise)
        pred = self.eps_model(xt, t, y)
        mse = torch.mean((pred - noise) ** 2)
        return {"loss": mse}

    @torch.no_grad()
    def sample(self, y: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.diffusion_steps
        num_steps = min(num_steps, self.diffusion_steps)

        x = torch.randn(y.shape[0], self.in_channels, self.time_steps, device=y.device)
        start = self.diffusion_steps - 1
        stop = self.diffusion_steps - num_steps

        for step in range(start, stop - 1, -1):
            t = torch.full((y.shape[0],), step, device=y.device, dtype=torch.long)
            eps = self.eps_model(x, t, y)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bar[step]
            beta = self.betas[step]
            z = torch.randn_like(x) if step > 0 else torch.zeros_like(x)

            x = (x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)) * eps) / torch.sqrt(alpha)
            x = x + torch.sqrt(beta) * z

        return x
