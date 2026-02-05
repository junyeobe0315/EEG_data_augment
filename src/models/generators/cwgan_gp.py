from __future__ import annotations

import math
import torch
from torch import nn


class CWGANGenerator(nn.Module):
    """Conditional conv generator for EEG windows (WGAN-GP)."""

    def __init__(
        self,
        in_channels: int,
        time_steps: int,
        num_classes: int,
        latent_dim: int = 128,
        base_channels: int = 64,
        cond_dim: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.class_embed = nn.Embedding(num_classes, cond_dim)
        self.init_t = max(4, math.ceil(time_steps / 8))

        self.fc = nn.Linear(latent_dim + cond_dim, base_channels * 4 * self.init_t)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = self.class_embed(y)
        h = self.fc(torch.cat([z, cond], dim=1)).view(z.shape[0], -1, self.init_t)
        out = self.net(h)
        if out.shape[-1] != self.time_steps:
            out = torch.nn.functional.interpolate(out, size=self.time_steps, mode="linear", align_corners=False)
        return out

    @torch.no_grad()
    def sample(self, y: torch.Tensor) -> torch.Tensor:
        z = torch.randn(y.shape[0], self.latent_dim, device=y.device)
        return self.forward(z, y)


class CWGANCritic(nn.Module):
    """Conditional conv critic for WGAN-GP."""

    def __init__(
        self,
        in_channels: int,
        time_steps: int,
        num_classes: int,
        base_channels: int = 64,
        cond_dim: int = 16,
    ):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, cond_dim)

        self.feat = nn.Sequential(
            nn.Conv1d(in_channels + cond_dim, base_channels, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels + cond_dim, time_steps)
            feat_dim = int(self.feat(dummy).numel())
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = self.class_embed(y).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        z = self.feat(torch.cat([x, cond], dim=1)).flatten(1)
        return self.head(z).squeeze(1)
