from __future__ import annotations

import torch
from torch import nn


class ConditionalVAE1D(nn.Module):
    """Conditional variational autoencoder for EEG trial generation."""

    def __init__(
        self,
        in_channels: int,
        time_steps: int,
        num_classes: int,
        latent_dim: int = 64,
        base_channels: int = 64,
        cond_dim: int = 16,
        beta_kl: float = 1.0e-3,
    ):
        """Initialize conditional VAE with lightweight 1D conv encoder/decoder.

        Inputs:
        - in_channels/time_steps/num_classes: EEG data shape and class count.
        - latent_dim/base_channels/cond_dim: architecture widths.
        - beta_kl: KL weight in the VAE objective.

        Outputs:
        - ConditionalVAE1D module.

        Internal logic:
        - Encodes x|y to Gaussian posterior and decodes z|y back to x.
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.time_steps = int(time_steps)
        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)
        self.cond_dim = int(cond_dim)
        self.beta_kl = float(beta_kl)

        self.class_embed = nn.Embedding(self.num_classes, self.cond_dim)

        self.encoder = nn.Sequential(
            nn.Conv1d(self.in_channels + self.cond_dim, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels * 2),
            nn.SiLU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels * 4),
            nn.SiLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels + self.cond_dim, self.time_steps)
            enc = self.encoder(dummy)
        self._enc_shape = tuple(enc.shape[1:])
        self._flat_dim = int(enc.numel())

        self.fc_mu = nn.Linear(self._flat_dim + self.cond_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim + self.cond_dim, self.latent_dim)

        self.fc_decode = nn.Linear(self.latent_dim + self.cond_dim, self._flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.SiLU(),
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.SiLU(),
            nn.ConvTranspose1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, self.in_channels, kernel_size=3, padding=1),
        )

    def _label_feature(self, y: torch.Tensor, length: int) -> torch.Tensor:
        """Create time-broadcasted label feature map for conditioning."""
        cond = self.class_embed(y)
        return cond.unsqueeze(-1).expand(-1, -1, length)

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode x|y into posterior parameters (mu, logvar)."""
        cond_time = self._label_feature(y, x.shape[-1])
        h = self.encoder(torch.cat([x, cond_time], dim=1)).flatten(1)
        cond = self.class_embed(y)
        h = torch.cat([h, cond], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=-30.0, max=20.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent z using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Decode z|y back to EEG sample space."""
        cond = self.class_embed(y)
        h = self.fc_decode(torch.cat([z, cond], dim=1))
        h = h.view(z.shape[0], *self._enc_shape)
        out = self.decoder(h)
        if out.shape[-1] != self.time_steps:
            out = torch.nn.functional.interpolate(out, size=self.time_steps, mode="linear", align_corners=False)
        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and posterior params."""
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    def loss(self, x0: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute VAE objective (MSE reconstruction + beta-weighted KL)."""
        recon, mu, logvar = self.forward(x0, y)
        recon_loss = torch.mean((recon - x0) ** 2)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta_kl * kl
        return {"loss": total, "recon_loss": recon_loss, "kl_loss": kl}

    @torch.no_grad()
    def sample(self, y: torch.Tensor) -> torch.Tensor:
        """Generate samples conditioned on labels."""
        z = torch.randn(y.shape[0], self.latent_dim, device=y.device)
        return self.decode(z, y)
