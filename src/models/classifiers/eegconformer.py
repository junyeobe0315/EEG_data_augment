from __future__ import annotations

import torch
from torch import nn

from src.models.classifiers.common import TransformerEncoder


class EEGConformer(nn.Module):
    """EEG-Conformer-like classifier using shallow conv patch embedding + Transformer."""

    def __init__(
        self,
        n_ch: int,
        n_t: int,
        n_classes: int,
        conv_channels: int = 40,
        emb_size: int = 40,
        n_heads: int = 10,
        n_layers: int = 6,
        dropout: float = 0.5,
        temporal_kernel: int = 25,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        fc1: int = 256,
        fc2: int = 32,
    ):
        """Initialize EEGConformer model.

        Inputs:
        - n_ch/n_t: input channels and time length.
        - n_classes: number of classes.
        - conv_channels/emb_size/n_heads/n_layers/dropout: architecture params.
        - temporal_kernel/pool_kernel/pool_stride: patch embedding params.
        - fc1/fc2: classifier head sizes.

        Outputs:
        - EEGConformer instance with patch embedder, transformer, and head.

        Internal logic:
        - Builds shallow conv patching, transformer encoder, and MLP head.
        """
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(1, temporal_kernel), stride=(1, 1), bias=False),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(n_ch, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride)),
            nn.Dropout(dropout),
            nn.Conv2d(conv_channels, emb_size, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )
        self.encoder = TransformerEncoder(emb_size=emb_size, n_heads=n_heads, depth=n_layers, dropout=dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            feat = self.patch(dummy).flatten(1)
            flat_dim = int(feat.shape[1])

        self.head = nn.Sequential(
            nn.Linear(flat_dim, fc1),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(fc1, fc2),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(fc2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EEGConformer.

        Inputs:
        - x: torch.Tensor [B, C, T]

        Outputs:
        - logits: torch.Tensor [B, K]

        Internal logic:
        - Creates patch tokens, encodes with transformer, then flattens to head.
        """
        z = self.patch(x.unsqueeze(1)).squeeze(2)  # [B, E, L]
        z = z.transpose(1, 2)  # [B, L, E]
        z = self.encoder(z)
        return self.head(z.flatten(1))


def build_eegconformer(cfg: dict, n_ch: int, n_t: int, n_classes: int) -> EEGConformer:
    """Construct EEGConformer from config dict.

    Inputs:
    - cfg: model config dict.
    - n_ch/n_t: input shape.
    - n_classes: output classes.

    Outputs:
    - EEGConformer instance.

    Internal logic:
    - Maps config dict fields to EEGConformer constructor arguments.
    """
    return EEGConformer(
        n_ch=n_ch,
        n_t=n_t,
        n_classes=n_classes,
        conv_channels=int(cfg.get("conv_channels", 40)),
        emb_size=int(cfg.get("emb_size", 40)),
        n_heads=int(cfg.get("n_heads", 10)),
        n_layers=int(cfg.get("n_layers", 6)),
        dropout=float(cfg.get("dropout", 0.5)),
        temporal_kernel=int(cfg.get("temporal_kernel", 25)),
        pool_kernel=int(cfg.get("pool_kernel", 75)),
        pool_stride=int(cfg.get("pool_stride", 15)),
        fc1=int(cfg.get("fc1", 256)),
        fc2=int(cfg.get("fc2", 32)),
    )
