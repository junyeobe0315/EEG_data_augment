from __future__ import annotations

import math
import torch
from torch import nn

from src.models.classifiers.common import TransformerEncoder


class CTNet(nn.Module):
    """CTNet-like classifier (EEGNet-style tokenization + positional Transformer)."""

    def __init__(
        self,
        n_ch: int,
        n_t: int,
        n_classes: int,
        emb_size: int = 40,
        n_heads: int = 8,
        n_layers: int = 3,
        f1: int = 20,
        d: int = 2,
        kernel_length: int = 64,
        sep_kernel_length: int = 16,
        pool1: int = 8,
        pool2: int = 8,
        dropout: float = 0.3,
        pos_dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        f2 = f1 * d
        self.emb_size = int(emb_size)
        self.patch = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, kernel_size=(n_ch, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
            nn.Conv2d(f2, f2, kernel_size=(1, sep_kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
            nn.Conv2d(f2, emb_size, kernel_size=(1, 1), bias=False),
        )
        self.pos_drop = nn.Dropout(pos_dropout)
        self.encoder = TransformerEncoder(emb_size=emb_size, n_heads=n_heads, depth=n_layers, dropout=dropout, ff_mult=ff_mult)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            tokens = self.patch(dummy).squeeze(2).transpose(1, 2)  # [1, L, E]
            n_tokens = int(tokens.shape[1])
            flat_dim = int(tokens.flatten(1).shape[1])

        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, emb_size))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch(x.unsqueeze(1)).squeeze(2).transpose(1, 2)  # [B, L, E]
        z = z * math.sqrt(self.emb_size)
        z = self.pos_drop(z + self.pos_embed[:, : z.shape[1], :])
        z = self.encoder(z)
        return self.classifier(z.flatten(1))


def build_ctnet(cfg: dict, n_ch: int, n_t: int, n_classes: int) -> CTNet:
    return CTNet(
        n_ch=n_ch,
        n_t=n_t,
        n_classes=n_classes,
        emb_size=int(cfg.get("emb_size", 40)),
        n_heads=int(cfg.get("n_heads", 8)),
        n_layers=int(cfg.get("n_layers", 3)),
        f1=int(cfg.get("F1", 20)),
        d=int(cfg.get("D", 2)),
        kernel_length=int(cfg.get("kernel_length", 64)),
        sep_kernel_length=int(cfg.get("sep_kernel_length", 16)),
        pool1=int(cfg.get("pool1", 8)),
        pool2=int(cfg.get("pool2", 8)),
        dropout=float(cfg.get("dropout", 0.3)),
        pos_dropout=float(cfg.get("pos_dropout", 0.1)),
        ff_mult=int(cfg.get("ff_mult", 4)),
    )
