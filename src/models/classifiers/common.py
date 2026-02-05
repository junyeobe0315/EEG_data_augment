from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, emb_size: int, n_heads: int, dropout: float, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * ff_mult, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size: int, n_heads: int, depth: int, dropout: float, ff_mult: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, n_heads, dropout, ff_mult=ff_mult) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
