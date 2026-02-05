from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, emb_size: int, n_heads: int, dropout: float, ff_mult: int = 4):
        """Initialize a single Transformer encoder block.

        Inputs:
        - emb_size: embedding dimension E.
        - n_heads: number of attention heads.
        - dropout: dropout probability.
        - ff_mult: feed-forward hidden multiplier.

        Outputs:
        - TransformerBlock module with MHA + FFN layers.

        Internal logic:
        - Builds pre-norm attention and a two-layer feed-forward network.
        """
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
        """Forward pass through one Transformer block.

        Inputs:
        - x: torch.Tensor [B, L, E]

        Outputs:
        - torch.Tensor [B, L, E]

        Internal logic:
        - Applies pre-norm attention and feed-forward residual connections.
        """
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size: int, n_heads: int, depth: int, dropout: float, ff_mult: int = 4):
        """Stack multiple Transformer blocks.

        Inputs:
        - emb_size: embedding dimension E.
        - n_heads: number of attention heads.
        - depth: number of stacked blocks.
        - dropout: dropout probability.
        - ff_mult: feed-forward hidden multiplier.

        Outputs:
        - TransformerEncoder with a list of TransformerBlock modules.

        Internal logic:
        - Constructs a ModuleList of identical Transformer blocks.
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, n_heads, dropout, ff_mult=ff_mult) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacked Transformer blocks.

        Inputs:
        - x: torch.Tensor [B, L, E]

        Outputs:
        - torch.Tensor [B, L, E]

        Internal logic:
        - Sequentially applies each TransformerBlock to the input.
        """
        for blk in self.blocks:
            x = blk(x)
        return x
