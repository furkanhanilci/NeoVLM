"""Query-based resampling from dense VLM tokens to compact driving tokens."""

from __future__ import annotations

import torch
from torch import nn


class QueryResampler(nn.Module):
    """Compress variable-length VLM tokens into a fixed query set."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_queries: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, output_dim) * 0.02)
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, sequence, dim]")

        batch_size = tokens.shape[0]
        keys = self.input_proj(tokens)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(
            query=queries,
            key=keys,
            value=keys,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = self.norm(queries + attended)
        return self.norm(x + self.mlp(x))
