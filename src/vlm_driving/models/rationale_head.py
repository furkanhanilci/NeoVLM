"""Structured rationale prediction heads for driving supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RationaleOutput:
    risk_logits: torch.Tensor
    meta_action_logits: torch.Tensor
    pooled: torch.Tensor


class StructuredRationaleHead(nn.Module):
    """Predict compact, auditable driving rationale labels from VLM tokens."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        risk_classes: int,
        meta_actions: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.risk_head = nn.Linear(hidden_dim, risk_classes)
        self.meta_action_head = nn.Linear(hidden_dim, meta_actions)

    def forward(self, compact_tokens: torch.Tensor) -> RationaleOutput:
        if compact_tokens.ndim != 3:
            raise ValueError("compact_tokens must have shape [batch, queries, dim]")

        pooled = compact_tokens.mean(dim=1)
        features = self.encoder(pooled)
        return RationaleOutput(
            risk_logits=self.risk_head(features),
            meta_action_logits=self.meta_action_head(features),
            pooled=features,
        )
