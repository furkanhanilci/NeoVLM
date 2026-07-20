"""Fast policy modules for IL warm-start and bounded residual PPO."""

from __future__ import annotations

import torch
from torch import nn


class ResidualActionHead(nn.Module):
    """Bounds residual actions per dimension with tanh scaling."""

    def __init__(self, input_dim: int, action_dim: int, limits: tuple[float, ...]) -> None:
        super().__init__()
        if len(limits) != action_dim:
            raise ValueError("limits length must match action_dim")
        self.head = nn.Linear(input_dim, action_dim)
        self.register_buffer("limits", torch.tensor(limits, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(features)) * self.limits


class FastPolicy(nn.Module):
    """Staleness-aware low-latency policy consuming proprioception and cached tokens."""

    def __init__(
        self,
        obs_dim: int,
        token_dim: int,
        hidden_dim: int,
        action_dim: int,
        residual_limits: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.token_pool = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim))
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.il_action = nn.Linear(hidden_dim, action_dim)
        self.residual_action = ResidualActionHead(hidden_dim, action_dim, residual_limits)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        observation: torch.Tensor,
        compact_tokens: torch.Tensor,
        token_age_s: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if compact_tokens.ndim != 3:
            raise ValueError("compact_tokens must have shape [batch, queries, dim]")
        if token_age_s.ndim == 1:
            token_age_s = token_age_s.unsqueeze(-1)

        token_features = self.token_pool(compact_tokens.mean(dim=1))
        features = self.trunk(torch.cat([observation, token_features, token_age_s], dim=-1))
        il_action = torch.tanh(self.il_action(features))
        residual = self.residual_action(features)
        return {
            "il_action": il_action,
            "residual": residual,
            "action": torch.clamp(il_action + residual, -1.0, 1.0),
            "value": self.value(features).squeeze(-1),
        }
