"""Reward shaping utilities informed by VLM rationale predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DrivingEvents:
    progress: torch.Tensor
    collision: torch.Tensor
    lane_violation: torch.Tensor
    red_light: torch.Tensor


def shaped_reward(
    events: DrivingEvents,
    risk_logits: torch.Tensor,
    meta_action_match: torch.Tensor,
    progress_weight: float,
    collision_penalty: float,
    lane_penalty: float,
    red_light_penalty: float,
    risk_weight: float,
    meta_action_weight: float,
) -> torch.Tensor:
    """Compute scalar reward from simulator events and rationale consistency."""

    risk_probs = risk_logits.softmax(dim=-1)
    risk_scale = torch.linspace(0.0, 1.0, risk_logits.shape[-1], device=risk_logits.device)
    expected_risk = risk_probs.matmul(risk_scale)
    return (
        progress_weight * events.progress
        - collision_penalty * events.collision
        - lane_penalty * events.lane_violation
        - red_light_penalty * events.red_light
        - risk_weight * expected_risk
        + meta_action_weight * meta_action_match
    )
