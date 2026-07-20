"""Typed configuration objects for the VLM driving research stack."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal


@dataclass(frozen=True)
class VLMConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    freeze: bool = True
    hidden_size: int = 2048
    token_dim: int = 512
    max_image_tokens: int = 256


@dataclass(frozen=True)
class ResamplerConfig:
    num_queries: int = 32
    input_dim: int = 2048
    output_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1


@dataclass(frozen=True)
class RationaleConfig:
    token_dim: int = 512
    hidden_dim: int = 256
    risk_classes: int = 4
    meta_actions: int = 6


@dataclass(frozen=True)
class PolicyConfig:
    obs_dim: int = 64
    token_dim: int = 512
    hidden_dim: int = 256
    action_dim: int = 2
    max_token_age_s: float = 0.5
    residual_limit: tuple[float, ...] = (0.15, 0.2)


@dataclass(frozen=True)
class RewardConfig:
    collision_penalty: float = 5.0
    lane_penalty: float = 1.0
    red_light_penalty: float = 2.0
    risk_weight: float = 0.25
    meta_action_weight: float = 0.1
    progress_weight: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 1
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    checkpoint_path: str = "results/bc_smoke/bc_checkpoint.pt"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    stage: Literal["il", "ppo", "eval"] = "il"
    vlm: VLMConfig = field(default_factory=VLMConfig)
    resampler: ResamplerConfig = field(default_factory=ResamplerConfig)
    rationale: RationaleConfig = field(default_factory=RationaleConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if self.resampler.input_dim != self.vlm.hidden_size:
            object.__setattr__(
                self,
                "resampler",
                replace(self.resampler, input_dim=self.vlm.hidden_size),
            )
