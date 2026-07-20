"""Behavior cloning training loop for cached frozen-VLM features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from vlm_driving.config import (
    ExperimentConfig,
    PolicyConfig,
    RationaleConfig,
    ResamplerConfig,
    RewardConfig,
    TrainConfig,
    VLMConfig,
)
from vlm_driving.models import FastPolicy, QueryResampler


@dataclass
class BCTrainingResult:
    resampler: QueryResampler
    policy: FastPolicy
    loss_history: list[float]
    checkpoint_path: Path | None

    @property
    def initial_loss(self) -> float:
        return self.loss_history[0]

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1]

    @property
    def steps(self) -> int:
        return len(self.loss_history)


@dataclass
class BCCheckpoint:
    config: ExperimentConfig
    train_config: TrainConfig
    resampler: QueryResampler
    policy: FastPolicy
    loss_history: list[float]
    path: Path


def build_bc_models(config: ExperimentConfig, device: str | torch.device = "cpu") -> tuple[QueryResampler, FastPolicy]:
    target_device = torch.device(device)
    resampler = QueryResampler(
        input_dim=config.resampler.input_dim,
        output_dim=config.resampler.output_dim,
        num_queries=config.resampler.num_queries,
        num_heads=config.resampler.num_heads,
        dropout=config.resampler.dropout,
    ).to(target_device)
    policy = FastPolicy(
        obs_dim=config.policy.obs_dim,
        token_dim=config.policy.token_dim,
        hidden_dim=config.policy.hidden_dim,
        action_dim=config.policy.action_dim,
        residual_limits=config.policy.residual_limit,
    ).to(target_device)
    return resampler, policy


def train_bc(
    dataset: Dataset,
    config: ExperimentConfig | None = None,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    shuffle: bool = True,
) -> BCTrainingResult:
    """Trains resampler + policy with MSE on the policy IL action head.

    The VLM is not part of this loop; every sample must carry a cached hidden
    state. The residual and value heads are intentionally excluded from the IL
    loss because they belong to the later PPO stage.
    """

    if len(dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("BC training requires a non-empty dataset")
    if config is None:
        config = getattr(dataset, "config", ExperimentConfig())
    _validate_train_config(config.train)

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    resampler, policy = build_bc_models(config, target_device)
    resampler.train()
    policy.train()
    optimizer = torch.optim.AdamW(
        list(resampler.parameters()) + list(policy.parameters()),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    loss_history: list[float] = []
    for epoch in range(config.train.epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(dataset, batch_size=None, shuffle=shuffle, generator=generator)
        pending: list[dict[str, Any]] = []
        for sample in loader:
            pending.append(sample)
            if len(pending) == config.train.batch_size:
                loss_history.append(_train_batch(pending, resampler, policy, optimizer, target_device))
                pending.clear()
        if pending:
            loss_history.append(_train_batch(pending, resampler, policy, optimizer, target_device))

    if not loss_history:
        raise RuntimeError("BC training completed without optimization steps")

    saved_path = Path(checkpoint_path) if checkpoint_path is not None else Path(config.train.checkpoint_path)
    save_bc_checkpoint(saved_path, config, resampler, policy, loss_history)

    return BCTrainingResult(
        resampler=resampler,
        policy=policy,
        loss_history=loss_history,
        checkpoint_path=saved_path,
    )


def predict_il_action(
    resampler: QueryResampler,
    policy: FastPolicy,
    samples: Sequence[dict[str, Any]],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    target_device = torch.device(device)
    observation, compact_tokens, _ = make_bc_batch(samples, resampler, target_device)
    token_age_s = torch.zeros(observation.shape[0], dtype=torch.float32, device=target_device)
    return policy(observation, compact_tokens, token_age_s)["il_action"]


def make_bc_batch(
    samples: Sequence[dict[str, Any]],
    resampler: QueryResampler,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observations: list[torch.Tensor] = []
    compact_tokens: list[torch.Tensor] = []
    expert_actions: list[torch.Tensor] = []

    for sample in samples:
        hidden = sample.get("cached_hidden")
        if hidden is None:
            raise ValueError("BC training requires cached_hidden; build/pass a T-011 feature cache first")
        if hidden.ndim != 2:
            raise ValueError(f"cached_hidden must have shape [S, H], got {tuple(hidden.shape)}")
        observation = sample["observation"]
        expert_action = sample["expert_action"]
        with torch.set_grad_enabled(resampler.training):
            compact = resampler(hidden.unsqueeze(0).to(device=device, dtype=torch.float32)).squeeze(0)
        observations.append(observation.to(device=device, dtype=torch.float32))
        compact_tokens.append(compact)
        expert_actions.append(expert_action.to(device=device, dtype=torch.float32))

    return torch.stack(observations), torch.stack(compact_tokens), torch.stack(expert_actions)


def save_bc_checkpoint(
    path: str | Path,
    config: ExperimentConfig,
    resampler: QueryResampler,
    policy: FastPolicy,
    loss_history: Sequence[float],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "bc_checkpoint_v1",
            "config": asdict(config),
            "train_config": asdict(config.train),
            "resampler_state_dict": resampler.state_dict(),
            "policy_state_dict": policy.state_dict(),
            "loss_history": [float(loss) for loss in loss_history],
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_bc_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> BCCheckpoint:
    checkpoint_path = Path(path)
    data = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if data.get("schema_version") != "bc_checkpoint_v1":
        raise ValueError(f"unsupported BC checkpoint schema: {data.get('schema_version')}")
    config = experiment_config_from_dict(data["config"])
    train_config = TrainConfig(**data.get("train_config", asdict(config.train)))
    if config.policy.action_dim != len(config.policy.residual_limit):
        raise ValueError("checkpoint config has action_dim/residual_limit mismatch")

    resampler, policy = build_bc_models(config, map_location)
    resampler.load_state_dict(data["resampler_state_dict"])
    policy.load_state_dict(data["policy_state_dict"])
    resampler.eval()
    policy.eval()
    return BCCheckpoint(
        config=config,
        train_config=train_config,
        resampler=resampler,
        policy=policy,
        loss_history=[float(loss) for loss in data["loss_history"]],
        path=checkpoint_path,
    )


def experiment_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    policy_data = dict(data.get("policy", {}))
    if "residual_limit" in policy_data:
        policy_data["residual_limit"] = tuple(policy_data["residual_limit"])
    return ExperimentConfig(
        seed=int(data.get("seed", 0)),
        stage=data.get("stage", "il"),
        vlm=VLMConfig(**data.get("vlm", {})),
        resampler=ResamplerConfig(**data.get("resampler", {})),
        rationale=RationaleConfig(**data.get("rationale", {})),
        policy=PolicyConfig(**policy_data),
        reward=RewardConfig(**data.get("reward", {})),
        train=TrainConfig(**data.get("train", {})),
    )


def _train_batch(
    samples: Sequence[dict[str, Any]],
    resampler: QueryResampler,
    policy: FastPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    observation, compact_tokens, expert_action = make_bc_batch(samples, resampler, device)
    token_age_s = torch.zeros(observation.shape[0], dtype=torch.float32, device=device)
    output = policy(observation, compact_tokens, token_age_s)
    loss = F.mse_loss(output["il_action"], expert_action)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _validate_train_config(train_config: TrainConfig) -> None:
    if train_config.batch_size <= 0:
        raise ValueError("train.batch_size must be positive")
    if train_config.epochs <= 0:
        raise ValueError("train.epochs must be positive")
    if train_config.learning_rate <= 0.0:
        raise ValueError("train.learning_rate must be positive")
    if train_config.weight_decay < 0.0:
        raise ValueError("train.weight_decay must be non-negative")


__all__ = [
    "BCCheckpoint",
    "BCTrainingResult",
    "build_bc_models",
    "experiment_config_from_dict",
    "load_bc_checkpoint",
    "make_bc_batch",
    "predict_il_action",
    "save_bc_checkpoint",
    "train_bc",
]
