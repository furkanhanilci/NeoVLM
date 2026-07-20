from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from vlm_driving.config import ExperimentConfig, PolicyConfig, ResamplerConfig, TrainConfig, VLMConfig
from vlm_driving.training import load_bc_checkpoint, predict_il_action, train_bc


class TinyCachedILDataset(Dataset):
    def __init__(self, include_cache: bool = True) -> None:
        self.config = _tiny_config()
        hidden_a = torch.tensor(
            [[1.0, 0.0, 0.5, -0.5], [0.2, -0.1, 0.0, 0.4], [0.3, 0.2, -0.2, 0.1]],
            dtype=torch.float32,
        )
        hidden_b = torch.tensor(
            [[-0.4, 0.8, 0.1, 0.0], [0.5, -0.7, 0.3, 0.2], [0.1, 0.4, -0.6, 0.3], [0.0, 0.2, 0.2, -0.3], [0.9, 0.1, -0.1, 0.5]],
            dtype=torch.float32,
        )
        self.samples = [
            {
                "observation": torch.tensor([0.0, 0.2, -0.1, 0.5, 0.0, 1.0], dtype=torch.float32),
                "cached_hidden": hidden_a if include_cache else None,
                "expert_action": torch.tensor([0.35, -0.25], dtype=torch.float32),
            },
            {
                "observation": torch.tensor([0.4, -0.2, 0.3, 0.0, 1.0, 0.0], dtype=torch.float32),
                "cached_hidden": hidden_b if include_cache else None,
                "expert_action": torch.tensor([-0.2, 0.45], dtype=torch.float32),
            },
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | None]:
        return self.samples[index]


def _tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=123,
        vlm=VLMConfig(hidden_size=4, token_dim=8),
        resampler=ResamplerConfig(num_queries=2, input_dim=4, output_dim=8, num_heads=2, dropout=0.0),
        policy=PolicyConfig(obs_dim=6, token_dim=8, hidden_dim=16, action_dim=2, residual_limit=(0.1, 0.1)),
        train=TrainConfig(batch_size=2, epochs=180, learning_rate=5e-3, weight_decay=0.0),
    )


def test_train_bc_overfits_tiny_cached_dataset_and_checkpoint_roundtrip(tmp_path: Path):
    dataset = TinyCachedILDataset()
    config = dataset.config
    checkpoint_path = tmp_path / "bc.pt"

    result = train_bc(dataset, config=config, device="cpu", checkpoint_path=checkpoint_path)

    assert result.steps == config.train.epochs
    assert result.final_loss < result.initial_loss * 0.5
    assert checkpoint_path.exists()

    loaded = load_bc_checkpoint(checkpoint_path, map_location="cpu")
    result.resampler.eval()
    result.policy.eval()
    sample = dataset[0]
    with torch.no_grad():
        original = predict_il_action(result.resampler, result.policy, [sample], device="cpu")
        restored = predict_il_action(loaded.resampler, loaded.policy, [sample], device="cpu")
    assert torch.allclose(original, restored, atol=1e-6, rtol=1e-6)
    assert loaded.config.policy.action_dim == 2
    assert loaded.config.policy.residual_limit == (0.1, 0.1)
    assert loaded.loss_history == result.loss_history


def test_train_bc_requires_cached_hidden_states():
    dataset = TinyCachedILDataset(include_cache=False)

    with pytest.raises(ValueError, match="cached_hidden"):
        train_bc(dataset, config=dataset.config, device="cpu")


def test_train_bc_rejects_empty_dataset():
    class EmptyDataset(Dataset):
        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            raise IndexError(index)

    with pytest.raises(ValueError, match="non-empty dataset"):
        train_bc(EmptyDataset(), config=_tiny_config(), device="cpu")
