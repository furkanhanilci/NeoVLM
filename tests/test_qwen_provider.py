from pathlib import Path

import pytest
import torch

from vlm_driving.config import ExperimentConfig
from vlm_driving.vlm import QwenTokenProvider


@pytest.mark.vlm
def test_qwen_provider_loads_frozen_and_returns_hidden_states():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real Qwen3-VL provider smoke")
    config = ExperimentConfig()
    frames = sorted(Path("results/smoke_rollout/frames").glob("*.png"))
    if not frames:
        pytest.skip("smoke rollout frames are required")

    provider = QwenTokenProvider.from_pretrained_with_fallback(
        model_id=config.vlm.model_id,
        device="cuda",
        image_paths=frames[:1],
        command_text="Keep lane and continue safely.",
    )
    hidden = provider.encode_observation(frames[0], "Keep lane and continue safely.")

    assert provider.hidden_size == config.vlm.hidden_size
    assert hidden.ndim == 3
    assert hidden.shape[0] == 1
    assert hidden.shape[-1] == config.vlm.hidden_size
    assert provider.model.training is False
    assert all(not parameter.requires_grad for parameter in provider.model.parameters())
    assert provider.load_info.precision in {"bf16", "4bit"}
