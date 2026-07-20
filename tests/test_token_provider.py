import pytest
import torch

from vlm_driving.config import ExperimentConfig
from vlm_driving.vlm import DummyTokenProvider


def test_dummy_token_provider_is_deterministic_for_same_seed():
    config = ExperimentConfig()
    seq_len = min(config.vlm.max_image_tokens, config.resampler.num_queries)
    first = DummyTokenProvider(hidden_size=config.vlm.hidden_size, seed=config.seed)
    second = DummyTokenProvider(hidden_size=config.vlm.hidden_size, seed=config.seed)

    first_tokens = first.encode(batch_size=1, seq_len=seq_len)
    second_tokens = second.encode(batch_size=1, seq_len=seq_len)

    assert first_tokens.shape == (1, seq_len, config.vlm.hidden_size)
    assert torch.equal(first_tokens, second_tokens)


def test_dummy_token_provider_validates_shapes_and_latency():
    config = ExperimentConfig()
    provider = DummyTokenProvider(hidden_size=config.vlm.hidden_size, seed=config.seed)

    with pytest.raises(ValueError, match="batch_size"):
        provider.encode(batch_size=0, seq_len=config.resampler.num_queries)
    with pytest.raises(ValueError, match="seq_len"):
        provider.encode(batch_size=1, seq_len=0)
    with pytest.raises(ValueError, match="hidden_size"):
        DummyTokenProvider(hidden_size=0)
    with pytest.raises(ValueError, match="simulated_latency_s"):
        DummyTokenProvider(hidden_size=config.vlm.hidden_size, simulated_latency_s=-0.1)
