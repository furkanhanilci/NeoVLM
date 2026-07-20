import torch

from vlm_driving.cache import AsyncTokenCache
from vlm_driving.config import ExperimentConfig
from vlm_driving.models import FastPolicy, QueryResampler
from vlm_driving.vlm import DummyTokenProvider, SlowTokenizer


def build_tokenizer(config: ExperimentConfig) -> tuple[SlowTokenizer, AsyncTokenCache, int]:
    seq_len = min(config.vlm.max_image_tokens, config.resampler.num_queries)
    cache = AsyncTokenCache(max_age_s=config.policy.max_token_age_s)
    resampler = QueryResampler(
        input_dim=config.resampler.input_dim,
        output_dim=config.resampler.output_dim,
        num_queries=config.resampler.num_queries,
        num_heads=config.resampler.num_heads,
        dropout=0.0,
    )
    resampler.eval()
    tokenizer = SlowTokenizer(
        provider=DummyTokenProvider(hidden_size=config.vlm.hidden_size, seed=config.seed),
        resampler=resampler,
        cache=cache,
        batch_size=1,
        seq_len=seq_len,
    )
    return tokenizer, cache, seq_len


def test_slow_tokenizer_writes_compact_tokens_to_cache():
    config = ExperimentConfig()
    tokenizer, cache, seq_len = build_tokenizer(config)

    result = tokenizer.step(now_s=1.25)
    cached_tokens, token_age_s, is_fresh = cache.read(now_s=1.50)

    assert result.hidden_states.shape == (1, seq_len, config.vlm.hidden_size)
    assert result.compact_tokens.shape == (1, config.resampler.num_queries, config.resampler.output_dim)
    assert cached_tokens is not None
    assert torch.equal(cached_tokens, result.compact_tokens)
    assert token_age_s == 0.25
    assert is_fresh is True


def test_token_pipeline_outputs_policy_action_within_bounds():
    config = ExperimentConfig()
    tokenizer, cache, _ = build_tokenizer(config)
    tokenizer.step(now_s=0.0)
    compact_tokens, token_age_s, is_fresh = cache.read(now_s=config.policy.max_token_age_s)
    assert compact_tokens is not None
    assert is_fresh is True
    policy = FastPolicy(
        obs_dim=config.policy.obs_dim,
        token_dim=config.policy.token_dim,
        hidden_dim=config.policy.hidden_dim,
        action_dim=config.policy.action_dim,
        residual_limits=config.policy.residual_limit,
    )

    output = policy(
        observation=torch.zeros(1, config.policy.obs_dim),
        compact_tokens=compact_tokens,
        token_age_s=torch.tensor([token_age_s], dtype=torch.float32),
    )

    assert output["action"].shape == (1, config.policy.action_dim)
    assert torch.all(output["action"] <= 1.0)
    assert torch.all(output["action"] >= -1.0)
