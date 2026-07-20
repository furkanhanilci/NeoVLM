#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$MICROMAMBA" run -n vlm python - <<'PYCODE'
from __future__ import annotations

import statistics
import time

import torch

from vlm_driving.cache import AsyncTokenCache
from vlm_driving.config import ExperimentConfig
from vlm_driving.models import FastPolicy, QueryResampler
from vlm_driving.vlm import DummyTokenProvider, SlowTokenizer

config = ExperimentConfig()
batch_size = 1
seq_len = min(config.vlm.max_image_tokens, config.resampler.num_queries)
steps = 10
provider = DummyTokenProvider(
    hidden_size=config.vlm.hidden_size,
    seed=config.seed,
    simulated_latency_s=0.001,
)
resampler = QueryResampler(
    input_dim=config.resampler.input_dim,
    output_dim=config.resampler.output_dim,
    num_queries=config.resampler.num_queries,
    num_heads=config.resampler.num_heads,
    dropout=0.0,
)
resampler.eval()
cache = AsyncTokenCache(max_age_s=config.policy.max_token_age_s)
tokenizer = SlowTokenizer(
    provider=provider,
    resampler=resampler,
    cache=cache,
    batch_size=batch_size,
    seq_len=seq_len,
)
policy = FastPolicy(
    obs_dim=config.policy.obs_dim,
    token_dim=config.policy.token_dim,
    hidden_dim=config.policy.hidden_dim,
    action_dim=config.policy.action_dim,
    residual_limits=config.policy.residual_limit,
)
policy.eval()
observation = torch.zeros(batch_size, config.policy.obs_dim)
latencies_ms: list[float] = []
last_action = None

for step in range(steps):
    now_s = float(step) * (config.policy.max_token_age_s / max(steps, 1))
    start = time.perf_counter()
    result = tokenizer.step(now_s=now_s)
    tokens, token_age_s, is_fresh = cache.read(now_s=now_s)
    if tokens is None or not is_fresh:
        raise RuntimeError("fresh tokens were expected after tokenizer.step")
    output = policy(
        observation=observation,
        compact_tokens=tokens,
        token_age_s=torch.full((batch_size,), token_age_s, dtype=torch.float32),
    )
    action = output["action"]
    if result.hidden_states.shape != (batch_size, seq_len, config.vlm.hidden_size):
        raise RuntimeError(f"unexpected hidden shape: {tuple(result.hidden_states.shape)}")
    expected_compact_shape = (batch_size, config.resampler.num_queries, config.resampler.output_dim)
    if result.compact_tokens.shape != expected_compact_shape:
        raise RuntimeError(f"unexpected compact shape: {tuple(result.compact_tokens.shape)}")
    if action.shape != (batch_size, config.policy.action_dim):
        raise RuntimeError(f"unexpected action shape: {tuple(action.shape)}")
    if not torch.all((action >= -1.0) & (action <= 1.0)):
        raise RuntimeError("policy action exceeded [-1, 1]")
    latencies_ms.append((time.perf_counter() - start) * 1000.0)
    last_action = action.detach()

p50_ms = statistics.median(latencies_ms)
index_95 = min(len(latencies_ms) - 1, int(round(0.95 * (len(latencies_ms) - 1))))
p95_ms = sorted(latencies_ms)[index_95]
print(
    "token pipeline smoke ok: "
    f"steps={steps} "
    f"hidden_shape={(batch_size, seq_len, config.vlm.hidden_size)} "
    f"compact_shape={(batch_size, config.resampler.num_queries, config.resampler.output_dim)} "
    f"action_shape={(batch_size, config.policy.action_dim)} "
    f"p50_ms={p50_ms:.3f} p95_ms={p95_ms:.3f} "
    f"last_action={last_action.squeeze(0).tolist()}"
)
PYCODE
