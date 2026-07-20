#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$MICROMAMBA" run -n vlm python - <<'PYCODE'
from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch

from vlm_driving.cache import AsyncTokenCache
from vlm_driving.config import ExperimentConfig
from vlm_driving.models import FastPolicy, QueryResampler
from vlm_driving.vlm import QwenTokenProvider

config = ExperimentConfig()
frames = sorted(Path("results/smoke_rollout/frames").glob("*.png"))[:3]
if not frames:
    raise RuntimeError("no smoke rollout frames found under results/smoke_rollout/frames")

if not torch.cuda.is_available():
    raise RuntimeError("make vlm-smoke requires CUDA for Qwen3-VL feasibility measurement")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
provider = QwenTokenProvider.from_pretrained_with_fallback(
    model_id=config.vlm.model_id,
    device="cuda",
    image_paths=frames,
    command_text="You are driving in CARLA. Keep lane and continue safely.",
)
if provider.hidden_size != config.vlm.hidden_size:
    raise RuntimeError(
        f"provider hidden_size {provider.hidden_size} != ExperimentConfig hidden_size {config.vlm.hidden_size}"
    )
resampler = QueryResampler(
    input_dim=provider.hidden_size,
    output_dim=config.resampler.output_dim,
    num_queries=config.resampler.num_queries,
    num_heads=config.resampler.num_heads,
    dropout=0.0,
).to("cuda")
resampler.eval()
cache = AsyncTokenCache(max_age_s=config.policy.max_token_age_s)
policy = FastPolicy(
    obs_dim=config.policy.obs_dim,
    token_dim=config.policy.token_dim,
    hidden_dim=config.policy.hidden_dim,
    action_dim=config.policy.action_dim,
    residual_limits=config.policy.residual_limit,
).to("cuda")
policy.eval()
observation = torch.zeros(1, config.policy.obs_dim, device="cuda")
latencies_ms: list[float] = []
last_action = None
last_hidden_shape = None
last_compact_shape = None

for step, frame in enumerate(frames):
    torch.cuda.synchronize()
    start = time.perf_counter()
    hidden = provider.encode_observation(frame, "Keep lane and continue safely.")
    compact = resampler(hidden.to(dtype=next(resampler.parameters()).dtype))
    cache.update(compact, timestamp_s=float(step))
    tokens, token_age_s, is_fresh = cache.read(now_s=float(step))
    if tokens is None or not is_fresh:
        raise RuntimeError("fresh cached tokens expected after Qwen provider step")
    output = policy(
        observation=observation,
        compact_tokens=tokens,
        token_age_s=torch.tensor([token_age_s], dtype=torch.float32, device="cuda"),
    )
    action = output["action"]
    if action.shape != (1, config.policy.action_dim):
        raise RuntimeError(f"unexpected action shape: {tuple(action.shape)}")
    if not torch.all((action >= -1.0) & (action <= 1.0)):
        raise RuntimeError("policy action exceeded [-1, 1]")
    torch.cuda.synchronize()
    latencies_ms.append((time.perf_counter() - start) * 1000.0)
    last_action = action.detach().cpu()
    last_hidden_shape = tuple(hidden.shape)
    last_compact_shape = tuple(compact.shape)

peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
p50_ms = statistics.median(latencies_ms)
index_95 = min(len(latencies_ms) - 1, int(round(0.95 * (len(latencies_ms) - 1))))
p95_ms = sorted(latencies_ms)[index_95]
print(
    "vlm provider smoke ok: "
    f"model_id={config.vlm.model_id} "
    f"precision={provider.load_info.precision} "
    f"device={provider.load_info.device} "
    f"steps={len(frames)} "
    f"hidden_shape={last_hidden_shape} "
    f"compact_shape={last_compact_shape} "
    f"action_shape={(1, config.policy.action_dim)} "
    f"peak_vram_gb={peak_vram_gb:.3f} "
    f"p50_ms={p50_ms:.3f} p95_ms={p95_ms:.3f} "
    f"last_action={last_action.squeeze(0).tolist()}"
)
PYCODE
