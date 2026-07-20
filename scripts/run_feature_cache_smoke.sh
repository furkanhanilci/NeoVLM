#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$MICROMAMBA" run -n vlm python - <<'PYCODE'
from __future__ import annotations

from pathlib import Path

import torch

from vlm_driving.config import ExperimentConfig
from vlm_driving.models import QueryResampler
from vlm_driving.vlm import CachedFeatureReader, QwenTokenProvider, build_feature_cache
from vlm_driving.vlm.feature_cache import frame_paths_from_episode

config = ExperimentConfig()
source_episode = Path("results/datasets/carla_il_smoke/episode_000")
fallback_frames = Path("results/smoke_rollout/frames")
frames_dir = frame_paths_from_episode(source_episode) if source_episode.exists() else fallback_frames
out_dir = Path("results/feature_cache_smoke")
command_text = "You are driving in CARLA. Keep lane and continue safely."
max_frames = 2

if not torch.cuda.is_available():
    raise RuntimeError("feature-cache-smoke requires CUDA for frozen Qwen3-VL cache build")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
provider = QwenTokenProvider.from_pretrained_with_fallback(
    model_id=config.vlm.model_id,
    device="cuda",
    command_text=command_text,
)
if provider.hidden_size != config.vlm.hidden_size:
    raise RuntimeError(
        f"provider hidden_size {provider.hidden_size} != ExperimentConfig hidden_size {config.vlm.hidden_size}"
    )
manifest = build_feature_cache(
    frames_dir=frames_dir,
    provider=provider,
    out_dir=out_dir,
    command_text=command_text,
    max_frames=max_frames,
)
reader = CachedFeatureReader(
    out_dir,
    expected_model_id=config.vlm.model_id,
    expected_hidden_size=config.vlm.hidden_size,
)
first_record = manifest.records[0]
cached_hidden_cpu = reader.read(first_record.frame_key)
if cached_hidden_cpu.dtype != torch.bfloat16:
    raise RuntimeError(f"expected cached bf16 tensor, got {cached_hidden_cpu.dtype}")

resampler = QueryResampler(
    input_dim=config.resampler.input_dim,
    output_dim=config.resampler.output_dim,
    num_queries=config.resampler.num_queries,
    num_heads=config.resampler.num_heads,
    dropout=0.0,
).to("cuda")
resampler.eval()
with torch.no_grad():
    cached_compact = resampler(cached_hidden_cpu.unsqueeze(0).to(device="cuda", dtype=torch.float32))
    live_hidden = provider.encode_observation(first_record.source_frame, command_text)
    live_compact = resampler(live_hidden.to(dtype=torch.float32))
if not torch.allclose(cached_compact, live_compact, atol=1e-4, rtol=1e-4):
    max_diff = (cached_compact - live_compact).abs().max().item()
    raise RuntimeError(f"cached/live resampler mismatch: max_diff={max_diff}")
peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(
    "feature cache smoke ok: "
    f"source={frames_dir} "
    f"out_dir={out_dir} "
    f"frames={manifest.num_frames} "
    f"model_id={manifest.model_id} "
    f"hidden_size={manifest.hidden_size} "
    f"precision={manifest.precision} "
    f"first_shape={first_record.shape} "
    f"first_dtype={first_record.dtype} "
    f"peak_vram_gb={peak_vram_gb:.3f} "
    f"equivalence=pass"
)
PYCODE
