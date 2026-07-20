#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
DATASET_ROOT="${CARLA_DATASET_ROOT:-results/datasets/carla_il_collect}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-results/feature_cache/carla_il_collect}"
COMMAND_TEXT="${FEATURE_CACHE_COMMAND_TEXT:-You are driving in CARLA. Keep lane and continue safely.}"
MAX_FRAMES_ARG=()
if [[ -n "${FEATURE_CACHE_MAX_FRAMES_PER_EPISODE:-}" ]]; then
  MAX_FRAMES_ARG=("${FEATURE_CACHE_MAX_FRAMES_PER_EPISODE}")
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$MICROMAMBA" run -n vlm python - "$DATASET_ROOT" "$FEATURE_CACHE_ROOT" "$COMMAND_TEXT" "${MAX_FRAMES_ARG[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from vlm_driving.config import ExperimentConfig
from vlm_driving.data import discover_episodes
from vlm_driving.vlm import QwenTokenProvider, build_feature_cache
from vlm_driving.vlm.feature_cache import MANIFEST_NAME, frame_paths_from_episode

dataset_root = Path(sys.argv[1])
cache_root = Path(sys.argv[2])
command_text = sys.argv[3]
max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None
config = ExperimentConfig()
episodes = discover_episodes(dataset_root)
if not episodes:
    raise RuntimeError(f"no episodes found under {dataset_root}")
if not torch.cuda.is_available():
    raise RuntimeError("feature-cache-dataset requires CUDA for frozen Qwen3-VL cache build")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
provider = QwenTokenProvider.from_pretrained_with_fallback(
    model_id=config.vlm.model_id,
    device="cuda",
    command_text=command_text,
)
if provider.hidden_size != config.vlm.hidden_size:
    raise RuntimeError(f"provider hidden_size {provider.hidden_size} != config {config.vlm.hidden_size}")
cache_root.mkdir(parents=True, exist_ok=True)
summary = []
for episode_dir in episodes:
    out_dir = cache_root / episode_dir.name
    manifest = build_feature_cache(
        frames_dir=frame_paths_from_episode(episode_dir),
        provider=provider,
        out_dir=out_dir,
        command_text=command_text,
        max_frames=max_frames,
    )
    summary.append({
        "episode_dir": str(episode_dir),
        "cache_dir": str(out_dir),
        "cache_manifest": str(out_dir / MANIFEST_NAME),
        "num_frames": manifest.num_frames,
    })
(cache_root / "dataset_cache_manifest.json").write_text(
    json.dumps(
        {
            "schema_version": "dataset_feature_cache_v1",
            "dataset_root": str(dataset_root),
            "cache_root": str(cache_root),
            "num_episodes": len(summary),
            "num_frames": sum(item["num_frames"] for item in summary),
            "episodes": summary,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(
    "feature cache dataset ok: "
    f"episodes={len(summary)} frames={sum(item['num_frames'] for item in summary)} "
    f"cache_root={cache_root} peak_vram_gb={peak_vram_gb:.3f}"
)
PY
