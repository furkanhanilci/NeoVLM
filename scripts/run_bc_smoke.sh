#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$MICROMAMBA" run -n vlm python - <<'PYCODE'
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from vlm_driving.config import ExperimentConfig
from vlm_driving.data import ILDataset
from vlm_driving.training import load_bc_checkpoint, predict_il_action, train_bc

base_config = ExperimentConfig()
config = replace(
    base_config,
    resampler=replace(base_config.resampler, dropout=0.0),
    train=replace(
        base_config.train,
        batch_size=1,
        epochs=120,
        learning_rate=1e-3,
        checkpoint_path="results/bc_smoke/bc_checkpoint.pt",
    ),
)
episode_dir = Path("results/datasets/carla_il_smoke/episode_000")
cache_dir = Path("results/feature_cache_smoke")
if not episode_dir.exists():
    raise RuntimeError(f"missing smoke episode: {episode_dir}; run make carla-dataset-smoke first")
if not (cache_dir / "cache_manifest.json").exists():
    raise RuntimeError(f"missing feature cache: {cache_dir}; run make feature-cache-smoke first")

dataset = ILDataset(episode_dir, feature_cache_dir=cache_dir, config=config)
if len(dataset) == 0:
    raise RuntimeError("BC smoke dataset is empty")
device = "cuda" if torch.cuda.is_available() else "cpu"
result = train_bc(dataset, config=config, device=device, checkpoint_path=config.train.checkpoint_path)
if result.final_loss >= result.initial_loss:
    raise RuntimeError(f"BC loss did not decrease: initial={result.initial_loss:.6f}, final={result.final_loss:.6f}")
loaded = load_bc_checkpoint(config.train.checkpoint_path, map_location=device)
result.resampler.eval()
result.policy.eval()
with torch.no_grad():
    sample = dataset[0]
    original_action = predict_il_action(result.resampler, result.policy, [sample], device=device)
    loaded_action = predict_il_action(loaded.resampler, loaded.policy, [sample], device=device)
if not torch.allclose(original_action, loaded_action, atol=1e-6, rtol=1e-6):
    raise RuntimeError("loaded BC checkpoint does not reproduce the saved model output")
print(
    "bc smoke ok: "
    f"samples={len(dataset)} "
    f"steps={result.steps} "
    f"device={device} "
    f"initial_loss={result.initial_loss:.6f} "
    f"final_loss={result.final_loss:.6f} "
    f"loss_ratio={result.final_loss / max(result.initial_loss, 1e-12):.4f} "
    f"checkpoint={result.checkpoint_path} "
    f"action_shape={tuple(loaded_action.shape)}"
)
PYCODE
