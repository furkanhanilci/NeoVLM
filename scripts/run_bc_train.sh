#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
DATASET_ROOT="${CARLA_DATASET_ROOT:-results/datasets/carla_il_collect}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-results/feature_cache/carla_il_collect}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$DATASET_ROOT/split_manifest.json}"
CHECKPOINT_PATH="${BC_CHECKPOINT:-results/bc_train/bc_checkpoint.pt}"
EPOCHS="${BC_TRAIN_EPOCHS:-40}"
BATCH_SIZE="${BC_TRAIN_BATCH_SIZE:-8}"
LEARNING_RATE="${BC_TRAIN_LR:-0.001}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$MICROMAMBA" run -n vlm python - <<PY
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from vlm_driving.config import ExperimentConfig
from vlm_driving.data import ILDataset, load_split_manifest
from vlm_driving.training import evaluate_bc_loss, load_bc_checkpoint, train_bc

dataset_root = Path("$DATASET_ROOT")
feature_cache_root = Path("$FEATURE_CACHE_ROOT")
split_manifest = Path("$SPLIT_MANIFEST")
checkpoint_path = Path("$CHECKPOINT_PATH")
base_config = ExperimentConfig()
config = replace(
    base_config,
    resampler=replace(base_config.resampler, dropout=0.0),
    train=replace(
        base_config.train,
        batch_size=int("$BATCH_SIZE"),
        epochs=int("$EPOCHS"),
        learning_rate=float("$LEARNING_RATE"),
        checkpoint_path=str(checkpoint_path),
    ),
)
split = load_split_manifest(split_manifest, dataset_root=dataset_root)
if not split.train:
    raise RuntimeError("split has no train episodes")
if not split.val:
    raise RuntimeError("split has no val episodes")
train_dataset = ILDataset(split.train, feature_cache_dir=feature_cache_root, config=config)
val_dataset = ILDataset(split.val, feature_cache_dir=feature_cache_root, config=config)
if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise RuntimeError(f"empty train/val dataset: train={len(train_dataset)} val={len(val_dataset)}")
device = "cuda" if torch.cuda.is_available() else "cpu"
result = train_bc(
    train_dataset,
    config=config,
    device=device,
    checkpoint_path=checkpoint_path,
    val_dataset=val_dataset,
)
final_val_loss = evaluate_bc_loss(val_dataset, result.resampler, result.policy, config=config, device=device)
loaded = load_bc_checkpoint(checkpoint_path, map_location=device)
if loaded.validation_loss_history != result.validation_loss_history:
    raise RuntimeError("checkpoint did not persist validation_loss_history")
print(
    "bc train ok: "
    f"train_samples={len(train_dataset)} val_samples={len(val_dataset)} "
    f"device={device} steps={result.steps} epochs={config.train.epochs} "
    f"initial_train_loss={result.initial_loss:.6f} final_train_loss={result.final_loss:.6f} "
    f"first_val_loss={result.validation_loss_history[0]:.6f} "
    f"final_val_loss={final_val_loss:.6f} checkpoint={checkpoint_path}"
)
PY
