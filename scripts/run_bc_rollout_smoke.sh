#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
LOG_DIR="${LOG_DIR:-$ROOT/setup_logs}"
CARLA_EGG="$ROOT/third_party/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$ROOT/src:$ROOT/third_party/CARLA_0.9.15/PythonAPI/carla:$CARLA_EGG${PYTHONPATH:+:$PYTHONPATH}"

"$MICROMAMBA" run -n vlm python - <<'PY' 2>&1 | tee "$LOG_DIR/bc_rollout_smoke.log"
from __future__ import annotations

import os
from pathlib import Path

try:
    import carla  # noqa: F401
except Exception as exc:
    raise RuntimeError(
        "bc-rollout-smoke is blocked in the current split environment: "
        "the vlm env has torch/VLM but cannot import CARLA 0.9.15 PythonAPI. "
        "Create a unified eval env with torch + CARLA PythonAPI, or run the CARLA-less "
        "agent tests until that environment exists."
    ) from exc

from vlm_driving.carla import RolloutConfig, run_rollout

feature_cache_env = os.environ.get("BC_FEATURE_CACHE_DIR")
feature_cache_dir = Path(feature_cache_env) if feature_cache_env else None
output_dir = run_rollout(
    RolloutConfig(
        frames=int(os.environ.get("BC_ROLLOUT_FRAMES", "40")),
        output_dir=Path("results/bc_rollout_smoke"),
        episode_id="bc_rollout_smoke_000",
        dataset_name="bc_rollout_smoke",
        save_every_n_frames=1,
        control_mode="bc_policy",
        target_speed_mps=6.0,
        bc_checkpoint_path=Path(os.environ.get("BC_CHECKPOINT", "results/bc_smoke/bc_checkpoint.pt")),
        bc_hidden_source=os.environ.get("BC_HIDDEN_SOURCE", "live"),
        bc_feature_cache_dir=feature_cache_dir,
    )
)
print(f"bc rollout smoke ok: {output_dir}")
PY
