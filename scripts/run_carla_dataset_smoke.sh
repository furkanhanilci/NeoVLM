#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
LOG_DIR="${LOG_DIR:-$ROOT/setup_logs}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$ROOT/src:$ROOT/third_party/CARLA_0.9.15/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"

"$MICROMAMBA" run -n carla python - <<'PY' 2>&1 | tee "$LOG_DIR/carla_dataset_smoke.log"
from pathlib import Path

from vlm_driving.carla import RolloutConfig, run_rollout

output_dir = run_rollout(
    RolloutConfig(
        frames=100,
        output_dir=Path("results/datasets/carla_il_smoke/episode_000"),
        episode_id="carla_il_smoke_000",
        dataset_name="carla_il_smoke",
        save_every_n_frames=5,
        control_mode="autopilot",
        target_speed_mps=6.0,
    )
)
print(f"carla dataset smoke ok: {output_dir}")
PY
