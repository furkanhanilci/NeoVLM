#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
LOG_DIR="${LOG_DIR:-$ROOT/setup_logs}"
DATASET_ROOT="${CARLA_DATASET_ROOT:-results/datasets/carla_il_collect}"
DATASET_NAME="${CARLA_DATASET_NAME:-carla_il_collect}"
NUM_EPISODES="${CARLA_NUM_EPISODES:-50}"
FRAMES_PER_EPISODE="${CARLA_FRAMES_PER_EPISODE:-200}"
SAVE_EVERY_N_FRAMES="${CARLA_SAVE_EVERY_N_FRAMES:-5}"
BASE_SEED="${CARLA_BASE_SEED:-1000}"
VAL_RATIO="${CARLA_VAL_RATIO:-0.15}"
SPLIT_SEED="${CARLA_SPLIT_SEED:-17}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$ROOT/src:$ROOT/third_party/CARLA_0.9.15/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"

CARLA_DATASET_ROOT="$DATASET_ROOT" \
CARLA_DATASET_NAME="$DATASET_NAME" \
CARLA_NUM_EPISODES="$NUM_EPISODES" \
CARLA_FRAMES_PER_EPISODE="$FRAMES_PER_EPISODE" \
CARLA_SAVE_EVERY_N_FRAMES="$SAVE_EVERY_N_FRAMES" \
CARLA_BASE_SEED="$BASE_SEED" \
"$MICROMAMBA" run -n carla python - <<'PY' 2>&1 | tee "$LOG_DIR/carla_dataset_collect.log"
from __future__ import annotations

import os
import time
from pathlib import Path

from vlm_driving.carla import RolloutConfig, run_rollout


def _csv(name: str, default: str) -> list[str]:
    value = os.environ.get(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _optional_csv(name: str) -> list[str]:
    return _csv(name, "")


dataset_root = Path(os.environ["CARLA_DATASET_ROOT"])
dataset_name = os.environ["CARLA_DATASET_NAME"]
num_episodes = int(os.environ["CARLA_NUM_EPISODES"])
frames_per_episode = int(os.environ["CARLA_FRAMES_PER_EPISODE"])
save_every_n_frames = int(os.environ["CARLA_SAVE_EVERY_N_FRAMES"])
base_seed = int(os.environ["CARLA_BASE_SEED"])
host = os.environ.get("CARLA_HOST", "127.0.0.1")
port = int(os.environ.get("CARLA_PORT", "2000"))
timeout_s = float(os.environ.get("CARLA_TIMEOUT_S", "10.0"))
target_speed_mps = float(os.environ.get("CARLA_TARGET_SPEED_MPS", "6.0"))
image_width = int(os.environ.get("CARLA_IMAGE_WIDTH", "800"))
image_height = int(os.environ.get("CARLA_IMAGE_HEIGHT", "450"))
traffic_manager_port = int(os.environ.get("CARLA_TRAFFIC_MANAGER_PORT", "8000"))
route_commands = _csv("CARLA_ROUTE_COMMANDS", "lane_follow,turn_left,turn_right,go_straight")
weather_presets = _csv("CARLA_WEATHER_PRESETS", "ClearNoon,CloudyNoon,WetNoon")
towns = _optional_csv("CARLA_TOWNS")

if towns:
    import carla

    client = carla.Client(host, port)
    client.set_timeout(timeout_s)
else:
    client = None

dataset_root.mkdir(parents=True, exist_ok=True)
for episode_idx in range(num_episodes):
    if client is not None:
        town = towns[episode_idx % len(towns)]
        client.load_world(town)
        time.sleep(2.0)

    route_command = route_commands[episode_idx % len(route_commands)]
    weather_preset = weather_presets[episode_idx % len(weather_presets)] if weather_presets else None
    episode_id = f"{dataset_name}_{episode_idx:04d}"
    output_dir = dataset_root / f"episode_{episode_idx:04d}"
    run_rollout(
        RolloutConfig(
            host=host,
            port=port,
            timeout_s=timeout_s,
            frames=frames_per_episode,
            output_dir=output_dir,
            episode_id=episode_id,
            dataset_name=dataset_name,
            save_every_n_frames=save_every_n_frames,
            image_width=image_width,
            image_height=image_height,
            seed=base_seed + episode_idx,
            route_command=route_command,
            target_speed_mps=target_speed_mps,
            control_mode="autopilot",
            traffic_manager_port=traffic_manager_port,
            weather_preset=weather_preset,
        )
    )
    print(f"collected {episode_id}: route={route_command} weather={weather_preset} output={output_dir}", flush=True)
PY

CARLA_DATASET_ROOT="$DATASET_ROOT" \
CARLA_VAL_RATIO="$VAL_RATIO" \
CARLA_SPLIT_SEED="$SPLIT_SEED" \
CARLA_SAVE_EVERY_N_FRAMES="$SAVE_EVERY_N_FRAMES" \
CARLA_FRAMES_PER_EPISODE="$FRAMES_PER_EPISODE" \
"$MICROMAMBA" run -n vlm python - <<'PY' 2>&1 | tee "$LOG_DIR/carla_dataset_split.log"
from __future__ import annotations

import os
from pathlib import Path

from vlm_driving.data import discover_episodes, split_episodes, write_split_manifest

dataset_root = Path(os.environ["CARLA_DATASET_ROOT"])
val_ratio = float(os.environ["CARLA_VAL_RATIO"])
split_seed = int(os.environ["CARLA_SPLIT_SEED"])
frames_per_episode = int(os.environ["CARLA_FRAMES_PER_EPISODE"])
save_every_n_frames = int(os.environ["CARLA_SAVE_EVERY_N_FRAMES"])
episodes = discover_episodes(dataset_root)
split = split_episodes(episodes, val_ratio=val_ratio, seed=split_seed)
manifest_path = write_split_manifest(
    dataset_root / "split_manifest.json",
    split,
    dataset_root=dataset_root,
    val_ratio=val_ratio,
    seed=split_seed,
)
saved_per_episode = (frames_per_episode + save_every_n_frames - 1) // save_every_n_frames
total_saved = saved_per_episode * len(episodes)
est_cache_gib = total_saved * 1.457 / 1024.0
print(
    "dataset split ok: "
    f"episodes={len(episodes)} train={len(split.train)} val={len(split.val)} "
    f"saved_frames_est={total_saved} cache_est_gib={est_cache_gib:.2f} "
    f"manifest={manifest_path}"
)
PY
