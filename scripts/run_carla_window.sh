#!/usr/bin/env bash
set -euo pipefail
CARLA_ROOT="${CARLA_ROOT:-$HOME/research/vlm-driving-thesis/third_party/CARLA_0.9.15}"
PORT="${CARLA_RPC_PORT:-2000}"
exec "$CARLA_ROOT/CarlaUE4.sh" -windowed -ResX=1280 -ResY=720 -quality-level=Low -nosound -carla-rpc-port="$PORT"
