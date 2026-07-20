#!/usr/bin/env bash
set -euo pipefail
CARLA_ROOT="${CARLA_ROOT:-$HOME/research/vlm-driving-thesis/third_party/CARLA_0.9.15}"
exec "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -carla-rpc-port=2000
