#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
LOG_DIR="$ROOT/setup_logs"
mkdir -p "$LOG_DIR"
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
fi
"${DOCKER[@]}" run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu22.04 nvidia-smi \
  2>&1 | tee "$LOG_DIR/check_docker_gpu.log"
