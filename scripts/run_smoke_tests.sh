#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
"$ROOT/scripts/check_gpu.sh"
"$ROOT/scripts/check_torch.sh"
"$ROOT/scripts/check_docker_gpu.sh"
"$ROOT/scripts/check_bench2drive.sh" || true
"$ROOT/scripts/check_carla.sh" || true
