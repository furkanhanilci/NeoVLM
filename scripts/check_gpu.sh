#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${LOG_DIR:-$HOME/research/vlm-driving-thesis/setup_logs}"
mkdir -p "$LOG_DIR"
{
  date -Is
  nvidia-smi
  echo "## lspci"
  lspci | grep -i nvidia || true
} 2>&1 | tee "$LOG_DIR/check_gpu.log"

