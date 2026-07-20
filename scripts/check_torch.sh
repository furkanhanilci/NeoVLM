#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${LOG_DIR:-$HOME/research/vlm-driving-thesis/setup_logs}"
mkdir -p "$LOG_DIR"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
"$MICROMAMBA" run -n vlm python - <<'PY' 2>&1 | tee "$LOG_DIR/check_torch.log"
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

