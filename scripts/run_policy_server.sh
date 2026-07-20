#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PORT="${POLICY_SERVER_PORT:-8765}"
HOST="${POLICY_SERVER_HOST:-127.0.0.1}"
CHECKPOINT="${BC_CHECKPOINT:-$ROOT/results/bc_smoke/bc_checkpoint.pt}"
DEVICE_ARG=()
if [[ -n "${POLICY_SERVER_DEVICE:-}" ]]; then
  DEVICE_ARG=(--device "$POLICY_SERVER_DEVICE")
fi

exec "$MICROMAMBA" run -n vlm python -m vlm_driving.eval.policy_server \
  --host "$HOST" \
  --port "$PORT" \
  --checkpoint "$CHECKPOINT" \
  "${DEVICE_ARG[@]}"
