#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
LOG_DIR="${LOG_DIR:-$ROOT/setup_logs}"
PORT="${CARLA_RPC_PORT:-2000}"
mkdir -p "$LOG_DIR"
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  echo "CARLA RPC port $PORT is already in use. Stop the existing server first."
  exit 1
fi
setsid "$ROOT/scripts/run_carla_window.sh" > "$LOG_DIR/carla_window_server.log" 2>&1 < /dev/null &
echo "CARLA window server starting on port $PORT"
echo "Log: $LOG_DIR/carla_window_server.log"
