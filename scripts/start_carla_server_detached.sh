#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
LOG_DIR="${LOG_DIR:-$ROOT/setup_logs}"
PORT="${CARLA_RPC_PORT:-2000}"
mkdir -p "$LOG_DIR"
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  echo "CARLA RPC port $PORT is already listening"
  exit 0
fi
setsid "$ROOT/scripts/run_carla_server.sh" > "$LOG_DIR/carla_detached_server.log" 2>&1 < /dev/null &
echo "CARLA server starting in background on port $PORT"
echo "Log: $LOG_DIR/carla_detached_server.log"
