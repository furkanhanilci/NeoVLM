#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
LOG_DIR="${LOG_DIR:-$ROOT/setup_logs}"
PORT="${POLICY_SERVER_PORT:-8765}"
HOST="${POLICY_SERVER_HOST:-127.0.0.1}"
FRAMES="${BC_BRIDGE_FRAMES:-40}"
IMAGE_WIDTH="${BC_BRIDGE_IMAGE_WIDTH:-800}"
IMAGE_HEIGHT="${BC_BRIDGE_IMAGE_HEIGHT:-450}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$ROOT/src:$ROOT/third_party/CARLA_0.9.15/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"

POLICY_SERVER_HOST="$HOST" POLICY_SERVER_PORT="$PORT" "$ROOT/scripts/run_policy_server.sh" > "$LOG_DIR/policy_server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"$MICROMAMBA" run -n carla python - <<PY 2>&1 | tee "$LOG_DIR/bc_bridge_smoke.log"
from __future__ import annotations

import socket
import time
from pathlib import Path

from vlm_driving.carla import RolloutConfig, run_rollout
from vlm_driving.eval.protocol import recv_message

host = "$HOST"
port = int("$PORT")
server_pid = int("$SERVER_PID")
deadline = time.time() + 180.0
last_error = None


def _server_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        stat = stat_path.read_text().split()
    except OSError:
        return False
    return len(stat) >= 3 and stat[2] != "Z"

while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1.0) as sock:
            ready = recv_message(sock)
        if ready.get("type") == "ready":
            break
        last_error = RuntimeError(f"unexpected policy server readiness message: {ready}")
    except (ConnectionError, OSError, TimeoutError) as exc:
        last_error = exc
        if not _server_alive(server_pid):
            raise RuntimeError("policy server exited before readiness; see setup_logs/policy_server.log") from exc
        time.sleep(1.0)
else:
    raise RuntimeError(f"policy server did not become reachable: {last_error}")

output_dir = run_rollout(
    RolloutConfig(
        frames=int("$FRAMES"),
        output_dir=Path("results/bc_bridge_smoke"),
        episode_id="bc_bridge_smoke_000",
        dataset_name="bc_bridge_smoke",
        save_every_n_frames=1,
        image_width=int("$IMAGE_WIDTH"),
        image_height=int("$IMAGE_HEIGHT"),
        control_mode="bc_remote",
        target_speed_mps=6.0,
        policy_server_host=host,
        policy_server_port=port,
    )
)
print(f"bc bridge smoke ok: {output_dir}")
PY
