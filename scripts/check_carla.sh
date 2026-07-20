#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${LOG_DIR:-$HOME/research/vlm-driving-thesis/setup_logs}"
mkdir -p "$LOG_DIR"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
export PYTHONPATH="$ROOT/third_party/CARLA_0.9.15/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"
"$MICROMAMBA" run -n carla python - <<'PY' 2>&1 | tee "$LOG_DIR/check_carla.log"
import carla
print("carla import ok")
print("carla module", carla.__file__)
import agents.navigation.global_route_planner
print("carla agents import ok")
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(5.0)
try:
    world = client.get_world()
    print("server ok", world.get_map().name)
except Exception as exc:
    print("server unavailable", repr(exc))
    import os
    if os.environ.get("REQUIRE_CARLA_SERVER") == "1":
        raise
PY
