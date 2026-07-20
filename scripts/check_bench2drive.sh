#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
LOG_DIR="${LOG_DIR:-$ROOT/setup_logs}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
mkdir -p "$LOG_DIR"
{
  date -Is
  test -d "$ROOT/third_party/Bench2Drive"
  test -d "$ROOT/third_party/Bench2Drive/leaderboard"
  test -d "$ROOT/third_party/Bench2Drive/scenario_runner"
  echo "Bench2Drive tree looks present"
  export PYTHONPATH="$ROOT/third_party/CARLA_0.9.15/PythonAPI/carla:$ROOT/third_party/Bench2Drive/leaderboard:$ROOT/third_party/Bench2Drive/scenario_runner${PYTHONPATH:+:$PYTHONPATH}"
  "$MICROMAMBA" run -n carla python - <<'PY'
import agents.navigation.global_route_planner
import leaderboard.leaderboard_evaluator
import srunner.scenariomanager.scenario_manager
print("Bench2Drive imports ok")
PY
} 2>&1 | tee "$LOG_DIR/check_bench2drive.log"
