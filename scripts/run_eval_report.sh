#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
ROLLOUT_DIRS="${EVAL_ROLLOUT_DIRS:-results/bc_bridge_smoke}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-results/eval_report}"
OUTPUT_JSON="${EVAL_OUTPUT_JSON:-$OUTPUT_DIR/eval_report.json}"
OUTPUT_SUMMARY="${EVAL_OUTPUT_SUMMARY:-$OUTPUT_DIR/eval_report.txt}"
BOOTSTRAP_ITERATIONS="${EVAL_BOOTSTRAP_ITERATIONS:-1000}"
BOOTSTRAP_SEED="${EVAL_BOOTSTRAP_SEED:-17}"
JERK_LIMIT_MPS3="${EVAL_JERK_LIMIT_MPS3:-10.0}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

read -r -a ROLLOUT_ARGS <<< "$ROLLOUT_DIRS"
if (( ${#ROLLOUT_ARGS[@]} == 0 )); then
  echo "EVAL_ROLLOUT_DIRS must contain at least one rollout directory" >&2
  exit 2
fi

"$MICROMAMBA" run -n vlm python -m vlm_driving.eval.metrics \
  "${ROLLOUT_ARGS[@]}" \
  --json-out "$OUTPUT_JSON" \
  --summary-out "$OUTPUT_SUMMARY" \
  --bootstrap-iterations "$BOOTSTRAP_ITERATIONS" \
  --bootstrap-seed "$BOOTSTRAP_SEED" \
  --jerk-limit-mps3 "$JERK_LIMIT_MPS3"
