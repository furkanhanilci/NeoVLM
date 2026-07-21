#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
MICROMAMBA="${MICROMAMBA:-$HOME/bin/micromamba}"
DATASET_ROOT="${DATASET_STATS_ROOT:-${CARLA_DATASET_ROOT:-results/datasets/carla_il_collect}}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$DATASET_ROOT/split_manifest.json}"
OUTPUT_DIR="${DATASET_STATS_OUTPUT_DIR:-results/dataset_stats}"
OUTPUT_JSON="${DATASET_STATS_JSON:-$OUTPUT_DIR/carla_il_collect_stats.json}"
OUTPUT_SUMMARY="${DATASET_STATS_SUMMARY:-$OUTPUT_DIR/carla_il_collect_summary.txt}"
HISTOGRAM_BINS="${DATASET_STATS_HISTOGRAM_BINS:-10}"
CACHE_MIB_PER_FRAME="${FEATURE_CACHE_MIB_PER_FRAME:-1.457}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SPLIT_ARGS=()
if [[ -f "$SPLIT_MANIFEST" ]]; then
  SPLIT_ARGS=(--split-manifest "$SPLIT_MANIFEST")
fi

"$MICROMAMBA" run -n vlm python -m vlm_driving.data.dataset_stats \
  "$DATASET_ROOT" \
  "${SPLIT_ARGS[@]}" \
  --histogram-bins "$HISTOGRAM_BINS" \
  --cache-mib-per-frame "$CACHE_MIB_PER_FRAME" \
  --json-out "$OUTPUT_JSON" \
  --summary-out "$OUTPUT_SUMMARY"
