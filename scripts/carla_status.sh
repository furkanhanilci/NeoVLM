#!/usr/bin/env bash
set -euo pipefail
PORT="${CARLA_RPC_PORT:-2000}"
pgrep -af 'CarlaUE4|CarlaUE4-Linux|UE4' || true
ss -ltnp "sport = :$PORT" || true
