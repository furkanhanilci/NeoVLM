"""Rollout metrics for smoke evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence


def summarize_rollout_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_steps": 0,
            "duration_s": 0.0,
            "distance_m": 0.0,
            "mean_speed_mps": 0.0,
            "collision_count": 0,
            "termination_reason": "none",
            "mean_abs_steer": 0.0,
            "mean_throttle": 0.0,
            "mean_brake": 0.0,
        }

    ego = [_dict(record.get("ego")) for record in records]
    actions = [_dict(record.get("action")) for record in records]
    controls = [_dict(record.get("control")) for record in records]
    events = [_dict(record.get("events")) for record in records]
    first_ego = ego[0]
    last_ego = ego[-1]
    duration_s = max(0.0, _float(last_ego.get("timestamp_s")) - _float(first_ego.get("timestamp_s")))
    distance_m = math.hypot(_float(last_ego.get("x")) - _float(first_ego.get("x")), _float(last_ego.get("y")) - _float(first_ego.get("y")))
    termination = _dict(records[-1].get("termination"))
    return {
        "num_steps": len(records),
        "duration_s": duration_s,
        "distance_m": distance_m,
        "mean_speed_mps": _mean(_float(item.get("speed_mps")) for item in ego),
        "collision_count": sum(1 for item in events if bool(item.get("collision", False))),
        "termination_reason": str(termination.get("reason", "unknown")),
        "mean_abs_steer": _mean(abs(_float(item.get("steer"))) for item in actions),
        "mean_throttle": _mean(_float(item.get("throttle")) for item in controls),
        "mean_brake": _mean(_float(item.get("brake")) for item in controls),
    }


def write_rollout_metrics(path: str | Path, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = summarize_rollout_metrics(records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Sequence[float] | Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


__all__ = ["summarize_rollout_metrics", "write_rollout_metrics"]
