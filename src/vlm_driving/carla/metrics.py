"""Rollout metrics for smoke evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from vlm_driving.eval.metrics import summarize_rollout_metrics


def write_rollout_metrics(path: str | Path, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = summarize_rollout_metrics(records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


__all__ = ["summarize_rollout_metrics", "write_rollout_metrics"]
