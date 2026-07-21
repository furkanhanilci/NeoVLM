"""Evaluation bridge utilities."""

from __future__ import annotations

from importlib import import_module

_METRICS_EXPORTS = {
    "aggregate_scores",
    "build_eval_report",
    "format_eval_report",
    "score_episode",
    "score_rollout_dir",
    "summarize_rollout_metrics",
    "write_eval_report",
}


def __getattr__(name: str):
    if name == "RemoteBCPolicy":
        policy_client = import_module("vlm_driving.eval.policy_client")
        value = getattr(policy_client, name)
        globals()[name] = value
        return value
    if name in _METRICS_EXPORTS:
        metrics = import_module("vlm_driving.eval.metrics")
        value = getattr(metrics, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RemoteBCPolicy",
    "aggregate_scores",
    "build_eval_report",
    "format_eval_report",
    "score_episode",
    "score_rollout_dir",
    "summarize_rollout_metrics",
    "write_eval_report",
]
