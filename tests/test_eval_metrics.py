import json
from pathlib import Path

import pytest

from vlm_driving.carla.metrics import summarize_rollout_metrics as summarize_carla_rollout_metrics
from vlm_driving.eval.metrics import (
    EVENT_LOGGING_REQUIRED,
    aggregate_scores,
    build_eval_report,
    format_eval_report,
    score_episode,
)


def _record(
    step: int,
    progress: float | None,
    distance: float | None,
    collision: bool = False,
    collision_actor_type: str | None = None,
    red_light: bool | None = None,
    latency_ms: float | None = None,
    acceleration: float = 0.0,
) -> dict:
    events = {"collision": collision, "collision_actor_type": collision_actor_type}
    if red_light is not None:
        events["red_light"] = red_light
    return {
        "episode_id": "episode_eval",
        "step": step,
        "ego": {
            "timestamp_s": float(step),
            "x": float(step),
            "y": 0.0,
            "speed_mps": 4.0 + step,
            "acceleration_mps2": acceleration,
        },
        "route": {
            "route_progress_m": progress,
            "distance_to_goal_m": distance,
        },
        "action": {"steer": 0.1 * step, "acceleration": 0.2},
        "control": {"throttle": 0.2, "brake": 0.0},
        "events": events,
        "policy": {"latency_ms": latency_ms} if latency_ms is not None else {},
        "termination": {"reason": "running"},
    }


def test_score_episode_uses_leaderboard_penalty_product_for_logged_infractions():
    records = [
        _record(0, progress=0.0, distance=100.0, acceleration=0.0, latency_ms=10.0),
        _record(
            1,
            progress=40.0,
            distance=60.0,
            collision=True,
            collision_actor_type="vehicle.tesla.model3",
            red_light=True,
            acceleration=2.0,
            latency_ms=30.0,
        ),
        _record(2, progress=80.0, distance=20.0, red_light=False, acceleration=5.0, latency_ms=50.0),
    ]

    score = score_episode(records)

    assert score["route_completion"]["value_pct"] == 80.0
    assert score["infractions"]["by_type"]["collisions_vehicle"]["count"] == 1
    assert score["infractions"]["by_type"]["red_light"]["count"] == 1
    assert score["infraction_penalty"]["value"] == pytest.approx(0.6 * 0.7)
    assert score["driving_score"]["value_pct"] == pytest.approx(80.0 * 0.6 * 0.7)
    assert score["driving_score"]["status"] == "partial_available_logs"
    assert score["comfort"]["mean_abs_jerk_mps3"] == pytest.approx(2.5)
    assert score["comfort"]["max_abs_jerk_mps3"] == pytest.approx(3.0)
    assert score["latency"]["p50_ms"] == 30.0
    assert score["latency"]["p95_ms"] == pytest.approx(48.0)


def test_missing_infraction_event_logging_is_na_not_zero():
    records = [_record(0, progress=10.0, distance=90.0)]

    score = score_episode(records)

    red_light = score["infractions"]["by_type"]["red_light"]
    stop = score["infractions"]["by_type"]["stop_infraction"]
    lane = score["infractions"]["by_type"]["outside_route_lanes"]
    assert red_light["count"] is None
    assert red_light["status"] == "not_available"
    assert red_light["note"] == EVENT_LOGGING_REQUIRED
    assert stop["count"] is None
    assert lane["count"] is None
    assert "red_light" in score["driving_score"]["missing_event_logging"]
    assert score["driving_score"]["leaderboard_comparable"] is False


def test_route_completion_and_driving_score_are_unavailable_without_route_progress():
    score = score_episode([_record(0, progress=None, distance=None)])

    assert score["route_completion"]["status"] == "not_available"
    assert score["route_completion"]["value_pct"] is None
    assert score["driving_score"]["value_pct"] is None
    assert score["driving_score"]["status"] == "unavailable_route_completion"


def test_aggregate_scores_reports_mean_std_and_deterministic_bootstrap():
    scores = [
        {"driving_score": {"value_pct": 10.0, "missing_event_logging": []}, "route_completion": {"value_pct": 20.0}, "infraction_penalty": {"value": 0.5}, "comfort": {"mean_abs_jerk_mps3": 1.0}, "latency": {"p50_ms": None}},
        {"driving_score": {"value_pct": 20.0, "missing_event_logging": ["red_light"]}, "route_completion": {"value_pct": 40.0}, "infraction_penalty": {"value": 0.5}, "comfort": {"mean_abs_jerk_mps3": 3.0}, "latency": {"p50_ms": None}},
    ]

    first = aggregate_scores(scores, bootstrap_iterations=100, seed=5)
    second = aggregate_scores(scores, bootstrap_iterations=100, seed=5)

    assert first == second
    assert first["metrics"]["driving_score_pct"]["mean"] == 15.0
    assert first["metrics"]["driving_score_pct"]["std"] == pytest.approx(7.0710678118654755)
    assert first["metrics"]["driving_score_pct"]["bootstrap_ci95"]["low"] <= 15.0
    assert first["metrics"]["driving_score_pct"]["bootstrap_ci95"]["high"] >= 15.0
    assert first["missing_event_logging"] == ["red_light"]


def test_carla_metrics_delegates_to_eval_basic_summary():
    records = [
        _record(0, progress=0.0, distance=10.0),
        _record(1, progress=1.0, distance=9.0, collision=True, collision_actor_type="static.prop"),
    ]

    metrics = summarize_carla_rollout_metrics(records)

    assert metrics["num_steps"] == 2
    assert metrics["distance_m"] == 1.0
    assert metrics["collision_count"] == 1
    assert metrics["termination_reason"] == "running"


def test_build_eval_report_reads_rollout_dir_and_formats_summary(tmp_path: Path):
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    (rollout / "manifest.json").write_text(
        json.dumps({"episode_id": "rollout_001", "route_length_m": 100.0}) + "\n",
        encoding="utf-8",
    )
    rows = [_record(0, progress=0.0, distance=None), _record(1, progress=50.0, distance=None)]
    (rollout / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = build_eval_report([rollout], bootstrap_iterations=10, bootstrap_seed=2)
    text = format_eval_report(report)

    assert report["episodes"][0]["episode_id"] == "rollout_001"
    assert report["episodes"][0]["route_completion"]["value_pct"] == 50.0
    assert "Route Completion: 50.00%" in text
