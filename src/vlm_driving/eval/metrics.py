"""Pure-Python closed-loop evaluation metrics for rollout records."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

LEADERBOARD_PENALTY_REFERENCE = (
    "third_party/Bench2Drive/leaderboard/leaderboard/utils/statistics_manager.py:"
    "PENALTY_VALUE_DICT and score_composed = score_route * score_penalty"
)

LEADERBOARD_INFRACTION_PENALTIES: dict[str, float] = {
    "collisions_pedestrian": 0.50,
    "collisions_vehicle": 0.60,
    "collisions_layout": 0.65,
    "red_light": 0.70,
    "stop_infraction": 0.80,
    "scenario_timeouts": 0.70,
    "yield_emergency_vehicle_infractions": 0.70,
}

PERCENT_INFRACTION_PENALTIES: dict[str, dict[str, Any]] = {
    "outside_route_lanes": {"coefficient": 0.0, "mode": "increases"},
    "min_speed_infractions": {"coefficient": 0.70, "mode": "unused"},
}

EXPLICIT_EVENT_INFRACTION_FIELDS: dict[str, tuple[str, ...]] = {
    "red_light": ("red_light", "traffic_light_infraction", "ran_red_light"),
    "stop_infraction": ("stop_infraction", "stop_sign_infraction", "ran_stop_sign"),
    "outside_route_lanes": ("outside_route_lanes", "lane_infraction", "outside_route_lanes_pct"),
    "route_dev": ("route_dev", "route_deviation"),
    "vehicle_blocked": ("vehicle_blocked",),
    "scenario_timeouts": ("scenario_timeout", "scenario_timeouts"),
    "yield_emergency_vehicle_infractions": ("yield_emergency_vehicle", "yield_emergency_vehicle_infraction"),
    "min_speed_infractions": ("min_speed_infraction", "min_speed_infractions"),
}

EVENT_LOGGING_REQUIRED = "N/A - event logging required"
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_BOOTSTRAP_SEED = 17
DEFAULT_JERK_LIMIT_MPS3 = 10.0


def summarize_rollout_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Basic rollout aggregates kept as the single source for carla.metrics."""

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
    distance_m = math.hypot(
        _float(last_ego.get("x")) - _float(first_ego.get("x")),
        _float(last_ego.get("y")) - _float(first_ego.get("y")),
    )
    termination = _dict(records[-1].get("termination"))
    return {
        "num_steps": len(records),
        "duration_s": duration_s,
        "distance_m": distance_m,
        "mean_speed_mps": _mean(_float(item.get("speed_mps")) for item in ego),
        "collision_count": sum(_collision_event_count(item) for item in events),
        "termination_reason": str(termination.get("reason", "unknown")),
        "mean_abs_steer": _mean(abs(_float(item.get("steer"))) for item in actions),
        "mean_throttle": _mean(_float(item.get("throttle")) for item in controls),
        "mean_brake": _mean(_float(item.get("brake")) for item in controls),
    }


def score_episode(
    records: Sequence[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
    episode_id: str | None = None,
    jerk_limit_mps3: float = DEFAULT_JERK_LIMIT_MPS3,
) -> dict[str, Any]:
    basic = summarize_rollout_metrics(records)
    route_completion = _route_completion(records, manifest or {})
    infractions = _infraction_breakdown(records)
    penalty = _infraction_penalty(infractions)
    comfort = _comfort_metrics(records, jerk_limit_mps3=jerk_limit_mps3)
    latency = _latency_metrics(records)
    missing_event_logging = [
        name for name, entry in infractions["by_type"].items() if entry.get("status") == "not_available"
    ]
    route_pct = route_completion.get("value_pct")
    driving_score_pct = None
    if route_pct is not None:
        driving_score_pct = max(float(route_pct) * float(penalty["value"]), 0.0)

    score_status = "leaderboard_comparable"
    if route_pct is None:
        score_status = "unavailable_route_completion"
    elif missing_event_logging or infractions["unknown_collision_count"]:
        score_status = "partial_available_logs"

    return {
        "schema_version": "eval_episode_score_v1",
        "episode_id": episode_id or _episode_id(records, manifest or {}),
        "source_manifest": str((manifest or {}).get("_path", "")),
        "basic": basic,
        "route_completion": route_completion,
        "infractions": infractions,
        "infraction_penalty": penalty,
        "driving_score": {
            "value_pct": driving_score_pct,
            "status": score_status,
            "formula": "RouteCompletion(%) * product(available logged infraction penalty coefficients)",
            "leaderboard_reference": LEADERBOARD_PENALTY_REFERENCE,
            "missing_event_logging": missing_event_logging,
            "leaderboard_comparable": score_status == "leaderboard_comparable",
        },
        "comfort": comfort,
        "latency": latency,
    }


def aggregate_scores(
    episode_scores: Sequence[dict[str, Any]],
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    metrics = {
        "driving_score_pct": [_get_path(score, ("driving_score", "value_pct")) for score in episode_scores],
        "route_completion_pct": [_get_path(score, ("route_completion", "value_pct")) for score in episode_scores],
        "infraction_penalty": [_get_path(score, ("infraction_penalty", "value")) for score in episode_scores],
        "mean_abs_jerk_mps3": [_get_path(score, ("comfort", "mean_abs_jerk_mps3")) for score in episode_scores],
        "latency_p50_ms": [_get_path(score, ("latency", "p50_ms")) for score in episode_scores],
    }
    return {
        "schema_version": "eval_aggregate_v1",
        "num_episodes": len(episode_scores),
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": seed,
        "metrics": {
            name: _aggregate_values(_valid_numbers(values), bootstrap_iterations=bootstrap_iterations, seed=seed)
            for name, values in metrics.items()
        },
        "missing_event_logging": sorted(
            {
                item
                for score in episode_scores
                for item in score.get("driving_score", {}).get("missing_event_logging", [])
            }
        ),
    }


def score_rollout_dir(path: str | Path, jerk_limit_mps3: float = DEFAULT_JERK_LIMIT_MPS3) -> dict[str, Any]:
    rollout_dir = Path(path)
    manifest_path = rollout_dir / "manifest.json"
    metadata_path = rollout_dir / "metadata.jsonl"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    if manifest_path.exists():
        manifest["_path"] = str(manifest_path)
    records = _read_jsonl(metadata_path)
    return score_episode(records, manifest=manifest, jerk_limit_mps3=jerk_limit_mps3)


def build_eval_report(
    rollout_dirs: Sequence[str | Path],
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    jerk_limit_mps3: float = DEFAULT_JERK_LIMIT_MPS3,
) -> dict[str, Any]:
    episode_scores = [score_rollout_dir(path, jerk_limit_mps3=jerk_limit_mps3) for path in rollout_dirs]
    return {
        "schema_version": "eval_report_v1",
        "rollout_dirs": [str(Path(path)) for path in rollout_dirs],
        "episodes": episode_scores,
        "aggregate": aggregate_scores(
            episode_scores,
            bootstrap_iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
    }


def format_episode_score(score: dict[str, Any]) -> str:
    rc = score["route_completion"]
    ds = score["driving_score"]
    basic = score["basic"]
    comfort = score["comfort"]
    latency = score["latency"]
    missing = ds.get("missing_event_logging", [])
    route_text = _fmt_pct(rc.get("value_pct")) if rc.get("value_pct") is not None else f"N/A ({rc.get('note')})"
    score_text = _fmt_pct(ds.get("value_pct")) if ds.get("value_pct") is not None else "N/A"
    lines = [
        f"Episode: {score['episode_id']}",
        f"Route Completion: {route_text}",
        f"Driving Score: {score_text} [{ds['status']}]",
        f"Infraction penalty: {score['infraction_penalty']['value']:.3f}",
        (
            "Collisions: "
            f"total={score['infractions']['collision_count']} "
            f"unknown={score['infractions']['unknown_collision_count']}"
        ),
        (
            "Basic: "
            f"steps={basic['num_steps']} duration_s={basic['duration_s']:.3f} "
            f"distance_m={basic['distance_m']:.3f} termination={basic['termination_reason']}"
        ),
        (
            "Comfort: "
            f"mean_abs_jerk={_fmt_number(comfort.get('mean_abs_jerk_mps3'))} "
            f"max_abs_jerk={_fmt_number(comfort.get('max_abs_jerk_mps3'))} "
            f"over_limit={comfort.get('jerk_over_limit_count')}"
        ),
        (
            "Latency: "
            if latency.get("status") != "available"
            else f"Latency: p50={latency['p50_ms']:.3f}ms p95={latency['p95_ms']:.3f}ms n={latency['count']}"
        ),
        "Missing infraction event logging: " + (", ".join(missing) if missing else "none"),
    ]
    if latency.get("status") != "available":
        lines[-2] += str(latency.get("note"))
    return "\n".join(lines)


def format_eval_report(report: dict[str, Any]) -> str:
    chunks = [format_episode_score(score) for score in report["episodes"]]
    aggregate = report["aggregate"]
    metric_lines = []
    for name, stats in aggregate["metrics"].items():
        if stats["count"] == 0:
            metric_lines.append(f"{name}: N/A")
        else:
            ci = stats["bootstrap_ci95"]
            metric_lines.append(
                f"{name}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
                f"ci95=[{ci['low']:.4f}, {ci['high']:.4f}] n={stats['count']}"
            )
    chunks.append("Aggregate:\n" + "\n".join(metric_lines))
    missing = aggregate.get("missing_event_logging", [])
    chunks.append("Aggregate missing infraction event logging: " + (", ".join(missing) if missing else "none"))
    return "\n\n".join(chunks)


def write_eval_report(report: dict[str, Any], json_out: str | Path, summary_out: str | Path | None = None) -> None:
    json_path = Path(json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary_out is not None:
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(format_eval_report(report) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score closed-loop rollout directories.")
    parser.add_argument("rollout_dirs", nargs="+", help="Rollout directories with manifest.json + metadata.jsonl")
    parser.add_argument("--json-out", default="results/eval_report/eval_report.json")
    parser.add_argument("--summary-out", default="results/eval_report/eval_report.txt")
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--jerk-limit-mps3", type=float, default=DEFAULT_JERK_LIMIT_MPS3)
    args = parser.parse_args(argv)

    report = build_eval_report(
        args.rollout_dirs,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        jerk_limit_mps3=args.jerk_limit_mps3,
    )
    write_eval_report(report, json_out=args.json_out, summary_out=args.summary_out)
    print(format_eval_report(report))
    print(
        "eval report ok: "
        f"episodes={len(report['episodes'])} json={args.json_out} summary={args.summary_out}"
    )
    return 0


def _route_completion(records: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    explicit_pct = _last_number(records, (("route", "route_completion_pct"), ("route", "route_completed_pct")))
    if explicit_pct is not None:
        return {
            "value_pct": _clip(explicit_pct, 0.0, 100.0),
            "status": "available",
            "source": "route.route_completion_pct",
            "note": None,
        }

    progress = _max_number(records, (("route", "route_progress_m"),))
    route_length = _manifest_route_length(manifest)
    if progress is not None and route_length is not None and route_length > 0:
        return {
            "value_pct": _clip(100.0 * progress / route_length, 0.0, 100.0),
            "status": "available",
            "source": "route_progress_m / manifest.route_length_m",
            "route_length_m": route_length,
            "note": None,
        }

    distance_to_goal = _last_number(records, (("route", "distance_to_goal_m"),))
    if progress is not None and distance_to_goal is not None and progress + distance_to_goal > 0:
        return {
            "value_pct": _clip(100.0 * progress / (progress + distance_to_goal), 0.0, 100.0),
            "status": "available",
            "source": "route_progress_m / (route_progress_m + distance_to_goal_m)",
            "route_length_m": progress + distance_to_goal,
            "note": "distance_to_goal-derived route length; validate against route definition in T-021",
        }

    termination_reason = str(_dict(records[-1].get("termination")).get("reason", "")) if records else ""
    if termination_reason in {"route_completed", "success", "completed", "goal_reached"}:
        return {
            "value_pct": 100.0,
            "status": "available",
            "source": "termination.reason",
            "note": "coarse completion from termination reason",
        }

    return {
        "value_pct": None,
        "status": "not_available",
        "source": None,
        "note": "N/A - route_progress_m plus route_length/distance_to_goal logging required",
    }


def _infraction_breakdown(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    collision_count = 0
    unknown_collision_count = 0
    collision_type_counts = {
        "collisions_pedestrian": 0,
        "collisions_vehicle": 0,
        "collisions_layout": 0,
    }
    has_collision_logging = any(_has_collision_logging(_dict(record.get("events"))) for record in records)

    for record in records:
        events = _dict(record.get("events"))
        event_count = _collision_event_count(events)
        if event_count <= 0:
            continue
        collision_count += event_count
        collision_name = _collision_infraction_name(events.get("collision_actor_type"))
        if collision_name is None:
            unknown_collision_count += event_count
        else:
            collision_type_counts[collision_name] += event_count

    for name, count in collision_type_counts.items():
        by_type[name] = _available_infraction(
            count=count if has_collision_logging else None,
            penalty=LEADERBOARD_INFRACTION_PENALTIES[name],
            status="available" if has_collision_logging else "not_available",
            note=None if has_collision_logging else EVENT_LOGGING_REQUIRED,
        )

    by_type["collision_unknown"] = {
        "count": unknown_collision_count if has_collision_logging else None,
        "status": "available" if has_collision_logging else "not_available",
        "penalty_coefficient": None,
        "note": None if has_collision_logging else EVENT_LOGGING_REQUIRED,
    }

    for name in LEADERBOARD_INFRACTION_PENALTIES:
        if name in by_type:
            continue
        count = _count_explicit_infraction(records, name)
        by_type[name] = _available_infraction(
            count=count,
            penalty=LEADERBOARD_INFRACTION_PENALTIES[name],
            status="available" if count is not None else "not_available",
            note=None if count is not None else EVENT_LOGGING_REQUIRED,
        )

    for name, config in PERCENT_INFRACTION_PENALTIES.items():
        value = _percent_infraction_value(records, name)
        by_type[name] = {
            "count": 0 if value is not None and value <= 0 else (1 if value is not None else None),
            "status": "available" if value is not None else "not_available",
            "penalty_coefficient": config["coefficient"],
            "penalty_mode": config["mode"],
            "percentage": value,
            "note": None if value is not None else EVENT_LOGGING_REQUIRED,
        }

    for name in ("route_dev", "vehicle_blocked"):
        count = _count_explicit_infraction(records, name)
        by_type[name] = {
            "count": count,
            "status": "available" if count is not None else "not_available",
            "penalty_coefficient": None,
            "note": None if count is not None else EVENT_LOGGING_REQUIRED,
        }

    by_type["route_timeout"] = {
        "count": _route_timeout_count(records),
        "status": "available",
        "penalty_coefficient": None,
        "note": "Derived from termination.reason; no Leaderboard multiplicative coefficient in Bench2Drive table.",
    }
    return {
        "leaderboard_reference": LEADERBOARD_PENALTY_REFERENCE,
        "collision_count": collision_count if has_collision_logging else None,
        "unknown_collision_count": unknown_collision_count if has_collision_logging else None,
        "by_type": dict(sorted(by_type.items())),
    }


def _infraction_penalty(infractions: dict[str, Any]) -> dict[str, Any]:
    penalty = 1.0
    applied: list[dict[str, Any]] = []
    skipped: list[str] = []
    for name, entry in infractions["by_type"].items():
        if entry.get("status") != "available":
            skipped.append(name)
            continue
        count = entry.get("count")
        coefficient = entry.get("penalty_coefficient")
        if coefficient is None or count in (None, 0):
            continue
        if name in PERCENT_INFRACTION_PENALTIES:
            if entry.get("penalty_mode") == "increases":
                percentage = _clip(_float(entry.get("percentage")), 0.0, 100.0)
                factor = 1.0 - (1.0 - float(coefficient)) * percentage / 100.0
                penalty *= factor
                applied.append({"name": name, "factor": factor, "count": count})
            continue
        factor = float(coefficient) ** int(count)
        penalty *= factor
        applied.append({"name": name, "coefficient": coefficient, "count": count, "factor": factor})
    return {
        "value": penalty,
        "status": "partial_available_logs" if skipped else "leaderboard_comparable",
        "applied": applied,
        "skipped_not_available": skipped,
    }


def _comfort_metrics(records: Sequence[dict[str, Any]], jerk_limit_mps3: float) -> dict[str, Any]:
    samples: list[tuple[float, float]] = []
    for record in records:
        ego = _dict(record.get("ego"))
        timestamp = _optional_float(ego.get("timestamp_s"))
        acceleration = _optional_float(ego.get("acceleration_mps2"))
        if timestamp is not None and acceleration is not None:
            samples.append((timestamp, acceleration))
    jerk_values = []
    for (t0, a0), (t1, a1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt > 0:
            jerk_values.append((a1 - a0) / dt)
    return {
        "status": "available" if jerk_values else "not_available",
        "count": len(jerk_values),
        "mean_abs_jerk_mps3": _mean(abs(value) for value in jerk_values) if jerk_values else None,
        "max_abs_jerk_mps3": max((abs(value) for value in jerk_values), default=None),
        "jerk_over_limit_count": sum(1 for value in jerk_values if abs(value) > jerk_limit_mps3),
        "jerk_limit_mps3": jerk_limit_mps3,
        "note": None if jerk_values else "N/A - at least two timestamped acceleration samples required",
    }


def _latency_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for record in records:
        latency_ms = _latency_ms(record)
        if latency_ms is not None:
            values.append(latency_ms)
    if not values:
        return {
            "status": "not_available",
            "count": 0,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "note": "N/A - policy latency logging required",
        }
    ordered = sorted(values)
    return {
        "status": "available",
        "count": len(ordered),
        "p50_ms": _percentile(ordered, 50),
        "p90_ms": _percentile(ordered, 90),
        "p95_ms": _percentile(ordered, 95),
        "p99_ms": _percentile(ordered, 99),
        "max_ms": max(ordered),
        "note": None,
    }


def _aggregate_values(values: Sequence[float], bootstrap_iterations: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "bootstrap_ci95": {"low": None, "high": None}}
    mean = _mean(values)
    std = _std(values)
    if len(values) == 1 or bootstrap_iterations <= 0:
        ci_low = mean
        ci_high = mean
    else:
        rng = random.Random(seed)
        boot_means = []
        for _ in range(bootstrap_iterations):
            sample = [values[rng.randrange(len(values))] for _ in values]
            boot_means.append(_mean(sample))
        boot_means.sort()
        ci_low = _percentile(boot_means, 2.5)
        ci_high = _percentile(boot_means, 97.5)
    return {"count": len(values), "mean": mean, "std": std, "bootstrap_ci95": {"low": ci_low, "high": ci_high}}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _episode_id(records: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> str:
    if manifest.get("episode_id"):
        return str(manifest["episode_id"])
    if records and records[0].get("episode_id"):
        return str(records[0]["episode_id"])
    return "unknown"


def _manifest_route_length(manifest: dict[str, Any]) -> float | None:
    for key in ("route_length_m", "total_route_length_m", "route_length"):
        value = _optional_float(manifest.get(key))
        if value is not None:
            return value
    return None


def _collision_infraction_name(actor_type: Any) -> str | None:
    if not actor_type:
        return None
    value = str(actor_type).lower()
    if "walker" in value or "pedestrian" in value:
        return "collisions_pedestrian"
    if "vehicle" in value:
        return "collisions_vehicle"
    return "collisions_layout"


def _has_collision_logging(events: dict[str, Any]) -> bool:
    return any(key in events for key in ("collision_event_count", "collision_new", "collision"))


def _collision_event_count(events: dict[str, Any]) -> int:
    explicit_count = _optional_float(events.get("collision_event_count"))
    if explicit_count is not None:
        return max(0, int(explicit_count))
    if "collision_new" in events:
        return 1 if _is_truthy_infraction(events.get("collision_new")) else 0
    return 1 if _is_truthy_infraction(events.get("collision")) else 0


def _available_infraction(count: int | None, penalty: float, status: str, note: str | None) -> dict[str, Any]:
    return {"count": count, "status": status, "penalty_coefficient": penalty, "note": note}


def _count_explicit_infraction(records: Sequence[dict[str, Any]], name: str) -> int | None:
    aliases = EXPLICIT_EVENT_INFRACTION_FIELDS.get(name, ())
    seen = False
    count = 0
    for record in records:
        events = _dict(record.get("events"))
        for alias in aliases:
            if alias in events:
                value = events[alias]
                if value is None:
                    continue
                seen = True
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, dict)):
                    count += len(value)
                elif _is_truthy_infraction(value):
                    count += int(_float(value)) if isinstance(value, (int, float)) and float(value) > 1 else 1
    return count if seen else None


def _percent_infraction_value(records: Sequence[dict[str, Any]], name: str) -> float | None:
    aliases = EXPLICIT_EVENT_INFRACTION_FIELDS.get(name, ())
    values = []
    for record in records:
        events = _dict(record.get("events"))
        for alias in aliases:
            if alias in events and events[alias] is not None:
                values.append(_float(events[alias]))
    if not values:
        return None
    return max(values)


def _route_timeout_count(records: Sequence[dict[str, Any]]) -> int:
    if not records:
        return 0
    reason = str(_dict(records[-1].get("termination")).get("reason", "")).lower()
    return 1 if reason in {"timeout", "route_timeout", "max_steps"} else 0


def _latency_ms(record: dict[str, Any]) -> float | None:
    containers = [record, _dict(record.get("policy")), _dict(record.get("timing"))]
    ms_keys = ("latency_ms", "policy_latency_ms", "inference_latency_ms", "elapsed_ms")
    s_keys = ("latency_s", "policy_latency_s", "inference_latency_s", "elapsed_s")
    for container in containers:
        for key in ms_keys:
            value = _optional_float(container.get(key))
            if value is not None:
                return value
        for key in s_keys:
            value = _optional_float(container.get(key))
            if value is not None:
                return value * 1000.0
    return None


def _last_number(records: Sequence[dict[str, Any]], paths: Sequence[tuple[str, ...]]) -> float | None:
    for record in reversed(records):
        for path in paths:
            value = _optional_float(_get_path(record, path))
            if value is not None:
                return value
    return None


def _max_number(records: Sequence[dict[str, Any]], paths: Sequence[tuple[str, ...]]) -> float | None:
    values = []
    for record in records:
        for path in paths:
            value = _optional_float(_get_path(record, path))
            if value is not None:
                values.append(value)
    return max(values) if values else None


def _get_path(data: dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _valid_numbers(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        numeric = _optional_float(value)
        if numeric is not None and math.isfinite(numeric):
            result.append(numeric)
    return result


def _is_truthy_infraction(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) > 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "no"}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return len(value) > 0
    return bool(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _float(value: Any) -> float:
    numeric = _optional_float(value)
    return numeric if numeric is not None else 0.0


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[int(rank)])
    weight = rank - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


def _fmt_pct(value: Any) -> str:
    return f"{float(value):.2f}%"


def _fmt_number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVENT_LOGGING_REQUIRED",
    "LEADERBOARD_INFRACTION_PENALTIES",
    "LEADERBOARD_PENALTY_REFERENCE",
    "aggregate_scores",
    "build_eval_report",
    "format_episode_score",
    "format_eval_report",
    "main",
    "score_episode",
    "score_rollout_dir",
    "summarize_rollout_metrics",
    "write_eval_report",
]
