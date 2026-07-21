"""Stdlib-only QA summaries for CARLA IL datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROUTE_COMMANDS: tuple[str, ...] = (
    "lane_follow",
    "turn_left",
    "turn_right",
    "go_straight",
    "change_lane_left",
    "change_lane_right",
    "stop",
)

DEFAULT_CACHE_MIB_PER_FRAME = 1.457
DEFAULT_HISTOGRAM_BINS = 10
EPSILON = 1e-6


@dataclass
class _ActionAccumulator:
    count: int = 0
    nonzero_count: int = 0
    steer_nonzero_count: int = 0
    accel_nonzero_count: int = 0
    double_pedal_count: int = 0
    control_record_count: int = 0
    steer_values: list[float] = field(default_factory=list)
    accel_values: list[float] = field(default_factory=list)

    def add(self, action: "_Action") -> None:
        self.count += 1
        self.steer_values.append(action.steer)
        self.accel_values.append(action.acceleration)
        steer_nonzero = abs(action.steer) > EPSILON
        accel_nonzero = abs(action.acceleration) > EPSILON
        if steer_nonzero or accel_nonzero:
            self.nonzero_count += 1
        if steer_nonzero:
            self.steer_nonzero_count += 1
        if accel_nonzero:
            self.accel_nonzero_count += 1
        if action.has_control:
            self.control_record_count += 1
        if action.double_pedal:
            self.double_pedal_count += 1

    def to_dict(self, histogram_bins: int = DEFAULT_HISTOGRAM_BINS) -> dict[str, Any]:
        return {
            "count": self.count,
            "nonzero_count": self.nonzero_count,
            "nonzero_pct": _pct(self.nonzero_count, self.count),
            "steer_nonzero_count": self.steer_nonzero_count,
            "steer_nonzero_pct": _pct(self.steer_nonzero_count, self.count),
            "accel_nonzero_count": self.accel_nonzero_count,
            "accel_nonzero_pct": _pct(self.accel_nonzero_count, self.count),
            "double_pedal_count": self.double_pedal_count,
            "double_pedal_pct": _pct(self.double_pedal_count, self.count),
            "control_record_count": self.control_record_count,
            "control_record_pct": _pct(self.control_record_count, self.count),
            "steer": _value_summary(self.steer_values, histogram_bins=histogram_bins),
            "acceleration": _value_summary(self.accel_values, histogram_bins=histogram_bins),
        }


@dataclass(frozen=True)
class _Action:
    steer: float
    acceleration: float
    throttle: float | None
    brake: float | None
    has_control: bool

    @property
    def double_pedal(self) -> bool:
        return self.throttle is not None and self.brake is not None and self.throttle > EPSILON and self.brake > EPSILON


def summarize_dataset(
    dataset_root: str | Path,
    split_manifest: str | Path | None = None,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    cache_mib_per_frame: float = DEFAULT_CACHE_MIB_PER_FRAME,
) -> dict[str, Any]:
    """Return JSON-serializable QA stats for a dataset root or single episode."""

    root = Path(dataset_root)
    if histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive")
    episodes = _discover_episodes(root)
    split_labels = _load_split_labels(root, split_manifest)

    all_actions = _ActionAccumulator()
    saved_actions = _ActionAccumulator()
    route_all: Counter[str] = Counter()
    route_saved: Counter[str] = Counter()
    route_episodes: Counter[str] = Counter()
    weather_episodes: Counter[str] = Counter()
    town_episodes: Counter[str] = Counter()
    split_totals: dict[str, dict[str, Any]] = {}
    episode_summaries: list[dict[str, Any]] = []
    total_records = 0
    total_saved_records = 0
    missing_saved_frame_files = 0

    for episode_dir in episodes:
        manifest = _read_json(episode_dir / "manifest.json")
        metadata_rows = _read_jsonl(episode_dir / str(manifest.get("metadata_file", "metadata.jsonl")))
        split_name = split_labels.get(_path_key(episode_dir), "unsplit")
        split_summary = split_totals.setdefault(split_name, _new_split_summary())
        episode_all_actions = _ActionAccumulator()
        episode_saved_actions = _ActionAccumulator()
        episode_missing_frames = 0
        episode_saved_records = 0
        manifest_route = _label(manifest.get("route_command"))
        weather = _label(manifest.get("weather_preset"))
        town = _label(manifest.get("town") or manifest.get("map_name"))

        route_episodes[manifest_route] += 1
        weather_episodes[weather] += 1
        town_episodes[town] += 1
        split_summary["episodes"] += 1

        for row in metadata_rows:
            action = _action_from_record(row)
            route_command = _route_command(row, fallback=manifest_route)
            is_saved, missing_frame = _has_existing_saved_frame(row, episode_dir)

            total_records += 1
            split_summary["records"] += 1
            route_all[route_command] += 1
            all_actions.add(action)
            episode_all_actions.add(action)

            if is_saved:
                total_saved_records += 1
                episode_saved_records += 1
                split_summary["saved_frame_records"] += 1
                route_saved[route_command] += 1
                saved_actions.add(action)
                episode_saved_actions.add(action)
            elif missing_frame:
                missing_saved_frame_files += 1
                episode_missing_frames += 1
                split_summary["missing_saved_frame_files"] += 1

        episode_summaries.append(
            {
                "episode_dir": _display_path(episode_dir, root),
                "episode_id": str(manifest.get("episode_id", "")),
                "split": split_name,
                "route_command": manifest_route,
                "weather_preset": weather,
                "town": town,
                "manifest_num_steps": _optional_int(manifest.get("num_steps")),
                "manifest_num_saved_frames": _optional_int(manifest.get("num_saved_frames")),
                "metadata_records": len(metadata_rows),
                "saved_frame_records": episode_saved_records,
                "missing_saved_frame_files": episode_missing_frames,
                "action": {
                    "all_records": episode_all_actions.to_dict(histogram_bins=histogram_bins),
                    "saved_frame_records": episode_saved_actions.to_dict(histogram_bins=histogram_bins),
                },
            }
        )

    for split_summary in split_totals.values():
        split_summary["records_pct"] = _pct(split_summary["records"], total_records)
        split_summary["saved_frame_records_pct"] = _pct(split_summary["saved_frame_records"], total_saved_records)

    summary = {
        "schema_version": "dataset_stats_v1",
        "dataset_root": str(root),
        "split_manifest": _split_manifest_display(root, split_manifest),
        "histogram_bins": histogram_bins,
        "cache_mib_per_frame": cache_mib_per_frame,
        "totals": {
            "num_episodes": len(episodes),
            "num_records": total_records,
            "num_saved_frame_records": total_saved_records,
            "missing_saved_frame_files": missing_saved_frame_files,
            "estimated_feature_cache_gib": total_saved_records * cache_mib_per_frame / 1024.0,
        },
        "actions": {
            "all_records": all_actions.to_dict(histogram_bins=histogram_bins),
            "saved_frame_records": saved_actions.to_dict(histogram_bins=histogram_bins),
        },
        "route_commands": {
            "known_commands": list(ROUTE_COMMANDS),
            "all_records": dict(sorted(route_all.items())),
            "saved_frame_records": dict(sorted(route_saved.items())),
            "episodes": dict(sorted(route_episodes.items())),
            "missing_known_commands_in_saved_frames": [
                command for command in ROUTE_COMMANDS if route_saved.get(command, 0) == 0
            ],
        },
        "weather": {"episodes": dict(sorted(weather_episodes.items()))},
        "town": {"episodes": dict(sorted(town_episodes.items()))},
        "split": dict(sorted(split_totals.items())),
        "episodes": episode_summaries,
    }
    return summary


def format_dataset_summary(summary: dict[str, Any]) -> str:
    """Format a compact human-readable summary for terminal logs."""

    totals = summary["totals"]
    saved_actions = summary["actions"]["saved_frame_records"]
    all_actions = summary["actions"]["all_records"]
    steer = saved_actions["steer"]
    accel = saved_actions["acceleration"]
    route_saved = summary["route_commands"]["saved_frame_records"]
    weather = summary["weather"]["episodes"]
    town = summary["town"]["episodes"]
    split = summary["split"]

    lines = [
        f"Dataset: {summary['dataset_root']}",
        (
            "Episodes/records: "
            f"episodes={totals['num_episodes']} records={totals['num_records']} "
            f"saved_frame_records={totals['num_saved_frame_records']} "
            f"missing_saved_frame_files={totals['missing_saved_frame_files']}"
        ),
        (
            "Saved action nonzero: "
            f"{saved_actions['nonzero_count']}/{saved_actions['count']} "
            f"({saved_actions['nonzero_pct']:.1f}%) "
            f"steer={saved_actions['steer_nonzero_pct']:.1f}% "
            f"accel={saved_actions['accel_nonzero_pct']:.1f}% "
            f"double_pedal={saved_actions['double_pedal_count']} "
            f"({saved_actions['double_pedal_pct']:.1f}%)"
        ),
        (
            "All action nonzero: "
            f"{all_actions['nonzero_count']}/{all_actions['count']} "
            f"({all_actions['nonzero_pct']:.1f}%)"
        ),
        (
            "Saved steer range: "
            f"min={_fmt_number(steer['min'])} max={_fmt_number(steer['max'])} "
            f"mean={_fmt_number(steer['mean'])} abs_mean={_fmt_number(steer['abs_mean'])}"
        ),
        (
            "Saved accel range: "
            f"min={_fmt_number(accel['min'])} max={_fmt_number(accel['max'])} "
            f"mean={_fmt_number(accel['mean'])} abs_mean={_fmt_number(accel['abs_mean'])}"
        ),
        f"Route commands (saved): {_format_counter(route_saved)}",
        f"Weather presets (episodes): {_format_counter(weather)}",
        f"Town/map coverage (episodes): {_format_counter(town)}",
        f"Train/val balance: {_format_split(split)}",
        f"Estimated feature cache: {totals['estimated_feature_cache_gib']:.2f} GiB",
    ]
    return "\n".join(lines)


def write_dataset_stats(
    summary: dict[str, Any],
    json_out: str | Path | None = None,
    summary_out: str | Path | None = None,
) -> None:
    if json_out is not None:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary_out is not None:
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(format_dataset_summary(summary) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize CARLA IL dataset QA stats.")
    parser.add_argument("dataset_root", help="Dataset root or single episode directory.")
    parser.add_argument("--split-manifest", default=None, help="Optional episode split manifest path.")
    parser.add_argument("--json-out", default=None, help="Optional JSON output path.")
    parser.add_argument("--summary-out", default=None, help="Optional text summary output path.")
    parser.add_argument("--histogram-bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    parser.add_argument("--cache-mib-per-frame", type=float, default=DEFAULT_CACHE_MIB_PER_FRAME)
    args = parser.parse_args(argv)

    summary = summarize_dataset(
        args.dataset_root,
        split_manifest=args.split_manifest,
        histogram_bins=args.histogram_bins,
        cache_mib_per_frame=args.cache_mib_per_frame,
    )
    write_dataset_stats(summary, json_out=args.json_out, summary_out=args.summary_out)
    print(format_dataset_summary(summary))
    if args.json_out or args.summary_out:
        print(
            "dataset stats ok: "
            f"episodes={summary['totals']['num_episodes']} "
            f"saved_frames={summary['totals']['num_saved_frame_records']} "
            f"saved_nonzero_pct={summary['actions']['saved_frame_records']['nonzero_pct']:.1f} "
            f"json={args.json_out or ''} summary={args.summary_out or ''}"
        )
    return 0


def _discover_episodes(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if (root / "manifest.json").exists() and (root / "metadata.jsonl").exists():
        return [root]
    episodes = {path.parent for path in root.rglob("manifest.json") if (path.parent / "metadata.jsonl").exists()}
    if not episodes:
        raise ValueError(f"no episodes with manifest.json + metadata.jsonl found under {root}")
    return sorted(episodes, key=lambda path: path.as_posix())


def _load_split_labels(root: Path, split_manifest: str | Path | None) -> dict[str, str]:
    manifest_path = Path(split_manifest) if split_manifest is not None else root / "split_manifest.json"
    if not manifest_path.exists():
        return {}
    data = _read_json(manifest_path)
    if data.get("schema_version") != "episode_split_v1":
        raise ValueError(f"unsupported split manifest schema: {data.get('schema_version')}")
    labels: dict[str, str] = {}
    for entry in data.get("episodes", []):
        episode_dir = Path(str(entry["episode_dir"]))
        if not episode_dir.is_absolute():
            episode_dir = manifest_path.parent / episode_dir
        labels[_path_key(episode_dir)] = str(entry["split"])
    return labels


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _action_from_record(record: dict[str, Any]) -> _Action:
    control = _dict(record.get("expert_control")) or _dict(record.get("control"))
    if control:
        throttle = _to_float(control.get("throttle"))
        brake = _to_float(control.get("brake"))
        return _Action(
            steer=_clip(_to_float(control.get("steer")), -1.0, 1.0),
            acceleration=_clip(throttle - brake, -1.0, 1.0),
            throttle=throttle,
            brake=brake,
            has_control=True,
        )

    action = _dict(record.get("action"))
    return _Action(
        steer=_clip(_to_float(action.get("steer")), -1.0, 1.0),
        acceleration=_clip(_to_float(action.get("acceleration")), -1.0, 1.0),
        throttle=None,
        brake=None,
        has_control=False,
    )


def _route_command(record: dict[str, Any], fallback: str) -> str:
    route = _dict(record.get("route"))
    return _label(route.get("command") or fallback)


def _has_existing_saved_frame(record: dict[str, Any], episode_dir: Path) -> tuple[bool, bool]:
    camera = _dict(record.get("camera"))
    camera_path = camera.get("path")
    if not camera_path:
        return False, False
    frame_path = episode_dir / str(camera_path)
    return frame_path.exists(), not frame_path.exists()


def _new_split_summary() -> dict[str, Any]:
    return {
        "episodes": 0,
        "records": 0,
        "records_pct": 0.0,
        "saved_frame_records": 0,
        "saved_frame_records_pct": 0.0,
        "missing_saved_frame_files": 0,
    }


def _value_summary(values: Sequence[float], histogram_bins: int) -> dict[str, Any]:
    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "abs_mean": None,
            "histogram": _histogram([], bins=histogram_bins),
        }
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "abs_mean": sum(abs(value) for value in values) / len(values),
        "histogram": _histogram(values, bins=histogram_bins),
    }


def _histogram(values: Sequence[float], bins: int, lower: float = -1.0, upper: float = 1.0) -> dict[str, Any]:
    width = (upper - lower) / bins
    counts = [0 for _ in range(bins)]
    below_range = 0
    above_range = 0
    for value in values:
        if value < lower:
            below_range += 1
        elif value > upper:
            above_range += 1
        else:
            index = int((value - lower) / width)
            if index == bins:
                index = bins - 1
            counts[index] += 1
    edges = [lower + index * width for index in range(bins + 1)]
    return {
        "range": [lower, upper],
        "edges": edges,
        "counts": counts,
        "below_range": below_range,
        "above_range": above_range,
    }


def _split_manifest_display(root: Path, split_manifest: str | Path | None) -> str | None:
    manifest_path = Path(split_manifest) if split_manifest is not None else root / "split_manifest.json"
    return str(manifest_path) if manifest_path.exists() else None


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _path_key(path: Path) -> str:
    return path.resolve(strict=False).as_posix()


def _format_counter(counter: dict[str, int]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def _format_split(split: dict[str, dict[str, Any]]) -> str:
    if not split:
        return "unsplit"
    parts = []
    for split_name, data in sorted(split.items()):
        parts.append(
            f"{split_name}:episodes={data['episodes']},saved={data['saved_frame_records']}"
            f"({data['saved_frame_records_pct']:.1f}%)"
        )
    return "; ".join(parts)


def _fmt_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _label(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ROUTE_COMMANDS", "format_dataset_summary", "main", "summarize_dataset", "write_dataset_stats"]
