import json
from pathlib import Path

from vlm_driving.data.dataset_stats import format_dataset_summary, summarize_dataset


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _row(step: int, camera_path: str | None, command: str, steer: float, throttle: float, brake: float = 0.0) -> dict:
    return {
        "step": step,
        "route": {"command": command, "target_speed_mps": 6.0},
        "camera": {"path": camera_path, "width": 800, "height": 450},
        "expert_control": {"steer": steer, "throttle": throttle, "brake": brake},
        "action": {"steer": steer, "acceleration": throttle - brake},
    }


def _episode(
    root: Path,
    name: str,
    command: str,
    weather: str,
    town: str,
    rows: list[dict],
    saved_frames: list[str],
) -> Path:
    episode_dir = root / name
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True)
    for frame_name in saved_frames:
        (frames_dir / frame_name).write_bytes(b"frame")
    _write_json(
        episode_dir / "manifest.json",
        {
            "schema_version": "carla_rollout_v1",
            "episode_id": name,
            "metadata_file": "metadata.jsonl",
            "num_steps": len(rows),
            "num_saved_frames": len(saved_frames),
            "route_command": command,
            "weather_preset": weather,
            "map_name": town,
        },
    )
    _write_jsonl(episode_dir / "metadata.jsonl", rows)
    return episode_dir


def test_summarize_dataset_reports_action_coverage_and_split_balance(tmp_path: Path):
    root = tmp_path / "dataset"
    train_episode = _episode(
        root,
        "episode_0000",
        command="lane_follow",
        weather="ClearNoon",
        town="Town01",
        rows=[
            _row(0, "frames/frame_00000.png", "lane_follow", steer=0.0, throttle=0.0),
            _row(1, "frames/frame_00001.png", "lane_follow", steer=0.2, throttle=0.5),
            _row(2, None, "lane_follow", steer=0.0, throttle=0.0),
        ],
        saved_frames=["frame_00000.png", "frame_00001.png"],
    )
    val_episode = _episode(
        root,
        "episode_0001",
        command="turn_left",
        weather="WetNoon",
        town="Town02",
        rows=[_row(0, "frames/frame_00000.png", "turn_left", steer=-0.4, throttle=0.3, brake=0.1)],
        saved_frames=["frame_00000.png"],
    )
    _write_json(
        root / "split_manifest.json",
        {
            "schema_version": "episode_split_v1",
            "num_episodes": 2,
            "num_train": 1,
            "num_val": 1,
            "episodes": [
                {"episode_dir": train_episode.name, "split": "train"},
                {"episode_dir": val_episode.name, "split": "val"},
            ],
        },
    )

    summary = summarize_dataset(root, histogram_bins=4)

    assert summary["schema_version"] == "dataset_stats_v1"
    assert summary["totals"]["num_episodes"] == 2
    assert summary["totals"]["num_records"] == 4
    assert summary["totals"]["num_saved_frame_records"] == 3
    assert summary["actions"]["saved_frame_records"]["nonzero_count"] == 2
    assert summary["actions"]["saved_frame_records"]["nonzero_pct"] == 100.0 * 2 / 3
    assert summary["actions"]["saved_frame_records"]["double_pedal_count"] == 1
    assert summary["route_commands"]["saved_frame_records"] == {"lane_follow": 2, "turn_left": 1}
    assert summary["weather"]["episodes"] == {"ClearNoon": 1, "WetNoon": 1}
    assert summary["town"]["episodes"] == {"Town01": 1, "Town02": 1}
    assert summary["split"]["train"]["saved_frame_records"] == 2
    assert summary["split"]["val"]["saved_frame_records"] == 1
    assert sum(summary["actions"]["saved_frame_records"]["steer"]["histogram"]["counts"]) == 3


def test_format_dataset_summary_is_human_readable(tmp_path: Path):
    root = tmp_path / "dataset"
    _episode(
        root,
        "episode_0000",
        command="lane_follow",
        weather="ClearNoon",
        town="Town01",
        rows=[_row(0, "frames/frame_00000.png", "lane_follow", steer=0.1, throttle=0.2)],
        saved_frames=["frame_00000.png"],
    )

    text = format_dataset_summary(summarize_dataset(root))

    assert "Saved action nonzero: 1/1 (100.0%)" in text
    assert "Route commands (saved): lane_follow=1" in text
    assert "Estimated feature cache:" in text
