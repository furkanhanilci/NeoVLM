"""Dataset utilities for behavior cloning on CARLA rollout logs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from vlm_driving.config import ExperimentConfig
from vlm_driving.vlm.feature_cache import CachedFeatureReader

ROUTE_COMMANDS: tuple[str, ...] = (
    "lane_follow",
    "turn_left",
    "turn_right",
    "go_straight",
    "change_lane_left",
    "change_lane_right",
    "stop",
)

OBSERVATION_FEATURES: tuple[str, ...] = (
    "speed_mps_over_30",
    "acceleration_mps2_over_10",
    "angular_velocity_z_dps_over_180",
    "sin_yaw",
    "cos_yaw",
    "pitch_deg_over_90",
    "roll_deg_over_90",
    "target_speed_mps_over_30",
    "route_progress_m_over_1000",
    "distance_to_goal_m_over_1000",
    "route_lane_follow",
    "route_turn_left",
    "route_turn_right",
    "route_go_straight",
    "route_change_lane_left",
    "route_change_lane_right",
    "route_stop",
    "previous_steer",
    "previous_acceleration",
)


@dataclass(frozen=True)
class ILDatasetRecord:
    episode_dir: Path
    raw: dict[str, Any]
    frame_key: str
    previous_action: tuple[float, float]


class ILDataset(Dataset):
    """Reads M3 rollout metadata and optional frozen-VLM hidden-state cache.

    Only metadata rows with an actually saved camera frame are used. If a feature
    cache is provided, rows whose frame key is missing from that cache are skipped
    as well. This keeps observation/hidden-state alignment explicit.
    """

    def __init__(
        self,
        episode_dirs: Sequence[str | Path] | str | Path,
        config: ExperimentConfig | None = None,
        feature_cache_dir: str | Path | None = None,
        metadata_filename: str = "metadata.jsonl",
    ) -> None:
        self.config = config or ExperimentConfig()
        if self.config.policy.action_dim != 2:
            raise ValueError("ILDataset expects policy.action_dim == 2 for [steer, acceleration]")
        if self.config.policy.obs_dim < len(OBSERVATION_FEATURES):
            raise ValueError(
                f"policy.obs_dim={self.config.policy.obs_dim} is smaller than "
                f"the {len(OBSERVATION_FEATURES)} documented IL observation features"
            )

        self.episode_dirs = _as_episode_dirs(episode_dirs)
        self.reader: CachedFeatureReader | None = None
        cache_keys: set[str] | None = None
        if feature_cache_dir is not None:
            self.reader = CachedFeatureReader(
                feature_cache_dir,
                expected_model_id=self.config.vlm.model_id,
                expected_hidden_size=self.config.vlm.hidden_size,
            )
            cache_keys = set(self.reader.keys())

        records: list[ILDatasetRecord] = []
        for episode_dir in self.episode_dirs:
            metadata_path = episode_dir / metadata_filename
            if not metadata_path.exists():
                raise FileNotFoundError(f"metadata file does not exist: {metadata_path}")

            previous_action = (0.0, 0.0)
            for raw in _read_jsonl(metadata_path):
                frame_key = _saved_frame_key(raw, episode_dir)
                if frame_key is not None and (cache_keys is None or frame_key in cache_keys):
                    records.append(
                        ILDatasetRecord(
                            episode_dir=episode_dir,
                            raw=raw,
                            frame_key=frame_key,
                            previous_action=previous_action,
                        )
                    )
                previous_action = _expert_action(raw)

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cached_hidden = None
        if self.reader is not None:
            cached_hidden = self.reader.read(record.frame_key)

        return {
            "observation": featurize_observation(
                record.raw,
                previous_action=record.previous_action,
                obs_dim=self.config.policy.obs_dim,
            ),
            "cached_hidden": cached_hidden,
            "expert_action": torch.tensor(_expert_action(record.raw), dtype=torch.float32),
            "frame_key": record.frame_key,
            "episode_id": str(record.raw.get("episode_id", "")),
            "step": int(record.raw.get("step", -1)),
        }


def featurize_observation(
    record: dict[str, Any],
    previous_action: tuple[float, float] = (0.0, 0.0),
    obs_dim: int = 64,
) -> torch.Tensor:
    """Converts one rollout metadata row into the fixed 64-dim policy input.

    The first len(OBSERVATION_FEATURES) positions are documented above; remaining
    dimensions are zero padding reserved for short history or future features.
    """

    if obs_dim < len(OBSERVATION_FEATURES):
        raise ValueError("obs_dim is smaller than the documented observation feature count")

    ego = _dict(record.get("ego"))
    route = _dict(record.get("route"))
    command = str(route.get("command", "lane_follow"))
    yaw_rad = math.radians(_to_float(ego.get("yaw_deg")))

    values = [
        _clip(_to_float(ego.get("speed_mps")) / 30.0, -1.0, 1.0),
        _clip(_to_float(ego.get("acceleration_mps2")) / 10.0, -1.0, 1.0),
        _clip(_to_float(ego.get("angular_velocity_z_dps")) / 180.0, -1.0, 1.0),
        math.sin(yaw_rad),
        math.cos(yaw_rad),
        _clip(_to_float(ego.get("pitch_deg")) / 90.0, -1.0, 1.0),
        _clip(_to_float(ego.get("roll_deg")) / 90.0, -1.0, 1.0),
        _clip(_to_float(route.get("target_speed_mps")) / 30.0, -1.0, 1.0),
        _clip(_to_float(route.get("route_progress_m")) / 1000.0, 0.0, 1.0),
        _clip(_to_float(route.get("distance_to_goal_m")) / 1000.0, 0.0, 1.0),
    ]
    values.extend(1.0 if command == route_command else 0.0 for route_command in ROUTE_COMMANDS)
    values.extend([_clip(previous_action[0], -1.0, 1.0), _clip(previous_action[1], -1.0, 1.0)])

    observation = torch.zeros(obs_dim, dtype=torch.float32)
    observation[: len(values)] = torch.tensor(values, dtype=torch.float32)
    return observation


def _as_episode_dirs(episode_dirs: Sequence[str | Path] | str | Path) -> list[Path]:
    if isinstance(episode_dirs, (str, Path)):
        return [Path(episode_dirs)]
    return [Path(path) for path in episode_dirs]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _saved_frame_key(record: dict[str, Any], episode_dir: Path) -> str | None:
    camera = _dict(record.get("camera"))
    camera_path = camera.get("path")
    if not camera_path:
        return None
    frame_path = episode_dir / str(camera_path)
    if not frame_path.exists():
        return None
    return frame_path.name


def _expert_action(record: dict[str, Any]) -> tuple[float, float]:
    expert_control = record.get("expert_control")
    if isinstance(expert_control, dict):
        return _action_from_control(expert_control)

    action = record.get("action")
    if isinstance(action, dict):
        return (
            _clip(_to_float(action.get("steer")), -1.0, 1.0),
            _clip(_to_float(action.get("acceleration")), -1.0, 1.0),
        )

    control = record.get("control")
    if isinstance(control, dict):
        return _action_from_control(control)

    raise ValueError("metadata row has no expert_control, action, or control field")


def _action_from_control(control: dict[str, Any]) -> tuple[float, float]:
    steer = _clip(_to_float(control.get("steer")), -1.0, 1.0)
    acceleration = _clip(_to_float(control.get("throttle")) - _to_float(control.get("brake")), -1.0, 1.0)
    return steer, acceleration


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


__all__ = ["ILDataset", "ILDatasetRecord", "OBSERVATION_FEATURES", "ROUTE_COMMANDS", "featurize_observation"]
