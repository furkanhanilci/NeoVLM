"""Rollout logging helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    dataset_name: str
    episode_id: str
    metadata_file: str
    frames_dir: str
    num_steps: int
    num_saved_frames: int
    control_mode: str
    map_name: str
    fixed_delta_seconds: float
    image_width: int
    image_height: int
    route_command: str
    target_speed_mps: float
    weather_preset: str | None = None
    route_length_m: float | None = None
    route_seed: int | None = None
    destination_spawn_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlRolloutLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.frames_dir = output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = output_dir / "metadata.jsonl"
        self.manifest_path = output_dir / "manifest.json"
        self._file = self.metadata_path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()

    def write_manifest(self, manifest: DatasetManifest) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JsonlRolloutLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
