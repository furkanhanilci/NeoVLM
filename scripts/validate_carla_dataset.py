#!/usr/bin/env python3
"""Validate a CARLA rollout dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_RECORD_KEYS = {
    "episode_id",
    "step",
    "carla_frame",
    "ego",
    "route",
    "action",
    "control",
    "camera",
    "termination",
    "policy",
    "events",
    "expert_control",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = REQUIRED_RECORD_KEYS - set(record)
            if missing:
                raise ValueError(f"line {line_no} missing keys: {sorted(missing)}")
            records.append(record)
    return records


def validate_dataset(dataset_dir: Path) -> None:
    manifest_path = dataset_dir / "manifest.json"
    metadata_path = dataset_dir / "metadata.jsonl"
    frames_dir = dataset_dir / "frames"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata: {metadata_path}")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"missing frames dir: {frames_dir}")

    manifest = _load_json(manifest_path)
    records = _load_jsonl(metadata_path)
    frame_files = sorted(frames_dir.glob("*.png"))

    if len(records) != manifest["num_steps"]:
        raise ValueError(f"num_steps mismatch: manifest={manifest['num_steps']} records={len(records)}")
    if len(frame_files) != manifest["num_saved_frames"]:
        raise ValueError(
            f"num_saved_frames mismatch: manifest={manifest['num_saved_frames']} files={len(frame_files)}"
        )
    if not records:
        raise ValueError("metadata has no records")
    if records[0]["step"] != 0:
        raise ValueError("first record step must be 0")
    if records[-1]["termination"]["done"] is not True:
        raise ValueError("last record must have termination.done=true")
    if records[-1]["termination"]["reason"] not in {"max_steps", "collision"}:
        raise ValueError("last termination reason must be max_steps or collision")

    expected_episode = manifest["episode_id"]
    for idx, record in enumerate(records):
        if record["episode_id"] != expected_episode:
            raise ValueError(f"line {idx + 1} episode_id mismatch")
        if record["step"] != idx:
            raise ValueError(f"line {idx + 1} step mismatch: expected {idx}, got {record['step']}")
        camera_path = record["camera"].get("path")
        if camera_path is not None and not (dataset_dir / camera_path).exists():
            raise ValueError(f"line {idx + 1} camera path missing: {camera_path}")
        if manifest["control_mode"] == "autopilot" and record["expert_control"] is None:
            raise ValueError(f"line {idx + 1} missing expert_control for autopilot dataset")

    print(
        f"dataset ok: {dataset_dir} records={len(records)} frames={len(frame_files)} "
        f"mode={manifest['control_mode']} map={manifest['map_name']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    validate_dataset(args.dataset_dir)


if __name__ == "__main__":
    main()
