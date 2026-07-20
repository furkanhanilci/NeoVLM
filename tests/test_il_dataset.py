import json
from pathlib import Path

import pytest
import torch

from vlm_driving.config import ExperimentConfig, PolicyConfig, ResamplerConfig, VLMConfig
from vlm_driving.data.il_dataset import ILDataset, OBSERVATION_FEATURES
from vlm_driving.vlm import FeatureCacheManifest, FeatureCacheRecord


def _write_jsonl(path: Path, rows: list[dict]):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _record(step: int, camera_path: str | None, steer: float, throttle: float, brake: float = 0.0) -> dict:
    return {
        "episode_id": "episode_000",
        "step": step,
        "carla_frame": 1000 + step,
        "ego": {
            "frame": 1000 + step,
            "timestamp_s": step * 0.05,
            "x": 1.0,
            "y": 2.0,
            "z": 0.0,
            "yaw_deg": 90.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "speed_mps": 6.0,
            "acceleration_mps2": 2.0,
            "angular_velocity_z_dps": 18.0,
        },
        "route": {
            "command": "lane_follow",
            "target_speed_mps": 9.0,
            "route_progress_m": 25.0,
            "distance_to_goal_m": 100.0,
        },
        "camera": {
            "sensor_name": "rgb_front",
            "frame": 1000 + step,
            "timestamp_s": step * 0.05,
            "path": camera_path,
            "width": 800,
            "height": 450,
        },
        "action": {"steer": steer, "acceleration": throttle - brake},
        "control": {"steer": steer, "throttle": throttle, "brake": brake, "hand_brake": False, "reverse": False},
        "expert_control": {"steer": steer, "throttle": throttle, "brake": brake, "hand_brake": False, "reverse": False},
        "termination": {"done": False, "reason": "running"},
        "policy": {"name": "expert", "control_mode": "autopilot", "is_expert": True},
        "events": {"collision": False, "collision_actor_type": None, "collision_impulse": None},
    }


def _build_episode(tmp_path: Path) -> Path:
    episode_dir = tmp_path / "episode_000"
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_00000.png").write_bytes(b"frame0")
    (frames_dir / "frame_00005.png").write_bytes(b"frame5")
    rows = [
        _record(step=0, camera_path="frames/frame_00000.png", steer=0.1, throttle=0.25),
        _record(step=1, camera_path=None, steer=0.2, throttle=0.5),
        _record(step=5, camera_path="frames/frame_00005.png", steer=-0.3, throttle=0.0, brake=0.75),
    ]
    _write_jsonl(episode_dir / "metadata.jsonl", rows)
    return episode_dir


def _build_cache(tmp_path: Path, config: ExperimentConfig) -> Path:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    hidden0 = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16)
    hidden5 = torch.arange(20, dtype=torch.float32).reshape(5, 4).to(torch.bfloat16)
    torch.save(hidden0, cache_dir / "frame_00000.pt")
    torch.save(hidden5, cache_dir / "frame_00005.pt")
    manifest = FeatureCacheManifest(
        schema_version="vlm_hidden_cache_v1",
        model_id=config.vlm.model_id,
        hidden_size=config.vlm.hidden_size,
        precision="bf16",
        command_text="keep lane",
        num_frames=2,
        records=[
            FeatureCacheRecord(
                frame_key="frame_00000.png",
                source_frame="frames/frame_00000.png",
                cache_file="frame_00000.pt",
                shape=[3, 4],
                dtype=str(hidden0.dtype),
            ),
            FeatureCacheRecord(
                frame_key="frame_00005.png",
                source_frame="frames/frame_00005.png",
                cache_file="frame_00005.pt",
                shape=[5, 4],
                dtype=str(hidden5.dtype),
            ),
        ],
    )
    manifest.write(cache_dir / "cache_manifest.json")
    return cache_dir


def _tiny_config() -> ExperimentConfig:
    return ExperimentConfig(vlm=VLMConfig(hidden_size=4), resampler=ResamplerConfig(input_dim=4))


def test_il_dataset_reads_saved_frames_and_cached_hidden_states(tmp_path: Path):
    config = _tiny_config()
    episode_dir = _build_episode(tmp_path)
    cache_dir = _build_cache(tmp_path, config)

    dataset = ILDataset(episode_dir, feature_cache_dir=cache_dir, config=config)

    assert len(dataset) == 2
    first = dataset[0]
    second = dataset[1]
    assert first["frame_key"] == "frame_00000.png"
    assert second["frame_key"] == "frame_00005.png"
    assert first["observation"].shape == (config.policy.obs_dim,)
    assert first["cached_hidden"].shape == (3, 4)
    assert first["cached_hidden"].dtype == torch.bfloat16
    assert torch.equal(first["expert_action"], torch.tensor([0.1, 0.25], dtype=torch.float32))
    assert torch.equal(second["expert_action"], torch.tensor([-0.3, -0.75], dtype=torch.float32))

    route_index = OBSERVATION_FEATURES.index("route_lane_follow")
    previous_steer_index = OBSERVATION_FEATURES.index("previous_steer")
    previous_accel_index = OBSERVATION_FEATURES.index("previous_acceleration")
    assert first["observation"][route_index].item() == 1.0
    assert first["observation"][previous_steer_index].item() == 0.0
    assert second["observation"][previous_steer_index].item() == pytest.approx(0.2)
    assert second["observation"][previous_accel_index].item() == pytest.approx(0.5)
    assert torch.count_nonzero(first["observation"][len(OBSERVATION_FEATURES) :]).item() == 0


def test_il_dataset_can_run_without_feature_cache(tmp_path: Path):
    episode_dir = _build_episode(tmp_path)

    dataset = ILDataset([episode_dir], config=_tiny_config())

    assert len(dataset) == 2
    assert dataset[0]["cached_hidden"] is None


def test_il_dataset_control_action_mapping_uses_throttle_minus_brake(tmp_path: Path):
    episode_dir = tmp_path / "episode_000"
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_00000.png").write_bytes(b"frame")
    _write_jsonl(
        episode_dir / "metadata.jsonl",
        [_record(step=0, camera_path="frames/frame_00000.png", steer=0.4, throttle=0.7, brake=0.2)],
    )

    dataset = ILDataset(episode_dir, config=_tiny_config())

    assert torch.equal(dataset[0]["expert_action"], torch.tensor([0.4, 0.5], dtype=torch.float32))


def test_il_dataset_validates_two_dim_action_config(tmp_path: Path):
    bad_config = ExperimentConfig(policy=PolicyConfig(action_dim=3, residual_limit=(0.1, 0.2, 0.3)))

    with pytest.raises(ValueError, match="action_dim == 2"):
        ILDataset(tmp_path, config=bad_config)
