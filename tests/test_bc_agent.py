from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vlm_driving.carla.bc_agent import BCAgent
from vlm_driving.carla.metrics import summarize_rollout_metrics, write_rollout_metrics
from vlm_driving.config import ExperimentConfig, PolicyConfig, ResamplerConfig, TrainConfig, VLMConfig
from vlm_driving.training import build_bc_models, save_bc_checkpoint
from vlm_driving.vlm import FeatureCacheManifest, FeatureCacheRecord


class StaticHiddenProvider:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.hidden = hidden
        self.calls: list[tuple[Any, str]] = []

    def encode_observation(self, images: Any, command_text: str) -> torch.Tensor:
        self.calls.append((images, command_text))
        return self.hidden.unsqueeze(0)


def _tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=5,
        vlm=VLMConfig(model_id="synthetic/model", hidden_size=4, token_dim=8),
        resampler=ResamplerConfig(num_queries=2, input_dim=4, output_dim=8, num_heads=2, dropout=0.0),
        policy=PolicyConfig(obs_dim=64, token_dim=8, hidden_dim=12, action_dim=2, residual_limit=(0.1, 0.1)),
        train=TrainConfig(batch_size=1, epochs=1, learning_rate=1e-3, checkpoint_path="unused.pt"),
    )


def _checkpoint(tmp_path: Path, config: ExperimentConfig) -> Path:
    resampler, policy = build_bc_models(config, device="cpu")
    path = tmp_path / "bc.pt"
    save_bc_checkpoint(path, config, resampler, policy, loss_history=[0.5, 0.25])
    return path


def _record(camera_path: str | None = "frames/frame_00000.png") -> dict:
    return {
        "ego": {
            "speed_mps": 4.0,
            "acceleration_mps2": 0.5,
            "angular_velocity_z_dps": 0.0,
            "yaw_deg": 10.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "timestamp_s": 1.0,
            "x": 1.0,
            "y": 2.0,
        },
        "route": {
            "command": "lane_follow",
            "target_speed_mps": 6.0,
            "route_progress_m": 3.0,
            "distance_to_goal_m": 20.0,
        },
        "camera": {"path": camera_path, "width": 800, "height": 450},
    }


def test_bc_agent_live_provider_outputs_clipped_normalized_action(tmp_path: Path):
    config = _tiny_config()
    checkpoint_path = _checkpoint(tmp_path, config)
    hidden = torch.randn(3, config.vlm.hidden_size)
    provider = StaticHiddenProvider(hidden)
    agent = BCAgent(
        checkpoint_path=checkpoint_path,
        hidden_source="live",
        hidden_provider=provider,
        command_text="keep lane",
        device="cpu",
    )

    output = agent.act_with_debug(_record(), image="frame.png")

    assert output.hidden_shape == (3, config.vlm.hidden_size)
    assert -1.0 <= output.action.steer <= 1.0
    assert -1.0 <= output.action.acceleration <= 1.0
    assert output.raw_action.shape == (2,)
    assert agent.previous_action == (output.action.steer, output.action.acceleration)
    assert provider.calls == [("frame.png", "keep lane")]


def test_bc_agent_cache_source_reads_frame_key(tmp_path: Path):
    config = _tiny_config()
    checkpoint_path = _checkpoint(tmp_path, config)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    hidden = torch.randn(5, config.vlm.hidden_size, dtype=torch.float32).to(torch.bfloat16)
    torch.save(hidden, cache_dir / "frame_00000.pt")
    manifest = FeatureCacheManifest(
        schema_version="vlm_hidden_cache_v1",
        model_id=config.vlm.model_id,
        hidden_size=config.vlm.hidden_size,
        precision="bf16",
        command_text="keep lane",
        num_frames=1,
        records=[
            FeatureCacheRecord(
                frame_key="frame_00000.png",
                source_frame="frames/frame_00000.png",
                cache_file="frame_00000.pt",
                shape=list(hidden.shape),
                dtype=str(hidden.dtype),
            )
        ],
    )
    manifest.write(cache_dir / "cache_manifest.json")
    agent = BCAgent(
        checkpoint_path=checkpoint_path,
        hidden_source="cache",
        feature_cache_dir=cache_dir,
        device="cpu",
    )

    action = agent.act(_record("frames/frame_00000.png"))

    assert -1.0 <= action.steer <= 1.0
    assert -1.0 <= action.acceleration <= 1.0


def test_rollout_metrics_summary_and_write(tmp_path: Path):
    records = [
        {
            "ego": {"timestamp_s": 0.0, "x": 0.0, "y": 0.0, "speed_mps": 2.0},
            "action": {"steer": -0.2, "acceleration": 0.1},
            "control": {"throttle": 0.1, "brake": 0.0},
            "events": {"collision": False},
            "termination": {"reason": "running"},
        },
        {
            "ego": {"timestamp_s": 2.0, "x": 3.0, "y": 4.0, "speed_mps": 4.0},
            "action": {"steer": 0.4, "acceleration": -0.1},
            "control": {"throttle": 0.0, "brake": 0.1},
            "events": {"collision": True},
            "termination": {"reason": "collision"},
        },
    ]

    metrics = summarize_rollout_metrics(records)
    written = write_rollout_metrics(tmp_path / "metrics.json", records)

    assert metrics["num_steps"] == 2
    assert metrics["duration_s"] == 2.0
    assert metrics["distance_m"] == 5.0
    assert metrics["mean_speed_mps"] == 3.0
    assert metrics["collision_count"] == 1
    assert metrics["termination_reason"] == "collision"
    assert metrics["mean_abs_steer"] == 0.30000000000000004
    assert written == metrics
    assert (tmp_path / "metrics.json").exists()
