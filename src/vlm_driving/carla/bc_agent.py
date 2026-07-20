"""BC policy agent wrapper for CARLA rollout integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from PIL import Image

from vlm_driving.carla.observations import NormalizedAction
from vlm_driving.data import featurize_observation
from vlm_driving.training import BCCheckpoint, load_bc_checkpoint
from vlm_driving.vlm import CachedFeatureReader, QwenTokenProvider

HiddenSource = Literal["live", "cache"]


class HiddenProvider(Protocol):
    def encode_observation(self, images: Any, command_text: str) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class BCAgentOutput:
    action: NormalizedAction
    raw_action: torch.Tensor
    hidden_shape: tuple[int, ...]


class BCAgent:
    """Runs a T-013 BC checkpoint on live or cached frozen-VLM hidden states."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        hidden_source: HiddenSource = "live",
        feature_cache_dir: str | Path | None = None,
        command_text: str = "You are driving in CARLA. Keep lane and continue safely.",
        device: str | torch.device | None = None,
        hidden_provider: HiddenProvider | None = None,
        feature_reader: CachedFeatureReader | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint: BCCheckpoint = load_bc_checkpoint(checkpoint_path, map_location=self.device)
        self.resampler = self.checkpoint.resampler
        self.policy = self.checkpoint.policy
        self.resampler.eval()
        self.policy.eval()
        self.hidden_source = hidden_source
        self.command_text = command_text
        self.previous_action = (0.0, 0.0)

        self.hidden_provider = hidden_provider
        self.feature_reader = feature_reader
        if hidden_source == "cache":
            if self.feature_reader is None:
                if feature_cache_dir is None:
                    raise ValueError("feature_cache_dir or feature_reader is required for cache hidden_source")
                self.feature_reader = CachedFeatureReader(
                    feature_cache_dir,
                    expected_model_id=self.checkpoint.config.vlm.model_id,
                    expected_hidden_size=self.checkpoint.config.vlm.hidden_size,
                )
        elif hidden_source == "live":
            if self.hidden_provider is None:
                self.hidden_provider = QwenTokenProvider.from_pretrained_with_fallback(
                    model_id=self.checkpoint.config.vlm.model_id,
                    device=self.device,
                    command_text=command_text,
                )
        else:
            raise ValueError("hidden_source must be 'live' or 'cache'")

    def act(self, record: dict[str, Any], image: str | Path | Image.Image | None = None) -> NormalizedAction:
        return self.act_with_debug(record, image=image).action

    def act_with_debug(
        self,
        record: dict[str, Any],
        image: str | Path | Image.Image | None = None,
    ) -> BCAgentOutput:
        hidden = self._hidden_for_record(record, image=image)
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)
        if hidden.ndim != 2:
            raise ValueError(f"hidden state must have shape [S, H], got {tuple(hidden.shape)}")

        observation = featurize_observation(
            record,
            previous_action=self.previous_action,
            obs_dim=self.checkpoint.config.policy.obs_dim,
        ).to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            compact = self.resampler(hidden.unsqueeze(0).to(device=self.device, dtype=torch.float32))
            token_age_s = torch.zeros(1, dtype=torch.float32, device=self.device)
            output = self.policy(observation.unsqueeze(0), compact, token_age_s)
            raw_action = output["il_action"].squeeze(0).detach().cpu()

        if raw_action.numel() != 2:
            raise ValueError(f"BC policy action must have 2 values, got {raw_action.numel()}")
        action = NormalizedAction(
            steer=float(raw_action[0].item()),
            acceleration=float(raw_action[1].item()),
        ).clipped()
        self.previous_action = (action.steer, action.acceleration)
        return BCAgentOutput(action=action, raw_action=raw_action, hidden_shape=tuple(hidden.shape))

    def _hidden_for_record(self, record: dict[str, Any], image: str | Path | Image.Image | None) -> torch.Tensor:
        if self.hidden_source == "cache":
            assert self.feature_reader is not None
            frame_key = _frame_key(record)
            return self.feature_reader.read(frame_key)

        assert self.hidden_provider is not None
        live_image = image if image is not None else _camera_path(record)
        if live_image is None:
            raise ValueError("live hidden_source requires an image argument or record['camera']['path']")
        return self.hidden_provider.encode_observation(live_image, self.command_text)


def _frame_key(record: dict[str, Any]) -> str:
    camera = record.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("record is missing camera metadata")
    path = camera.get("path")
    if not path:
        raise ValueError("cache hidden_source requires record['camera']['path']")
    return Path(str(path)).name


def _camera_path(record: dict[str, Any]) -> str | None:
    camera = record.get("camera")
    if not isinstance(camera, dict):
        return None
    path = camera.get("path")
    return str(path) if path else None


__all__ = ["BCAgent", "BCAgentOutput", "HiddenProvider", "HiddenSource"]
