"""Offline frozen VLM hidden-state feature cache."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from vlm_driving.vlm.qwen_provider import QwenTokenProvider

MANIFEST_NAME = "cache_manifest.json"


@dataclass(frozen=True)
class FeatureCacheRecord:
    frame_key: str
    source_frame: str
    cache_file: str
    shape: list[int]
    dtype: str


@dataclass(frozen=True)
class FeatureCacheManifest:
    schema_version: str
    model_id: str
    hidden_size: int
    precision: str
    command_text: str
    num_frames: int
    records: list[FeatureCacheRecord]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["records"] = [asdict(record) for record in self.records]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureCacheManifest":
        return cls(
            schema_version=str(data["schema_version"]),
            model_id=str(data["model_id"]),
            hidden_size=int(data["hidden_size"]),
            precision=str(data["precision"]),
            command_text=str(data["command_text"]),
            num_frames=int(data["num_frames"]),
            records=[FeatureCacheRecord(**record) for record in data["records"]],
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "FeatureCacheManifest":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_feature_cache(
    frames_dir: str | Path,
    provider: QwenTokenProvider,
    out_dir: str | Path,
    command_text: str,
    max_frames: int | None = None,
) -> FeatureCacheManifest:
    frames_path = Path(frames_dir)
    output_path = Path(out_dir)
    if not frames_path.exists():
        raise FileNotFoundError(f"frames_dir does not exist: {frames_path}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    frame_paths = sorted(path for path in frames_path.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    total_frames = len(frame_paths)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        raise ValueError(f"no image frames found in {frames_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    records: list[FeatureCacheRecord] = []
    approx_bytes_per_frame = None

    for index, frame_path in enumerate(frame_paths):
        hidden = provider.encode_observation(frame_path, command_text)
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise RuntimeError(f"expected hidden shape [1, S, H], got {tuple(hidden.shape)}")
        hidden_2d = hidden.squeeze(0).detach().cpu().contiguous()
        if hidden_2d.shape[-1] != provider.hidden_size:
            raise RuntimeError(
                f"hidden size mismatch for {frame_path}: got {hidden_2d.shape[-1]}, expected {provider.hidden_size}"
            )
        approx_bytes_per_frame = hidden_2d.numel() * hidden_2d.element_size()
        cache_file = f"frame_{index:05d}.pt"
        torch.save(hidden_2d, output_path / cache_file)
        records.append(
            FeatureCacheRecord(
                frame_key=frame_path.name,
                source_frame=str(frame_path),
                cache_file=cache_file,
                shape=list(hidden_2d.shape),
                dtype=str(hidden_2d.dtype),
            )
        )

    manifest = FeatureCacheManifest(
        schema_version="vlm_hidden_cache_v1",
        model_id=provider.model_id,
        hidden_size=provider.hidden_size,
        precision=provider.load_info.precision,
        command_text=command_text,
        num_frames=len(records),
        records=records,
    )
    manifest.write(output_path / MANIFEST_NAME)

    if max_frames is not None and total_frames > len(frame_paths):
        print(f"feature cache smoke limited frames: cached {len(frame_paths)} of {total_frames} frames")
    if approx_bytes_per_frame is not None:
        scale_gb_100k = approx_bytes_per_frame * 100_000 / (1024 ** 3)
        print(
            "feature cache scale warning: "
            f"approx {approx_bytes_per_frame / (1024 ** 2):.3f} MiB/frame; "
            f"100k frames ~= {scale_gb_100k:.1f} GiB"
        )
    return manifest


class CachedFeatureReader:
    def __init__(
        self,
        cache_dir: str | Path,
        expected_model_id: str | None = None,
        expected_hidden_size: int | None = None,
        max_cached_tensors: int = 128,
    ) -> None:
        if max_cached_tensors < 0:
            raise ValueError("max_cached_tensors must be non-negative")
        self.cache_dir = Path(cache_dir)
        self.manifest = FeatureCacheManifest.read(self.cache_dir / MANIFEST_NAME)
        if expected_model_id is not None and self.manifest.model_id != expected_model_id:
            raise ValueError(
                f"cache model_id mismatch: got {self.manifest.model_id}, expected {expected_model_id}"
            )
        if expected_hidden_size is not None and self.manifest.hidden_size != expected_hidden_size:
            raise ValueError(
                f"cache hidden_size mismatch: got {self.manifest.hidden_size}, expected {expected_hidden_size}"
            )
        self._records = {record.frame_key: record for record in self.manifest.records}
        self.max_cached_tensors = max_cached_tensors
        self._tensor_cache: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    def keys(self) -> list[str]:
        return [record.frame_key for record in self.manifest.records]

    def read(self, frame_key: str, map_location: str | torch.device = "cpu") -> torch.Tensor:
        try:
            record = self._records[frame_key]
        except KeyError as exc:
            raise KeyError(f"frame_key not found in cache: {frame_key}") from exc

        cache_key = (frame_key, str(map_location))
        if self.max_cached_tensors > 0 and cache_key in self._tensor_cache:
            self._cache_hits += 1
            tensor = self._tensor_cache.pop(cache_key)
            self._tensor_cache[cache_key] = tensor
            return tensor

        self._cache_misses += 1
        tensor = torch.load(self.cache_dir / record.cache_file, map_location=map_location, weights_only=True)
        if list(tensor.shape) != record.shape:
            raise ValueError(f"cached tensor shape mismatch for {frame_key}: {list(tensor.shape)} != {record.shape}")
        if str(tensor.dtype) != record.dtype:
            raise ValueError(f"cached tensor dtype mismatch for {frame_key}: {tensor.dtype} != {record.dtype}")
        if tensor.shape[-1] != self.manifest.hidden_size:
            raise ValueError(
                f"cached tensor hidden size mismatch for {frame_key}: {tensor.shape[-1]} != {self.manifest.hidden_size}"
            )
        if self.max_cached_tensors > 0:
            self._tensor_cache[cache_key] = tensor
            if len(self._tensor_cache) > self.max_cached_tensors:
                self._tensor_cache.popitem(last=False)
                self._cache_evictions += 1
        return tensor

    def cache_info(self) -> dict[str, Any]:
        return {
            "max_cached_tensors": self.max_cached_tensors,
            "size": len(self._tensor_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "evictions": self._cache_evictions,
            "keys": [
                {"frame_key": frame_key, "map_location": map_location}
                for frame_key, map_location in self._tensor_cache.keys()
            ],
        }


def frame_paths_from_episode(episode_dir: str | Path) -> Path:
    episode_path = Path(episode_dir)
    frames_dir = episode_path / "frames"
    if frames_dir.exists():
        return frames_dir
    return episode_path
