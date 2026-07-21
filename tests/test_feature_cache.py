from pathlib import Path
from unittest import mock

import pytest
import torch

from vlm_driving.config import ExperimentConfig
from vlm_driving.models import QueryResampler
from vlm_driving.vlm import (
    CachedFeatureReader,
    FeatureCacheManifest,
    FeatureCacheRecord,
    QwenTokenProvider,
    build_feature_cache,
)


def test_cached_feature_reader_roundtrip_with_synthetic_cpu_files(tmp_path: Path):
    features = torch.randn(5, 7, dtype=torch.bfloat16)
    torch.save(features, tmp_path / "frame_00000.pt")
    manifest = FeatureCacheManifest(
        schema_version="vlm_hidden_cache_v1",
        model_id="synthetic/model",
        hidden_size=7,
        precision="bf16",
        command_text="keep lane",
        num_frames=1,
        records=[
            FeatureCacheRecord(
                frame_key="frame_00000.png",
                source_frame="frames/frame_00000.png",
                cache_file="frame_00000.pt",
                shape=[5, 7],
                dtype=str(features.dtype),
            )
        ],
    )
    manifest.write(tmp_path / "cache_manifest.json")

    reader = CachedFeatureReader(
        tmp_path,
        expected_model_id="synthetic/model",
        expected_hidden_size=7,
    )

    assert reader.keys() == ["frame_00000.png"]
    loaded = reader.read("frame_00000.png")
    assert loaded.dtype == torch.bfloat16
    assert torch.equal(loaded, features)
    with pytest.raises(KeyError, match="frame_key"):
        reader.read("missing.png")
    with pytest.raises(ValueError, match="model_id mismatch"):
        CachedFeatureReader(tmp_path, expected_model_id="other/model")
    with pytest.raises(ValueError, match="hidden_size mismatch"):
        CachedFeatureReader(tmp_path, expected_hidden_size=9)


def _write_synthetic_cache(cache_dir: Path, values: list[float]) -> None:
    records = []
    for index, value in enumerate(values):
        tensor = torch.full((2, 3), value, dtype=torch.float32)
        cache_file = f"frame_{index:05d}.pt"
        torch.save(tensor, cache_dir / cache_file)
        records.append(
            FeatureCacheRecord(
                frame_key=f"frame_{index:05d}.png",
                source_frame=f"frames/frame_{index:05d}.png",
                cache_file=cache_file,
                shape=[2, 3],
                dtype=str(tensor.dtype),
            )
        )
    FeatureCacheManifest(
        schema_version="vlm_hidden_cache_v1",
        model_id="synthetic/model",
        hidden_size=3,
        precision="fp32",
        command_text="keep lane",
        num_frames=len(records),
        records=records,
    ).write(cache_dir / "cache_manifest.json")


def test_cached_feature_reader_lru_hits_and_evicts_least_recently_used(tmp_path: Path):
    _write_synthetic_cache(tmp_path, [0.0, 1.0, 2.0])
    reader = CachedFeatureReader(tmp_path, max_cached_tensors=2)

    with mock.patch("torch.load", wraps=torch.load) as wrapped_load:
        frame0 = reader.read("frame_00000.png")
        frame1 = reader.read("frame_00001.png")
        frame0_again = reader.read("frame_00000.png")
        frame2 = reader.read("frame_00002.png")

    assert frame0_again is frame0
    assert torch.all(frame1 == 1.0)
    assert torch.all(frame2 == 2.0)
    assert wrapped_load.call_count == 3
    assert reader.cache_info() == {
        "max_cached_tensors": 2,
        "size": 2,
        "hits": 1,
        "misses": 3,
        "evictions": 1,
        "keys": [
            {"frame_key": "frame_00000.png", "map_location": "cpu"},
            {"frame_key": "frame_00002.png", "map_location": "cpu"},
        ],
    }


def test_cached_feature_reader_capacity_zero_preserves_reload_behavior(tmp_path: Path):
    _write_synthetic_cache(tmp_path, [4.0])
    reader = CachedFeatureReader(tmp_path, max_cached_tensors=0)

    first = reader.read("frame_00000.png")
    second = reader.read("frame_00000.png")

    assert torch.equal(first, second)
    assert first is not second
    assert reader.cache_info() == {
        "max_cached_tensors": 0,
        "size": 0,
        "hits": 0,
        "misses": 2,
        "evictions": 0,
        "keys": [],
    }


@pytest.mark.vlm
def test_build_feature_cache_matches_live_qwen_resampler(tmp_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real Qwen3-VL feature cache build")
    config = ExperimentConfig()
    frames_dir = Path("results/datasets/carla_il_smoke/episode_000/frames")
    if not frames_dir.exists():
        frames_dir = Path("results/smoke_rollout/frames")
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        pytest.skip("smoke frames are required")
    command = "You are driving in CARLA. Keep lane and continue safely."
    provider = QwenTokenProvider.from_pretrained_with_fallback(
        model_id=config.vlm.model_id,
        device="cuda",
        command_text=command,
    )

    manifest = build_feature_cache(
        frames_dir=frames_dir,
        provider=provider,
        out_dir=tmp_path,
        command_text=command,
        max_frames=1,
    )
    reader = CachedFeatureReader(
        tmp_path,
        expected_model_id=config.vlm.model_id,
        expected_hidden_size=config.vlm.hidden_size,
    )
    record = manifest.records[0]
    cached_hidden = reader.read(record.frame_key)
    assert cached_hidden.dtype == torch.bfloat16
    assert cached_hidden.shape[-1] == config.vlm.hidden_size

    resampler = QueryResampler(
        input_dim=config.resampler.input_dim,
        output_dim=config.resampler.output_dim,
        num_queries=config.resampler.num_queries,
        num_heads=config.resampler.num_heads,
        dropout=0.0,
    ).to("cuda")
    resampler.eval()
    with torch.no_grad():
        cached_compact = resampler(cached_hidden.unsqueeze(0).to(device="cuda", dtype=torch.float32))
        live_hidden = provider.encode_observation(record.source_frame, command)
        live_compact = resampler(live_hidden.to(dtype=torch.float32))

    assert torch.allclose(cached_compact, live_compact, atol=1e-4, rtol=1e-4)
