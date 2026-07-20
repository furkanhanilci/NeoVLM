import json
from pathlib import Path

import pytest

from vlm_driving.data import discover_episodes, load_split_manifest, split_episodes, write_split_manifest


def _episode(root: Path, name: str) -> Path:
    episode_dir = root / name
    episode_dir.mkdir(parents=True)
    (episode_dir / "manifest.json").write_text('{"schema_version":"carla_rollout_v1"}\n', encoding="utf-8")
    (episode_dir / "metadata.jsonl").write_text('{"episode_id":"' + name + '"}\n', encoding="utf-8")
    return episode_dir


def test_discover_episodes_returns_sorted_valid_episode_dirs(tmp_path: Path):
    root = tmp_path / "dataset"
    episode_b = _episode(root, "episode_b")
    episode_a = _episode(root, "episode_a")
    invalid = root / "episode_invalid"
    invalid.mkdir(parents=True)
    (invalid / "manifest.json").write_text("{}\n", encoding="utf-8")

    assert discover_episodes(root) == [episode_a, episode_b]
    assert discover_episodes(episode_a) == [episode_a]


def test_split_episodes_is_episode_level_deterministic_and_non_overlapping(tmp_path: Path):
    episodes = [_episode(tmp_path, f"episode_{idx:03d}") for idx in range(6)]

    first = split_episodes(episodes, val_ratio=0.34, seed=17)
    second = split_episodes(reversed(episodes), val_ratio=0.34, seed=17)

    assert first == second
    assert len(first.val) == 2
    assert len(first.train) == 4
    assert set(first.train).isdisjoint(first.val)
    assert set(first.all()) == set(episodes)


def test_split_episodes_handles_single_episode_and_validates_inputs(tmp_path: Path):
    episode = _episode(tmp_path, "episode_000")

    split = split_episodes([episode], val_ratio=0.15, seed=0)

    assert split.train == (episode,)
    assert split.val == ()
    with pytest.raises(ValueError, match="at least one episode"):
        split_episodes([])
    with pytest.raises(ValueError, match="val_ratio"):
        split_episodes([episode], val_ratio=1.0)


def test_write_split_manifest_records_episode_level_labels(tmp_path: Path):
    root = tmp_path / "dataset"
    episodes = [_episode(root, f"episode_{idx:03d}") for idx in range(4)]
    split = split_episodes(episodes, val_ratio=0.25, seed=4)

    manifest_path = write_split_manifest(root / "split_manifest.json", split, dataset_root=root, val_ratio=0.25, seed=4)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["schema_version"] == "episode_split_v1"
    assert data["num_episodes"] == 4
    assert data["num_train"] == 3
    assert data["num_val"] == 1
    assert {entry["split"] for entry in data["episodes"]} == {"train", "val"}
    assert all(entry["episode_dir"].startswith("episode_") for entry in data["episodes"])
    loaded = load_split_manifest(manifest_path)
    assert loaded == split
