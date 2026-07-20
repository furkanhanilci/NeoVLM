"""Episode-level dataset discovery and train/validation splitting."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[Path, ...]
    val: tuple[Path, ...]

    def all(self) -> tuple[Path, ...]:
        return self.train + self.val


def discover_episodes(
    root: str | Path,
    manifest_name: str = "manifest.json",
    metadata_name: str = "metadata.jsonl",
) -> list[Path]:
    dataset_root = Path(root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if (dataset_root / manifest_name).exists() and (dataset_root / metadata_name).exists():
        return [dataset_root]

    episodes = {path.parent for path in dataset_root.rglob(manifest_name) if (path.parent / metadata_name).exists()}
    return sorted(episodes, key=lambda path: path.as_posix())


def split_episodes(episodes: Sequence[str | Path], val_ratio: float = 0.15, seed: int = 0) -> EpisodeSplit:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")
    unique = sorted({Path(path) for path in episodes}, key=lambda path: path.as_posix())
    if not unique:
        raise ValueError("split_episodes requires at least one episode")

    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1 or val_ratio == 0.0:
        val_count = 0
    else:
        val_count = max(1, round(len(shuffled) * val_ratio))
        val_count = min(val_count, len(shuffled) - 1)

    val = sorted(shuffled[:val_count], key=lambda path: path.as_posix())
    train = sorted(shuffled[val_count:], key=lambda path: path.as_posix())
    return EpisodeSplit(train=tuple(train), val=tuple(val))


def load_split_manifest(path: str | Path, dataset_root: str | Path | None = None) -> EpisodeSplit:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "episode_split_v1":
        raise ValueError(f"unsupported split manifest schema: {data.get('schema_version')}")
    root = Path(dataset_root) if dataset_root is not None else manifest_path.parent
    train: list[Path] = []
    val: list[Path] = []
    for entry in data.get("episodes", []):
        episode_dir = Path(str(entry["episode_dir"]))
        if not episode_dir.is_absolute():
            episode_dir = root / episode_dir
        split_name = str(entry["split"])
        if split_name == "train":
            train.append(episode_dir)
        elif split_name == "val":
            val.append(episode_dir)
        else:
            raise ValueError(f"unsupported split label: {split_name}")
    return EpisodeSplit(
        train=tuple(sorted(train, key=lambda item: item.as_posix())),
        val=tuple(sorted(val, key=lambda item: item.as_posix())),
    )


def write_split_manifest(
    output_path: str | Path,
    split: EpisodeSplit,
    dataset_root: str | Path | None = None,
    val_ratio: float | None = None,
    seed: int | None = None,
) -> Path:
    path = Path(output_path)
    root = Path(dataset_root) if dataset_root is not None else None
    records = []
    for split_name, episodes in (("train", split.train), ("val", split.val)):
        for episode in episodes:
            records.append({"episode_dir": _display_path(episode, root), "split": split_name})
    data = {
        "schema_version": "episode_split_v1",
        "num_episodes": len(records),
        "num_train": len(split.train),
        "num_val": len(split.val),
        "val_ratio": val_ratio,
        "seed": seed,
        "episodes": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _display_path(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return path.as_posix()


__all__ = ["EpisodeSplit", "discover_episodes", "load_split_manifest", "split_episodes", "write_split_manifest"]
