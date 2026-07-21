"""Data loading utilities for imitation learning."""

from importlib import import_module

from vlm_driving.data.splits import EpisodeSplit, discover_episodes, load_split_manifest, split_episodes, write_split_manifest

_DATASET_STATS_EXPORTS = {
    "format_dataset_summary",
    "summarize_dataset",
    "write_dataset_stats",
}
_IL_DATASET_EXPORTS = {
    "ILDataset",
    "ILDatasetRecord",
    "OBSERVATION_FEATURES",
    "ROUTE_COMMANDS",
    "featurize_observation",
}


def __getattr__(name: str):
    if name in _DATASET_STATS_EXPORTS:
        dataset_stats = import_module("vlm_driving.data.dataset_stats")
        value = getattr(dataset_stats, name)
        globals()[name] = value
        return value
    if name in _IL_DATASET_EXPORTS:
        il_dataset = import_module("vlm_driving.data.il_dataset")
        value = getattr(il_dataset, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EpisodeSplit",
    "ILDataset",
    "ILDatasetRecord",
    "OBSERVATION_FEATURES",
    "ROUTE_COMMANDS",
    "discover_episodes",
    "featurize_observation",
    "format_dataset_summary",
    "load_split_manifest",
    "split_episodes",
    "summarize_dataset",
    "write_split_manifest",
    "write_dataset_stats",
]
