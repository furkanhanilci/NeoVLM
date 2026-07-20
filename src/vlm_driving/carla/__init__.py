"""CARLA integration utilities for closed-loop smoke tests.

The heavy CARLA PythonAPI dependency is imported lazily so schema modules can be
unit-tested in non-CARLA environments.
"""

from __future__ import annotations

from typing import Any

__all__ = ["RolloutConfig", "run_rollout"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from vlm_driving.carla.rollout import RolloutConfig, run_rollout

        return {"RolloutConfig": RolloutConfig, "run_rollout": run_rollout}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
