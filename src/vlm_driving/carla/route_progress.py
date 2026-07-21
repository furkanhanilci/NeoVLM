"""Pure-Python route geometry helpers for CARLA rollout logging."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RouteProgress:
    route_length_m: float
    route_progress_m: float
    distance_to_goal_m: float
    closest_segment_index: int | None
    lateral_distance_m: float


def point_xyz(point: Any) -> tuple[float, float, float]:
    """Return an ``(x, y, z)`` tuple from CARLA-like or plain point objects."""

    point = _unwrap_point(point)
    if isinstance(point, Mapping):
        return (
            float(point.get("x", 0.0)),
            float(point.get("y", 0.0)),
            float(point.get("z", 0.0)),
        )
    if isinstance(point, Sequence) and not isinstance(point, (str, bytes, bytearray)):
        x = float(point[0]) if len(point) > 0 else 0.0
        y = float(point[1]) if len(point) > 1 else 0.0
        z = float(point[2]) if len(point) > 2 else 0.0
        return (x, y, z)
    return (
        float(getattr(point, "x")),
        float(getattr(point, "y")),
        float(getattr(point, "z", 0.0)),
    )


def route_length_m(points: Sequence[Any]) -> float:
    xyz = [point_xyz(point) for point in points]
    return sum(_distance(a, b) for a, b in zip(xyz, xyz[1:]))


def project_route_progress(points: Sequence[Any], location: Any) -> RouteProgress:
    """Project ``location`` onto a route polyline and return progress metrics."""

    xyz = [point_xyz(point) for point in points]
    route_length = route_length_m(xyz)
    position = point_xyz(location)
    if not xyz:
        return RouteProgress(
            route_length_m=0.0,
            route_progress_m=0.0,
            distance_to_goal_m=0.0,
            closest_segment_index=None,
            lateral_distance_m=0.0,
        )
    if len(xyz) == 1 or route_length <= 0.0:
        lateral = _distance(position, xyz[-1])
        return RouteProgress(
            route_length_m=0.0,
            route_progress_m=0.0,
            distance_to_goal_m=lateral,
            closest_segment_index=None,
            lateral_distance_m=lateral,
        )

    best_progress = 0.0
    best_segment_index: int | None = None
    best_lateral = math.inf
    cumulative = 0.0
    for segment_index, (start, end) in enumerate(zip(xyz, xyz[1:])):
        vector = _sub(end, start)
        segment_length_sq = _dot(vector, vector)
        segment_length = math.sqrt(segment_length_sq)
        if segment_length <= 0.0:
            continue
        t = _clip(_dot(_sub(position, start), vector) / segment_length_sq, 0.0, 1.0)
        projection = (
            start[0] + t * vector[0],
            start[1] + t * vector[1],
            start[2] + t * vector[2],
        )
        lateral = _distance(position, projection)
        progress = cumulative + t * segment_length
        if lateral < best_lateral:
            best_lateral = lateral
            best_progress = progress
            best_segment_index = segment_index
        cumulative += segment_length

    progress_clipped = _clip(best_progress, 0.0, route_length)
    return RouteProgress(
        route_length_m=route_length,
        route_progress_m=progress_clipped,
        distance_to_goal_m=max(0.0, route_length - progress_clipped),
        closest_segment_index=best_segment_index,
        lateral_distance_m=0.0 if math.isinf(best_lateral) else best_lateral,
    )


def _unwrap_point(point: Any) -> Any:
    transform = getattr(point, "transform", None)
    if transform is not None:
        return getattr(transform, "location", point)
    location = getattr(point, "location", None)
    return location if location is not None else point


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


__all__ = ["RouteProgress", "point_xyz", "project_route_progress", "route_length_m"]
