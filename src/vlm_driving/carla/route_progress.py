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


class RouteProgressTracker:
    """Stateful forward-only route progress tracker.

    The stateless projection helper can still be used for one-off geometry
    queries. Rollouts should use this tracker so loops or nearby later route
    segments cannot make progress jump forward from a globally-nearest match.
    """

    def __init__(
        self,
        points: Sequence[Any],
        lookahead_segments: int = 25,
        max_progress_step_m: float = 25.0,
    ) -> None:
        self._points = list(points)
        self._last_segment_index = 0
        self._last_progress_m = 0.0
        self.lookahead_segments = lookahead_segments
        self.max_progress_step_m = max_progress_step_m

    @property
    def last_segment_index(self) -> int:
        return self._last_segment_index

    @property
    def last_progress_m(self) -> float:
        return self._last_progress_m

    @property
    def route_length_m(self) -> float:
        return route_length_m(self._points)

    def update(self, location: Any) -> RouteProgress:
        progress = project_route_progress(
            self._points,
            location,
            start_segment_index=self._last_segment_index,
            min_progress_m=self._last_progress_m,
            lookahead_segments=self.lookahead_segments,
            max_progress_m=self._last_progress_m + self.max_progress_step_m,
        )
        if progress.closest_segment_index is not None:
            self._last_segment_index = max(self._last_segment_index, progress.closest_segment_index)
        self._last_progress_m = max(self._last_progress_m, progress.route_progress_m)
        return progress


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


def project_route_progress(
    points: Sequence[Any],
    location: Any,
    start_segment_index: int = 0,
    min_progress_m: float = 0.0,
    lookahead_segments: int | None = None,
    max_progress_m: float | None = None,
) -> RouteProgress:
    """Project ``location`` onto a route polyline and return progress metrics.

    ``start_segment_index`` and ``min_progress_m`` make the projection suitable
    for forward-only tracking: segments before the last match are not searched
    and returned progress is clamped so it never decreases.
    """

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

    best_progress = _clip(float(min_progress_m), 0.0, route_length)
    best_segment_index: int | None = None
    best_lateral = math.inf
    segment_count = len(xyz) - 1
    first_segment = min(max(int(start_segment_index), 0), segment_count - 1)
    last_segment = segment_count
    if lookahead_segments is not None:
        last_segment = min(segment_count, first_segment + max(1, int(lookahead_segments)))
    cumulative_lengths = _cumulative_lengths(xyz)
    for segment_index in range(first_segment, last_segment):
        start = xyz[segment_index]
        end = xyz[segment_index + 1]
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
        raw_progress = cumulative_lengths[segment_index] + t * segment_length
        if max_progress_m is not None and raw_progress > max_progress_m and segment_index > first_segment:
            continue
        bounded_progress = _bounded_progress(raw_progress, min_progress_m, max_progress_m, route_length)
        if best_segment_index is None or lateral < best_lateral or (
            lateral == best_lateral and bounded_progress < best_progress
        ):
            best_lateral = lateral
            best_progress = bounded_progress
            best_segment_index = segment_index

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


def _cumulative_lengths(points: Sequence[tuple[float, float, float]]) -> list[float]:
    cumulative = [0.0]
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += _distance(start, end)
        cumulative.append(total)
    return cumulative


def _bounded_progress(
    raw_progress: float,
    min_progress_m: float,
    max_progress_m: float | None,
    route_length: float,
) -> float:
    progress = max(float(min_progress_m), raw_progress)
    if max_progress_m is not None:
        progress = min(float(max_progress_m), progress)
    return _clip(progress, 0.0, route_length)


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


__all__ = ["RouteProgress", "RouteProgressTracker", "point_xyz", "project_route_progress", "route_length_m"]
