import pytest

from vlm_driving.carla.route_progress import point_xyz, project_route_progress, route_length_m


class _Location:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _Transform:
    def __init__(self, location: _Location) -> None:
        self.location = location


class _Waypoint:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.transform = _Transform(_Location(x, y, z))


def test_route_length_accepts_tuples_dicts_and_carla_like_objects():
    points = [(0.0, 0.0), {"x": 3.0, "y": 4.0, "z": 0.0}, _Location(6.0, 8.0)]

    assert point_xyz(_Waypoint(1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
    assert route_length_m(points) == pytest.approx(10.0)


def test_project_route_progress_on_l_shaped_route():
    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]

    progress = project_route_progress(route, (5.0, 2.0))

    assert progress.route_length_m == pytest.approx(20.0)
    assert progress.route_progress_m == pytest.approx(5.0)
    assert progress.distance_to_goal_m == pytest.approx(15.0)
    assert progress.closest_segment_index == 0
    assert progress.lateral_distance_m == pytest.approx(2.0)


def test_project_route_progress_clamps_to_nearest_segment_projection():
    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]

    progress = project_route_progress(route, (12.0, 8.0))

    assert progress.route_progress_m == pytest.approx(18.0)
    assert progress.distance_to_goal_m == pytest.approx(2.0)
    assert progress.closest_segment_index == 1
    assert progress.lateral_distance_m == pytest.approx(2.0)


def test_project_route_progress_handles_degenerate_routes():
    empty = project_route_progress([], (1.0, 2.0))
    single = project_route_progress([(4.0, 6.0)], (1.0, 2.0))

    assert empty.route_length_m == 0.0
    assert empty.distance_to_goal_m == 0.0
    assert single.route_progress_m == 0.0
    assert single.distance_to_goal_m == pytest.approx(5.0)
