import importlib
import sys
import types


class _Location:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _Vehicle:
    def __init__(self) -> None:
        self.autopilot_calls: list[tuple[bool, int]] = []

    def set_autopilot(self, enabled: bool, port: int) -> None:
        self.autopilot_calls.append((enabled, port))


class _TrafficManager:
    def __init__(self) -> None:
        self.speed_calls: list[tuple[_Vehicle, float]] = []
        self.set_path_calls: list[tuple[_Vehicle, list[_Location]]] = []

    def get_port(self) -> int:
        return 8000

    def vehicle_percentage_speed_difference(self, vehicle: _Vehicle, percentage: float) -> None:
        self.speed_calls.append((vehicle, percentage))

    def set_path(self, vehicle: _Vehicle, path: list[_Location]) -> None:
        self.set_path_calls.append((vehicle, path))


def _load_rollout_with_fake_carla(monkeypatch):
    sys.modules.pop("vlm_driving.carla.rollout", None)
    sys.modules.pop("vlm_driving.carla.actions", None)
    fake_carla = types.SimpleNamespace(VehicleControl=object)
    monkeypatch.setitem(sys.modules, "carla", fake_carla)
    return importlib.import_module("vlm_driving.carla.rollout")


def test_eval_mode_autopilot_uses_manual_follower_not_tm_set_path(monkeypatch):
    rollout = _load_rollout_with_fake_carla(monkeypatch)
    vehicle = _Vehicle()
    traffic_manager = _TrafficManager()

    follower = rollout._configure_autopilot(
        vehicle=vehicle,
        traffic_manager=traffic_manager,
        route_points=[_Location(0.0, 0.0), _Location(10.0, 0.0)],
        config=rollout.RolloutConfig(terminate_on_collision=False),
    )

    assert isinstance(follower, rollout._SequentialRouteFollower)
    assert vehicle.autopilot_calls == [(False, 8000)]
    assert traffic_manager.set_path_calls == []
    assert traffic_manager.speed_calls == []


def test_data_collection_autopilot_path_keeps_tm_autopilot_on(monkeypatch):
    rollout = _load_rollout_with_fake_carla(monkeypatch)
    vehicle = _Vehicle()
    traffic_manager = _TrafficManager()

    follower = rollout._configure_autopilot(
        vehicle=vehicle,
        traffic_manager=traffic_manager,
        route_points=[_Location(0.0, 0.0), _Location(10.0, 0.0)],
        config=rollout.RolloutConfig(),
    )

    assert follower is None
    assert rollout.RolloutConfig().terminate_on_collision is True
    assert vehicle.autopilot_calls == [(True, 8000)]
    assert traffic_manager.speed_calls == [(vehicle, 35.0)]
    assert traffic_manager.set_path_calls == []
