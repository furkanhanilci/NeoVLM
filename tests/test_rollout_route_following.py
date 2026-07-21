import importlib
import sys
import types

import pytest


class _Location:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _VehicleControl:
    def __init__(
        self,
        throttle: float = 0.0,
        steer: float = 0.0,
        brake: float = 0.0,
        hand_brake: bool = False,
        reverse: bool = False,
    ) -> None:
        self.throttle = throttle
        self.steer = steer
        self.brake = brake
        self.hand_brake = hand_brake
        self.reverse = reverse


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


class _BasicAgent:
    instances: list["_BasicAgent"] = []

    def __init__(
        self,
        vehicle: _Vehicle,
        target_speed: float = 20.0,
        opt_dict=None,
        map_inst=None,
        grp_inst=None,
    ) -> None:
        self.vehicle = vehicle
        self.target_speed = target_speed
        self.destinations: list[tuple[_Location, _Location | None]] = []
        self.run_step_calls = 0
        self.__class__.instances.append(self)

    def set_destination(self, end_location: _Location, start_location: _Location | None = None) -> None:
        self.destinations.append((end_location, start_location))

    def run_step(self) -> _VehicleControl:
        self.run_step_calls += 1
        return _VehicleControl(throttle=0.4, steer=0.1, brake=0.0, hand_brake=False, reverse=False)


def _load_rollout_with_fake_carla(monkeypatch):
    sys.modules.pop("vlm_driving.carla.rollout", None)
    sys.modules.pop("vlm_driving.carla.actions", None)
    _BasicAgent.instances = []
    fake_carla = types.SimpleNamespace(VehicleControl=_VehicleControl)
    agents_module = types.ModuleType("agents")
    navigation_module = types.ModuleType("agents.navigation")
    basic_agent_module = types.ModuleType("agents.navigation.basic_agent")
    basic_agent_module.BasicAgent = _BasicAgent
    monkeypatch.setitem(sys.modules, "carla", fake_carla)
    monkeypatch.setitem(sys.modules, "agents", agents_module)
    monkeypatch.setitem(sys.modules, "agents.navigation", navigation_module)
    monkeypatch.setitem(sys.modules, "agents.navigation.basic_agent", basic_agent_module)
    return importlib.import_module("vlm_driving.carla.rollout"), _BasicAgent


def test_eval_mode_autopilot_uses_basic_agent_not_tm_set_path(monkeypatch):
    rollout, basic_agent = _load_rollout_with_fake_carla(monkeypatch)
    vehicle = _Vehicle()
    traffic_manager = _TrafficManager()
    destination = _Location(10.0, 0.0)

    follower = rollout._configure_autopilot(
        vehicle=vehicle,
        traffic_manager=traffic_manager,
        destination=destination,
        config=rollout.RolloutConfig(terminate_on_collision=False),
    )

    assert isinstance(follower, rollout._BasicAgentRouteFollower)
    assert vehicle.autopilot_calls == [(False, 8000)]
    assert len(basic_agent.instances) == 1
    assert basic_agent.instances[0].target_speed == pytest.approx(
        rollout.RolloutConfig().target_speed_mps * 3.6
    )
    assert basic_agent.instances[0].destinations == [(destination, None)]
    assert traffic_manager.set_path_calls == []
    assert traffic_manager.speed_calls == []

    action, control = follower.act(vehicle)

    assert basic_agent.instances[0].run_step_calls == 1
    assert action.steer == pytest.approx(0.1)
    assert action.acceleration == pytest.approx(0.4)
    assert control.throttle == pytest.approx(0.4)
    assert control.steer == pytest.approx(0.1)


def test_data_collection_autopilot_path_keeps_tm_autopilot_on(monkeypatch):
    rollout, basic_agent = _load_rollout_with_fake_carla(monkeypatch)
    vehicle = _Vehicle()
    traffic_manager = _TrafficManager()

    follower = rollout._configure_autopilot(
        vehicle=vehicle,
        traffic_manager=traffic_manager,
        destination=_Location(10.0, 0.0),
        config=rollout.RolloutConfig(),
    )

    assert follower is None
    assert rollout.RolloutConfig().terminate_on_collision is True
    assert vehicle.autopilot_calls == [(True, 8000)]
    assert traffic_manager.speed_calls == [(vehicle, 35.0)]
    assert traffic_manager.set_path_calls == []
    assert basic_agent.instances == []
