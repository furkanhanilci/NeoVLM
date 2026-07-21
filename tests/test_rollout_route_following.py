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


class _Vector:
    def __init__(self, x: float = 1.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _Actor:
    def __init__(self, actor_id: int | None, type_id: str) -> None:
        if actor_id is not None:
            self.id = actor_id
        self.type_id = type_id


class _CollisionEvent:
    def __init__(self, frame: int, actor: _Actor | None, impulse: _Vector | None = None) -> None:
        self.frame = frame
        self.other_actor = actor
        self.normal_impulse = impulse if impulse is not None else _Vector()


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


def test_collision_contact_episode_dedup_counts_continuous_actor_once(monkeypatch):
    rollout, _basic_agent = _load_rollout_with_fake_carla(monkeypatch)
    dedup_state = rollout._CollisionDedupState(cooldown_frames=20)
    actor = _Actor(actor_id=7, type_id="static.car")
    collision_events = []
    processed_count = 0
    counted_events = 0

    for frame in range(100):
        collision_events.append(_CollisionEvent(frame=frame, actor=actor))
        event_state, processed_count = rollout._collision_event(
            collision_events,
            processed_count,
            dedup_state=dedup_state,
            current_frame=frame,
        )
        counted_events += event_state.collision_event_count

    assert processed_count == 100
    assert counted_events == 1
    assert event_state.collision is True
    assert event_state.collision_new is False
    assert event_state.collision_event_count == 0
    assert event_state.collision_actor_type == "static.car"


def test_collision_contact_episode_dedup_counts_different_actors(monkeypatch):
    rollout, _basic_agent = _load_rollout_with_fake_carla(monkeypatch)
    dedup_state = rollout._CollisionDedupState(cooldown_frames=20)
    collision_events = [
        _CollisionEvent(frame=5, actor=_Actor(actor_id=7, type_id="static.car")),
        _CollisionEvent(frame=5, actor=_Actor(actor_id=8, type_id="static.trafficsign")),
    ]

    event_state, processed_count = rollout._collision_event(
        collision_events,
        processed_count=0,
        dedup_state=dedup_state,
        current_frame=5,
    )

    assert processed_count == 2
    assert event_state.collision is True
    assert event_state.collision_new is True
    assert event_state.collision_event_count == 2
    assert event_state.collision_actor_type == "static.trafficsign"


def test_collision_contact_episode_dedup_counts_same_actor_after_cooldown_gap(monkeypatch):
    rollout, _basic_agent = _load_rollout_with_fake_carla(monkeypatch)
    dedup_state = rollout._CollisionDedupState(cooldown_frames=20)
    actor = _Actor(actor_id=7, type_id="static.car")
    collision_events = []
    processed_count = 0
    counts = []

    for frame in (0, 10, 31):
        collision_events.append(_CollisionEvent(frame=frame, actor=actor))
        event_state, processed_count = rollout._collision_event(
            collision_events,
            processed_count,
            dedup_state=dedup_state,
            current_frame=frame,
        )
        counts.append(event_state.collision_event_count)

    assert processed_count == 3
    assert counts == [1, 0, 1]


def test_collision_contact_episode_dedup_falls_back_to_actor_type_without_id(monkeypatch):
    rollout, _basic_agent = _load_rollout_with_fake_carla(monkeypatch)
    dedup_state = rollout._CollisionDedupState(cooldown_frames=20)
    collision_events = [
        _CollisionEvent(frame=0, actor=_Actor(actor_id=None, type_id="static.prop")),
        _CollisionEvent(frame=1, actor=_Actor(actor_id=None, type_id="static.prop")),
    ]

    event_state, processed_count = rollout._collision_event(
        collision_events,
        processed_count=0,
        dedup_state=dedup_state,
        current_frame=1,
    )

    assert processed_count == 2
    assert event_state.collision is True
    assert event_state.collision_new is True
    assert event_state.collision_event_count == 1
    assert event_state.collision_actor_type == "static.prop"
