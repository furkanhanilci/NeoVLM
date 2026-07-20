"""Stable observation and action schemas for CARLA driving rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

RouteCommand = Literal[
    "lane_follow",
    "turn_left",
    "turn_right",
    "go_straight",
    "change_lane_left",
    "change_lane_right",
    "stop",
]

TerminationReason = Literal[
    "running",
    "max_steps",
    "collision",
    "off_route",
    "sensor_timeout",
    "manual_stop",
]

ControlMode = Literal["rule_based", "autopilot", "bc_policy"]


@dataclass(frozen=True)
class EgoState:
    frame: int
    timestamp_s: float
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    speed_mps: float
    acceleration_mps2: float
    angular_velocity_z_dps: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SensorFrame:
    sensor_name: str
    frame: int | None
    timestamp_s: float | None
    path: str | None
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteState:
    command: RouteCommand
    target_speed_mps: float
    route_progress_m: float | None = None
    distance_to_goal_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlState:
    throttle: float
    steer: float
    brake: float
    hand_brake: bool = False
    reverse: bool = False

    def clipped(self) -> "ControlState":
        return ControlState(
            throttle=min(1.0, max(0.0, float(self.throttle))),
            steer=min(1.0, max(-1.0, float(self.steer))),
            brake=min(1.0, max(0.0, float(self.brake))),
            hand_brake=bool(self.hand_brake),
            reverse=bool(self.reverse),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.clipped())


@dataclass(frozen=True)
class NormalizedAction:
    steer: float
    acceleration: float

    def clipped(self) -> "NormalizedAction":
        return NormalizedAction(
            steer=min(1.0, max(-1.0, float(self.steer))),
            acceleration=min(1.0, max(-1.0, float(self.acceleration))),
        )

    def to_control(self) -> ControlState:
        action = self.clipped()
        throttle = max(0.0, action.acceleration)
        brake = max(0.0, -action.acceleration)
        return ControlState(throttle=throttle, steer=action.steer, brake=brake)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.clipped())


@dataclass(frozen=True)
class EventState:
    collision: bool = False
    collision_actor_type: str | None = None
    collision_impulse: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TerminationState:
    done: bool
    reason: TerminationReason

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyState:
    name: str
    control_mode: ControlMode
    is_expert: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutRecord:
    episode_id: str
    step: int
    carla_frame: int
    ego: EgoState
    route: RouteState
    action: NormalizedAction
    control: ControlState
    camera: SensorFrame
    termination: TerminationState
    policy: PolicyState
    events: EventState
    expert_control: ControlState | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "carla_frame": self.carla_frame,
            "ego": self.ego.to_dict(),
            "route": self.route.to_dict(),
            "action": self.action.to_dict(),
            "control": self.control.to_dict(),
            "camera": self.camera.to_dict(),
            "termination": self.termination.to_dict(),
            "policy": self.policy.to_dict(),
            "events": self.events.to_dict(),
            "expert_control": self.expert_control.to_dict() if self.expert_control else None,
        }
