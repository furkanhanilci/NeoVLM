"""Action conversion helpers for CARLA integration."""

from __future__ import annotations

import carla

from vlm_driving.carla.observations import ControlState, NormalizedAction


def normalized_to_control(action: NormalizedAction) -> ControlState:
    return action.to_control()


def control_to_carla(control: ControlState) -> carla.VehicleControl:
    clipped = control.clipped()
    return carla.VehicleControl(
        throttle=float(clipped.throttle),
        steer=float(clipped.steer),
        brake=float(clipped.brake),
        hand_brake=clipped.hand_brake,
        reverse=clipped.reverse,
    )
