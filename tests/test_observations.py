import pytest

from vlm_driving.carla.observations import (
    ControlState,
    EgoState,
    EventState,
    NormalizedAction,
    PolicyState,
    RolloutRecord,
    RouteState,
    SensorFrame,
    TerminationState,
    control_to_normalized_action,
)


def test_normalized_action_clips_and_maps_to_control():
    action = NormalizedAction(steer=2.0, acceleration=-1.5)

    clipped = action.clipped()
    control = action.to_control()

    assert clipped.steer == 1.0
    assert clipped.acceleration == -1.0
    assert control.throttle == 0.0
    assert control.brake == 1.0
    assert control.steer == 1.0


def test_control_state_clips_to_carla_ranges():
    control = ControlState(throttle=2.0, steer=-2.0, brake=-0.5, hand_brake=1, reverse=0)

    assert control.to_dict() == {
        "throttle": 1.0,
        "steer": -1.0,
        "brake": 0.0,
        "hand_brake": True,
        "reverse": False,
    }


def test_control_to_normalized_action_uses_throttle_minus_brake():
    control = ControlState(throttle=0.7, steer=0.25, brake=0.2)

    action = control_to_normalized_action(control)
    roundtrip = action.to_control()

    assert action.steer == 0.25
    assert action.acceleration == pytest.approx(0.5)
    assert roundtrip.steer == 0.25
    assert roundtrip.throttle == pytest.approx(0.5)
    assert roundtrip.brake == 0.0
    assert control_to_normalized_action({"steer": -2.0, "throttle": 0.25, "brake": 1.75}).to_dict() == {
        "steer": -1.0,
        "acceleration": -1.0,
    }


def test_rollout_record_serializes_nested_schema():
    record = RolloutRecord(
        episode_id="ep",
        step=3,
        carla_frame=10,
        ego=EgoState(
            frame=10,
            timestamp_s=1.5,
            x=1.0,
            y=2.0,
            z=0.0,
            yaw_deg=90.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            speed_mps=4.0,
            acceleration_mps2=0.2,
            angular_velocity_z_dps=0.1,
        ),
        route=RouteState(command="lane_follow", target_speed_mps=6.0),
        action=NormalizedAction(steer=0.1, acceleration=0.2),
        control=ControlState(throttle=0.2, steer=0.1, brake=0.0),
        camera=SensorFrame(sensor_name="rgb_front", frame=10, timestamp_s=1.5, path="frames/000.png", width=800, height=450),
        termination=TerminationState(done=False, reason="running"),
        policy=PolicyState(name="test", control_mode="rule_based", is_expert=False),
        events=EventState(),
        expert_control=None,
    )

    data = record.to_dict()

    assert data["episode_id"] == "ep"
    assert data["route"]["command"] == "lane_follow"
    assert data["camera"]["path"] == "frames/000.png"
    assert data["expert_control"] is None
