"""Minimal closed-loop CARLA rollout and dataset smoke helpers."""

from __future__ import annotations

import math
import queue
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import carla

from vlm_driving.carla.actions import control_to_carla, normalized_to_control
from vlm_driving.carla.logger import DatasetManifest, JsonlRolloutLogger
from vlm_driving.carla.metrics import write_rollout_metrics
from vlm_driving.carla.observations import (
    ControlMode,
    ControlState,
    EgoState,
    EventState,
    NormalizedAction,
    PolicyState,
    control_to_normalized_action,
    RolloutRecord,
    RouteState,
    SensorFrame,
    TerminationState,
)
from vlm_driving.carla.route_progress import RouteProgressTracker, route_length_m


@dataclass(frozen=True)
class RolloutConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_s: float = 10.0
    frames: int = 120
    fixed_delta_seconds: float = 0.05
    output_dir: Path = Path("results/smoke_rollout")
    save_every_n_frames: int = 5
    image_width: int = 800
    image_height: int = 450
    camera_fov: float = 90.0
    seed: int = 7
    episode_id: str = "smoke_rollout_000"
    dataset_name: str = "carla_smoke"
    route_command: str = "lane_follow"
    target_speed_mps: float = 6.0
    control_mode: ControlMode = "rule_based"
    overwrite: bool = True
    traffic_manager_port: int = 8000
    bc_checkpoint_path: Path = Path("results/bc_smoke/bc_checkpoint.pt")
    bc_hidden_source: str = "live"
    bc_feature_cache_dir: Path | None = None
    bc_command_text: str = "You are driving in CARLA. Keep lane and continue safely."
    bc_device: str | None = None
    policy_server_host: str = "127.0.0.1"
    policy_server_port: int = 8765
    policy_server_timeout_s: float = 60.0
    weather_preset: str | None = None
    route_seed: int | None = None
    destination_spawn_index: int | None = None
    route_sampling_resolution_m: float = 2.0
    goal_distance_threshold_m: float = 5.0
    terminate_on_collision: bool = True
    blocked_speed_threshold_mps: float = 0.2
    blocked_steps_threshold: int = 80
    route_progress_lookahead_segments: int = 25
    route_progress_max_step_m: float = 25.0
    route_follow_waypoint_reached_m: float = 4.0


def _speed_mps(vehicle: carla.Vehicle) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def _acceleration_mps2(vehicle: carla.Vehicle) -> float:
    acceleration = vehicle.get_acceleration()
    return math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)


def _ego_state(vehicle: carla.Vehicle, frame: int, timestamp_s: float) -> EgoState:
    transform = vehicle.get_transform()
    angular_velocity = vehicle.get_angular_velocity()
    return EgoState(
        frame=frame,
        timestamp_s=timestamp_s,
        x=transform.location.x,
        y=transform.location.y,
        z=transform.location.z,
        yaw_deg=transform.rotation.yaw,
        pitch_deg=transform.rotation.pitch,
        roll_deg=transform.rotation.roll,
        speed_mps=_speed_mps(vehicle),
        acceleration_mps2=_acceleration_mps2(vehicle),
        angular_velocity_z_dps=angular_velocity.z,
    )


def _safe_action(frame_idx: int, speed_mps: float, target_speed_mps: float) -> NormalizedAction:
    acceleration = 0.35 if speed_mps < target_speed_mps else 0.08
    if speed_mps > target_speed_mps + 1.0:
        acceleration = -0.15
    steer = 0.08 * math.sin(frame_idx / 35.0)
    return NormalizedAction(steer=steer, acceleration=acceleration)


def _control_to_action(control: ControlState) -> NormalizedAction:
    return control_to_normalized_action(control)


def _carla_control_to_state(control: carla.VehicleControl) -> ControlState:
    return ControlState(
        throttle=control.throttle,
        steer=control.steer,
        brake=control.brake,
        hand_brake=control.hand_brake,
        reverse=control.reverse,
    )


def _choose_route_transforms(
    world: carla.World,
    spawn_rng: random.Random,
    destination_rng: random.Random,
    destination_spawn_index: int | None,
) -> tuple[carla.Transform, carla.Transform, int]:
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("CARLA map has no spawn points")
    spawn_index = spawn_rng.randrange(len(spawn_points))
    spawn_transform = spawn_points[spawn_index]
    if destination_spawn_index is not None:
        destination_index = destination_spawn_index % len(spawn_points)
        return spawn_transform, spawn_points[destination_index], destination_index

    far_candidates = [
        index
        for index, transform in enumerate(spawn_points)
        if index != spawn_index and _transform_distance_m(spawn_transform, transform) >= 20.0
    ]
    candidates = far_candidates or [index for index in range(len(spawn_points)) if index != spawn_index]
    if not candidates:
        return spawn_transform, spawn_transform, spawn_index
    destination_index = destination_rng.choice(candidates)
    return spawn_transform, spawn_points[destination_index], destination_index


def _transform_distance_m(first: carla.Transform, second: carla.Transform) -> float:
    first_location = first.location
    second_location = second.location
    return math.sqrt(
        (first_location.x - second_location.x) ** 2
        + (first_location.y - second_location.y) ** 2
        + (first_location.z - second_location.z) ** 2
    )


def _weather_preset(name: str) -> carla.WeatherParameters:
    preset = getattr(carla.WeatherParameters, name, None)
    if preset is None:
        raise ValueError(f"unknown CARLA weather preset: {name}")
    return preset


def _make_camera_frame(
    image: carla.Image | None,
    image_path: Path | None,
    output_dir: Path,
    width: int,
    height: int,
) -> SensorFrame:
    return SensorFrame(
        sensor_name="rgb_front",
        frame=image.frame if image is not None else None,
        timestamp_s=image.timestamp if image is not None else None,
        path=str(image_path.relative_to(output_dir)) if image_path else None,
        width=width,
        height=height,
    )


def _carla_image_to_pil(image: carla.Image):
    from PIL import Image

    return Image.frombytes(
        "RGBA",
        (image.width, image.height),
        bytes(image.raw_data),
        "raw",
        "BGRA",
    ).convert("RGB")


def _make_bc_agent(config: RolloutConfig):
    if not config.bc_checkpoint_path.exists():
        raise FileNotFoundError(f"missing BC checkpoint: {config.bc_checkpoint_path}; run make bc-smoke first")
    try:
        from vlm_driving.carla.bc_agent import BCAgent
    except ImportError as exc:
        raise RuntimeError(
            "bc_policy rollout requires torch/VLM dependencies in the same Python process as CARLA. "
            "Current local setup has CARLA in the 'carla' env and torch/VLM in the 'vlm' env; "
            "run CARLA-less agent tests or create a unified eval env before live rollout."
        ) from exc
    return BCAgent(
        checkpoint_path=config.bc_checkpoint_path,
        hidden_source=config.bc_hidden_source,
        feature_cache_dir=config.bc_feature_cache_dir,
        command_text=config.bc_command_text,
        device=config.bc_device,
    )


def _plan_route_points(
    world_map: carla.Map,
    origin: carla.Location,
    destination: carla.Location,
    sampling_resolution_m: float,
) -> list[carla.Location]:
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    if sampling_resolution_m <= 0.0:
        raise ValueError("route_sampling_resolution_m must be positive")
    if _location_distance_m(origin, destination) <= 1e-3:
        return [origin, destination]
    planner = GlobalRoutePlanner(world_map, sampling_resolution_m)
    route_trace = planner.trace_route(origin, destination)
    points = [origin]
    points.extend(waypoint.transform.location for waypoint, _road_option in route_trace)
    points.append(destination)
    return _dedupe_route_points(points)


def _dedupe_route_points(points: list[carla.Location]) -> list[carla.Location]:
    deduped: list[carla.Location] = []
    for point in points:
        if not deduped or _location_distance_m(point, deduped[-1]) > 1e-3:
            deduped.append(point)
    return deduped


def _location_distance_m(first: carla.Location, second: carla.Location) -> float:
    return math.sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2 + (first.z - second.z) ** 2)


class _SequentialRouteFollower:
    def __init__(
        self,
        route_points: list[carla.Location],
        target_speed_mps: float,
        waypoint_reached_m: float,
    ) -> None:
        self.route_points = route_points
        self.target_speed_mps = target_speed_mps
        self.waypoint_reached_m = waypoint_reached_m
        self.target_index = 1 if len(route_points) > 1 else 0

    def act(self, vehicle: carla.Vehicle) -> tuple[NormalizedAction, ControlState]:
        location = vehicle.get_location()
        while self.target_index < len(self.route_points) - 1:
            if _location_distance_m(location, self.route_points[self.target_index]) > self.waypoint_reached_m:
                break
            self.target_index += 1

        target = self.route_points[self.target_index] if self.route_points else location
        transform = vehicle.get_transform()
        desired_yaw = math.atan2(target.y - location.y, target.x - location.x)
        current_yaw = math.radians(transform.rotation.yaw)
        yaw_error = _normalize_radians(desired_yaw - current_yaw)
        steer = _clip(yaw_error / math.radians(45.0), -1.0, 1.0)
        speed = _speed_mps(vehicle)
        acceleration = 0.45 if speed < self.target_speed_mps else 0.05
        if speed > self.target_speed_mps + 1.0:
            acceleration = -0.25
        action = NormalizedAction(steer=steer, acceleration=acceleration)
        control = normalized_to_control(action)
        return action, control


def _configure_autopilot(
    vehicle: carla.Vehicle,
    traffic_manager: carla.TrafficManager,
    route_points: list[carla.Location],
    config: RolloutConfig,
) -> _SequentialRouteFollower | None:
    eval_mode = not config.terminate_on_collision
    if eval_mode:
        vehicle.set_autopilot(False, traffic_manager.get_port())
        return _SequentialRouteFollower(
            route_points=route_points,
            target_speed_mps=config.target_speed_mps,
            waypoint_reached_m=config.route_follow_waypoint_reached_m,
        )

    vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.vehicle_percentage_speed_difference(vehicle, 35.0)
    return None


def _route_state(config: RolloutConfig, tracker: RouteProgressTracker, ego: EgoState) -> RouteState:
    progress = tracker.update((ego.x, ego.y, ego.z))
    return RouteState(
        command=config.route_command,
        target_speed_mps=config.target_speed_mps,
        route_progress_m=progress.route_progress_m,
        route_length_m=progress.route_length_m,
        distance_to_goal_m=progress.distance_to_goal_m,
    )


def _normalize_radians(value: float) -> float:
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


def _collision_event(collision_events: list[carla.CollisionEvent], processed_count: int) -> tuple[EventState, int]:
    new_events = collision_events[processed_count:]
    if not new_events:
        return EventState(), len(collision_events)
    event = new_events[-1]
    impulse = event.normal_impulse
    magnitude = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    return EventState(
        collision=True,
        collision_new=True,
        collision_event_count=len(new_events),
        collision_actor_type=event.other_actor.type_id if event.other_actor else None,
        collision_impulse=magnitude,
    ), len(collision_events)


def _termination_state(
    config: RolloutConfig,
    events: EventState,
    route: RouteState,
    blocked_steps: int,
    is_last_step: bool,
) -> TerminationState:
    if config.terminate_on_collision and events.collision_event_count > 0:
        return TerminationState(done=True, reason="collision")

    eval_mode = not config.terminate_on_collision
    if eval_mode and route.distance_to_goal_m is not None and route.distance_to_goal_m <= config.goal_distance_threshold_m:
        return TerminationState(done=True, reason="goal_reached")
    if eval_mode and blocked_steps >= config.blocked_steps_threshold:
        return TerminationState(done=True, reason="blocked")
    if is_last_step:
        return TerminationState(done=True, reason="max_steps")
    return TerminationState(done=False, reason="running")


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def run_rollout(config: RolloutConfig) -> Path:
    rng = random.Random(config.seed)
    effective_route_seed = config.route_seed if config.route_seed is not None else config.seed
    if config.route_seed is None:
        spawn_rng = rng
        destination_rng = random.Random(effective_route_seed)
    else:
        route_rng = random.Random(effective_route_seed)
        spawn_rng = route_rng
        destination_rng = route_rng
    output_dir = config.output_dir
    _prepare_output_dir(output_dir, config.overwrite)

    client = carla.Client(config.host, config.port)
    client.set_timeout(config.timeout_s)
    world = client.get_world()
    world_map = world.get_map()
    map_name = world_map.name
    original_settings = world.get_settings()
    original_weather = world.get_weather()
    traffic_manager = client.get_trafficmanager(config.traffic_manager_port)

    vehicle = None
    camera = None
    collision_sensor = None
    actors: list[carla.Actor] = []
    image_queue: queue.Queue[carla.Image] = queue.Queue(maxsize=8)
    collision_events: list[carla.CollisionEvent] = []
    processed_collision_count = 0
    saved_frames = 0
    completed_steps = 0
    written_records: list[dict] = []
    bc_agent = None
    remote_policy = None
    autopilot_follower = None
    route_points: list[carla.Location] = []
    route_tracker = None
    route_length_value: float | None = None
    destination_index: int | None = None
    blocked_steps = 0

    try:
        if config.weather_preset is not None:
            world.set_weather(_weather_preset(config.weather_preset))

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config.fixed_delta_seconds
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(config.seed)

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = rng.choice(blueprint_library.filter("vehicle.tesla.model3"))
        spawn_transform, destination_transform, destination_index = _choose_route_transforms(
            world,
            spawn_rng=spawn_rng,
            destination_rng=destination_rng,
            destination_spawn_index=config.destination_spawn_index,
        )
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
        if vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")
        actors.append(vehicle)
        route_points = _plan_route_points(
            world_map,
            origin=vehicle.get_location(),
            destination=destination_transform.location,
            sampling_resolution_m=config.route_sampling_resolution_m,
        )
        route_length_value = route_length_m(route_points)
        route_tracker = RouteProgressTracker(
            route_points,
            lookahead_segments=config.route_progress_lookahead_segments,
            max_progress_step_m=config.route_progress_max_step_m,
        )

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(config.image_width))
        camera_bp.set_attribute("image_size_y", str(config.image_height))
        camera_bp.set_attribute("fov", str(config.camera_fov))
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actors.append(camera)
        camera.listen(image_queue.put)

        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        actors.append(collision_sensor)
        collision_sensor.listen(collision_events.append)

        if config.control_mode == "autopilot":
            autopilot_follower = _configure_autopilot(vehicle, traffic_manager, route_points, config)
        elif config.control_mode == "bc_policy":
            bc_agent = _make_bc_agent(config)
        elif config.control_mode == "bc_remote":
            from vlm_driving.eval.policy_client import RemoteBCPolicy

            remote_policy = RemoteBCPolicy(
                host=config.policy_server_host,
                port=config.policy_server_port,
                timeout_s=config.policy_server_timeout_s,
            ).connect()
            remote_policy.reset()

        policy_name = {
            "autopilot": "carla_autopilot",
            "bc_policy": "bc_il",
            "bc_remote": "bc_remote_il",
        }.get(config.control_mode, "rule_based_smoke")

        with JsonlRolloutLogger(output_dir) as logger:
            for step in range(config.frames):
                if config.control_mode == "rule_based":
                    speed = _speed_mps(vehicle)
                    action = _safe_action(step, speed, config.target_speed_mps)
                    control = normalized_to_control(action)
                    vehicle.apply_control(control_to_carla(control))
                elif autopilot_follower is not None:
                    action, control = autopilot_follower.act(vehicle)
                    vehicle.apply_control(control_to_carla(control))

                frame_id = world.tick()
                snapshot = world.get_snapshot()

                image = None
                image_path = None
                try:
                    image = image_queue.get(timeout=2.0)
                    should_save_frame = step % config.save_every_n_frames == 0 or config.control_mode == "bc_remote"
                    if should_save_frame:
                        image_path = logger.frames_dir / f"frame_{step:05d}.png"
                        image.save_to_disk(str(image_path))
                        saved_frames += 1
                except queue.Empty:
                    image = None
                    image_path = None

                ego_state = _ego_state(vehicle, frame_id, snapshot.timestamp.elapsed_seconds)
                if route_tracker is None:
                    raise RuntimeError("route progress tracker was not initialized")
                route = _route_state(config, route_tracker, ego_state)
                camera_frame = _make_camera_frame(
                    image=image,
                    image_path=image_path,
                    output_dir=output_dir,
                    width=config.image_width,
                    height=config.image_height,
                )

                policy_latency_ms = None
                if config.control_mode == "autopilot":
                    if autopilot_follower is None:
                        control = _carla_control_to_state(vehicle.get_control())
                        action = _control_to_action(control)
                    expert_control = control
                elif config.control_mode == "bc_policy":
                    if bc_agent is None:
                        raise RuntimeError("bc_policy selected but BC agent was not initialized")
                    agent_record = {
                        "ego": ego_state.to_dict(),
                        "route": route.to_dict(),
                        "camera": camera_frame.to_dict(),
                    }
                    live_image = _carla_image_to_pil(image) if image is not None else None
                    policy_start_s = time.perf_counter()
                    action = bc_agent.act(agent_record, image=live_image)
                    policy_latency_ms = (time.perf_counter() - policy_start_s) * 1000.0
                    control = normalized_to_control(action)
                    vehicle.apply_control(control_to_carla(control))
                    expert_control = None
                elif config.control_mode == "bc_remote":
                    if remote_policy is None:
                        raise RuntimeError("bc_remote selected but remote policy client was not initialized")
                    if camera_frame.path is None:
                        raise RuntimeError("bc_remote requires a saved camera frame path for every step")
                    agent_record = {
                        "ego": ego_state.to_dict(),
                        "route": route.to_dict(),
                        "camera": camera_frame.to_dict(),
                    }
                    policy_start_s = time.perf_counter()
                    action = remote_policy.act(agent_record, output_dir / camera_frame.path)
                    policy_latency_ms = (time.perf_counter() - policy_start_s) * 1000.0
                    control = normalized_to_control(action)
                    vehicle.apply_control(control_to_carla(control))
                    expert_control = None
                else:
                    expert_control = None

                events, processed_collision_count = _collision_event(collision_events, processed_collision_count)
                if ego_state.speed_mps <= config.blocked_speed_threshold_mps:
                    blocked_steps += 1
                else:
                    blocked_steps = 0
                is_last_step = step == config.frames - 1
                termination = _termination_state(
                    config=config,
                    events=events,
                    route=route,
                    blocked_steps=blocked_steps,
                    is_last_step=is_last_step,
                )
                policy = PolicyState(
                    name=policy_name,
                    control_mode=config.control_mode,
                    is_expert=config.control_mode == "autopilot",
                    latency_ms=policy_latency_ms,
                )
                record = RolloutRecord(
                    episode_id=config.episode_id,
                    step=step,
                    carla_frame=frame_id,
                    ego=ego_state,
                    route=route,
                    action=action,
                    control=control,
                    camera=camera_frame,
                    termination=termination,
                    policy=policy,
                    events=events,
                    expert_control=expert_control,
                )
                record_dict = record.to_dict()
                logger.write(record_dict)
                written_records.append(record_dict)
                completed_steps += 1
                if termination.done:
                    break

            logger.write_manifest(
                DatasetManifest(
                    schema_version="carla_rollout_v1",
                    dataset_name=config.dataset_name,
                    episode_id=config.episode_id,
                    metadata_file="metadata.jsonl",
                    frames_dir="frames",
                    num_steps=completed_steps,
                    num_saved_frames=saved_frames,
                    control_mode=config.control_mode,
                    map_name=map_name,
                    fixed_delta_seconds=config.fixed_delta_seconds,
                    image_width=config.image_width,
                    image_height=config.image_height,
                    route_command=config.route_command,
                    target_speed_mps=config.target_speed_mps,
                    weather_preset=config.weather_preset,
                    route_length_m=route_length_value,
                    route_seed=effective_route_seed,
                    destination_spawn_index=destination_index,
                )
            )
            write_rollout_metrics(output_dir / "metrics.json", written_records)

        return output_dir
    finally:
        if vehicle is not None and config.control_mode == "autopilot":
            vehicle.set_autopilot(False, traffic_manager.get_port())
        if remote_policy is not None:
            remote_policy.close()
        if camera is not None:
            camera.stop()
        if collision_sensor is not None:
            collision_sensor.stop()
        for actor in reversed(actors):
            if actor.is_alive:
                actor.destroy()
        traffic_manager.set_synchronous_mode(False)
        if config.weather_preset is not None:
            world.set_weather(original_weather)
        world.apply_settings(original_settings)
        time.sleep(0.2)
