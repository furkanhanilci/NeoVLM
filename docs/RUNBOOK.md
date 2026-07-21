# CARLA Runbook

This runbook documents the local end-to-end smoke path for the thesis codebase: start a CARLA server, verify connectivity, run a short closed-loop rollout, and inspect the generated dataset-like output.

## Preconditions

- Run commands from the repository root: `/home/mehmet-g-rkem/research/vlm-driving-thesis`.
- Micromamba is available at `~/bin/micromamba`.
- Set the same mamba root used during setup before running environment-backed commands:

```bash
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
export PATH="$HOME/bin:$PATH"
```

- The `carla` environment is installed and the local package has been installed editable:

```bash
micromamba env create -f environments/carla.lock.yml
micromamba run -n carla pip install -e . --no-deps
```

- CARLA 0.9.15 is present under `third_party/CARLA_0.9.15`.
- The CARLA server listens on port `2000`.

## Command Flow

```bash
make carla-start
make carla-status
make carla-rollout-smoke
test -f results/smoke_rollout/manifest.json && head -1 results/smoke_rollout/metadata.jsonl | python -m json.tool
```

Use `make carla-window` instead of `make carla-start` when a visible CARLA window is needed. The default detached server path is intended for offscreen/headless smoke runs.

## Make Targets

| Target | Script | Environment | Purpose |
|--------|--------|-------------|---------|
| `make carla-start` | `scripts/start_carla_server_detached.sh` | host shell | Starts CARLA detached, normally offscreen. |
| `make carla-window` | `scripts/start_carla_window_detached.sh` | host shell | Starts CARLA detached with a visible window. |
| `make carla-status` | `scripts/carla_status.sh` | host shell + CARLA client check | Checks whether CARLA is active/reachable. |
| `make carla-rollout-smoke` | `scripts/run_carla_rollout_smoke.sh` | `micromamba run -n carla` | Runs an 80-step rule-based rollout into `results/smoke_rollout/`. |
| `make carla-dataset-smoke` | `scripts/run_carla_dataset_smoke.sh` | `micromamba run -n carla` | Runs the IL dataset smoke path. |
| `make carla-dataset-collect` | `scripts/run_carla_dataset_collect.sh` | `carla` collection + `vlm` split manifest | Collects multiple autopilot episodes and writes `split_manifest.json`; T-019 live scale run uses this path. |
| `make dataset-stats` | `scripts/run_dataset_stats.sh` | `micromamba run -n vlm` | Summarizes dataset QA before spending GPU time on feature cache. |
| `make eval-report` | `scripts/run_eval_report.sh` | `micromamba run -n vlm` | Scores saved rollout directories with the CARLA-free eval metrics layer. |
| `make bc-smoke` | `scripts/run_bc_smoke.sh` | `micromamba run -n vlm` | Trains the tiny BC policy on the cached feature smoke set and writes `results/bc_smoke/bc_checkpoint.pt`. |
| `make feature-cache-dataset` | `scripts/build_feature_cache_dataset.sh` | `micromamba run -n vlm` + CUDA | Builds per-episode frozen-Qwen caches for `results/datasets/carla_il_collect/`. |
| `make bc-train` | `scripts/run_bc_train.sh` | `micromamba run -n vlm` | Trains BC on the train split and reports validation loss. |
| `make bc-rollout-smoke` | `scripts/run_bc_rollout_smoke.sh` | deprecated in this split env | In-process learned-policy rollout path; use `make bc-bridge-smoke` instead because CARLA and torch/VLM run in separate envs. |
| `make bc-bridge-smoke` | `scripts/run_bc_bridge_smoke.sh` | `vlm` policy server + `carla` rollout client | Runs the two-process learned-policy bridge into `results/bc_bridge_smoke/`; requires a running CARLA server. |
| `make validate-carla-dataset` | `scripts/validate_carla_dataset.py` | script shebang/current Python | Validates `results/datasets/carla_il_smoke/episode_000`. |
| `make smoke` | `scripts/run_smoke_tests.sh` | mixed | Runs broad host/env checks, but some CARLA/Bench2Drive failures are intentionally masked with `|| true`. Use targeted checks for gating. |

## BC Policy Smoke

The offline IL warm-start smoke path is:

```bash
make feature-cache-smoke
make bc-smoke
```

This trains `resampler + FastPolicy` from cached frozen-VLM hidden states and writes a checkpoint under `results/bc_smoke/`, which is ignored by git.

`make bc-rollout-smoke` is deprecated for this local split environment: the `vlm` environment has torch/VLM but cannot import CARLA 0.9.15 PythonAPI, while the `carla` environment imports CARLA but does not have torch. Use `make bc-bridge-smoke` for learned-policy closed-loop checks.

## BC Bridge Smoke

The live learned-policy rollout uses a two-process bridge so CARLA stays in the `carla` environment and torch/Qwen stay in the `vlm` environment:

```bash
make carla-start
make carla-status
make bc-bridge-smoke
```

`scripts/run_bc_bridge_smoke.sh` starts `scripts/run_policy_server.sh` in the background, waits for the socket `ready` handshake, then runs `control_mode="bc_remote"` in the `carla` environment. The client writes each RGB frame to disk and sends the frame path plus rollout record state over localhost TCP; the VLM-side server returns a normalized steering/acceleration action.

Useful overrides:

```bash
BC_BRIDGE_FRAMES=20 make bc-bridge-smoke
BC_BRIDGE_IMAGE_WIDTH=320 BC_BRIDGE_IMAGE_HEIGHT=180 make bc-bridge-smoke
POLICY_SERVER_PORT=8877 BC_CHECKPOINT=results/bc_smoke/bc_checkpoint.pt make bc-bridge-smoke
BC_BRIDGE_TERMINATE_ON_COLLISION=1 make bc-bridge-smoke
```

Outputs are written to `results/bc_bridge_smoke/`, with policy-server logs in `setup_logs/policy_server.log` and client smoke logs in `setup_logs/bc_bridge_smoke.log`. On 8 GB GPUs, offscreen CARLA and Qwen may not fit at the same time; use `POLICY_SERVER_DEVICE=cpu` for a slower integration-only check or install a supported 4-bit stack before expecting GPU live rollout.

## Eval Mode Logging

T-021 adds Leaderboard-style rollout logging without changing the data-collection default:

- `RolloutConfig.terminate_on_collision=True` remains the default. Dataset collection still stops on the first new collision event or `max_steps`.
- Eval mode sets `terminate_on_collision=False`. Collisions are accumulated as discrete events and termination becomes `goal_reached`, `max_steps`, or `blocked`.
- `route_seed` deterministically selects the spawn/destination route pair when set; when omitted it defaults to `seed` for manifest traceability.
- `route_progress_m`, `route_length_m`, and `distance_to_goal_m` are logged from a GlobalRoutePlanner spawn-to-destination route with a forward-only `RouteProgressTracker`; progress starts near zero at spawn and never decreases across rollout steps.
- `collision_event_count` and `collision_new` mark only new collision events on each step; latched collision state is not counted repeatedly.
- `policy.latency_ms` is logged around `policy.act` for `bc_policy` and `bc_remote`.
- Eval-mode `control_mode="autopilot"` disables Traffic Manager autopilot and uses the internal sequential waypoint follower directly. CARLA 0.9.15 `TrafficManager.set_path` can silently no-op, so it is not the default route-following path. Dataset collection keeps the old Traffic Manager behavior because `terminate_on_collision=True` remains the default.
- The waypoint follower is a simple geometric controller for validation. A future `BasicAgent`/`BehaviorAgent` route follower can replace it when traffic-light and lane behavior need higher fidelity.
- BC policies are not goal-conditioned in this task; their Route Completion remains an evaluation of how well the local driving policy happens to stay on the planned route.

Dry checks before a live eval run:

```bash
bash -n scripts/run_bc_bridge_smoke.sh
PYTHONPATH="src:third_party/CARLA_0.9.15/PythonAPI/carla" micromamba run -n carla python -c "from vlm_driving.carla import RolloutConfig; print(RolloutConfig().terminate_on_collision)"
micromamba run -n vlm pytest tests/test_eval_metrics.py tests/test_route_progress.py -q
```

T-022 live scoring flow, with CARLA already running:

```bash
make carla-status
BC_BRIDGE_TERMINATE_ON_COLLISION=0 make bc-bridge-smoke
make eval-report
cat results/eval_report/eval_report.txt
```

Traffic-light and lane-invasion logging remain T-022/live-validation items unless their event fields are explicitly present in rollout metadata. Stop sign, scenario timeout, yield, and min-speed remain `N/A - event logging required`.

## Dataset Collection

T-018 keeps the next scale step CARLA-ready without starting a live run. Start from a modest first scale dataset rather than the 100k-frame research target:

- Initial target: `CARLA_NUM_EPISODES=50`, `CARLA_FRAMES_PER_EPISODE=200`, `CARLA_SAVE_EVERY_N_FRAMES=5`. This yields about 2,000 saved RGB frames.
- Feature-cache budget: T-011 measured about 1.457 MiB per saved frame, so 2,000 frames is about 2.85 GiB. Keep the first pass under 5,000 saved frames, about 7.1 GiB cache.
- Split policy: split by episode, not by frame, with the default 85/15 train/val split and `CARLA_SPLIT_SEED=17`.
- Diversity knobs: `CARLA_BASE_SEED` changes spawn/traffic-manager seed; `CARLA_ROUTE_COMMANDS` and `CARLA_WEATHER_PRESETS` cycle per episode. `CARLA_TOWNS` can be set to comma-separated CARLA map names when changing town is needed.

Scale/disk budget:

| Run | Episodes | Frames/episode | Save every | Saved frames | Cache estimate |
|-----|----------|----------------|------------|--------------|----------------|
| T-017 proof dataset | 6 | 150 | 5 | 180 | 0.26 GiB |
| T-019 first scale target | 50 | 200 | 5 | 2,000 | 2.85 GiB |
| Local first-pass ceiling | 125 | 200 | 5 | 5,000 | 7.11 GiB |
| Avoid for local iteration | n/a | n/a | n/a | 100,000 | 142.3 GiB |

`scripts/run_carla_dataset_collect.sh` refuses targets above `CARLA_MAX_SAVED_FRAMES=5000` unless that budget is raised explicitly.

Dry validation before a CARLA run:

```bash
bash -n scripts/run_carla_dataset_collect.sh
micromamba run -n vlm python -c "from vlm_driving.data import discover_episodes, split_episodes; print('ok')"
micromamba run -n carla python -c "from vlm_driving.carla import RolloutConfig; from vlm_driving.data.splits import discover_episodes; print('carla dry import ok')"
```

Actual collection, with CARLA already running:

```bash
make carla-start
make carla-status
CARLA_NUM_EPISODES=50 CARLA_FRAMES_PER_EPISODE=200 CARLA_SAVE_EVERY_N_FRAMES=5 make carla-dataset-collect
```

The script writes episodes under `results/datasets/carla_il_collect/episode_XXXX/` and an episode-level `split_manifest.json` at the dataset root.

Before spending GPU time on frozen-Qwen cache, run dataset QA:

```bash
make dataset-stats
```

The T-017 proof dataset should report saved action nonzero around `64.4%` and `double_pedal=0`; large deviations in a new run are a signal to inspect collection quality before caching.

After QA passes, keep CARLA closed so the VLM has GPU memory, then build per-episode caches and train BC on the split:

```bash
make carla-status   # should show no CARLA process before GPU cache build
make feature-cache-dataset
make bc-train
```

`make feature-cache-dataset` writes `results/feature_cache/carla_il_collect/episode_XXXX/cache_manifest.json`, one cache namespace per episode. This avoids collisions from repeated frame names like `frame_00000.png`. `make bc-train` reads `split_manifest.json`, trains on train episodes, evaluates validation loss each epoch, and writes `results/bc_train/bc_checkpoint.pt`.

## Rollout Output

`make carla-rollout-smoke` writes to:

```text
results/smoke_rollout/
  manifest.json
  metadata.jsonl
  frames/
    frame_00000.png
    ...
```

The current smoke configuration writes 80 metadata rows and saves every fifth RGB frame, so a passing run usually has 16 frame images.

## `manifest.json`

The manifest uses `schema_version: carla_rollout_v1` and records run-level metadata:

| Field | Meaning |
|-------|---------|
| `schema_version` | Output schema identifier, currently `carla_rollout_v1`. |
| `dataset_name` | Logical dataset/run name, currently `carla_smoke`. |
| `episode_id` | Episode identifier, currently `smoke_rollout_000`. |
| `metadata_file` | Relative metadata JSONL filename. |
| `frames_dir` | Relative image frame directory. |
| `num_steps` | Number of rollout records written. |
| `num_saved_frames` | Number of frame files saved. |
| `control_mode` | Policy/control source, e.g. `rule_based`. |
| `map_name` | CARLA map used by the run. |
| `fixed_delta_seconds` | Simulator fixed time step. |
| `image_width`, `image_height` | RGB camera resolution. |
| `route_command` | Route command label, e.g. `lane_follow`. |
| `target_speed_mps` | Target speed in meters per second. |
| `weather_preset` | CARLA weather preset used by the run when set. |
| `route_length_m` | Planned GlobalRoutePlanner route length in meters. |
| `route_seed` | Seed used for deterministic route selection. |
| `destination_spawn_index` | Spawn-point index used as the destination. |

## `metadata.jsonl`

Each line is one `RolloutRecord` serialized as JSON. Top-level fields:

| Field | Meaning |
|-------|---------|
| `episode_id` | Episode id matching the manifest. |
| `step` | Zero-based rollout step. |
| `carla_frame` | CARLA simulator frame id. |
| `ego` | Ego pose, speed, acceleration, angular velocity, and timestamp. |
| `route` | Route command, target speed, route length, progress, and distance-to-goal fields. |
| `action` | Normalized policy action: `steer` and `acceleration`, clipped to `[-1, 1]`. |
| `control` | CARLA control command: throttle, steer, brake, hand brake, reverse. |
| `camera` | RGB sensor metadata and relative saved frame path when saved. |
| `termination` | `done` flag and reason (`running`, `max_steps`, `collision`, `goal_reached`, `blocked`, `off_route`, `sensor_timeout`, `manual_stop`). |
| `policy` | Policy name, control mode, expert/autopilot flag, and optional `latency_ms`. |
| `events` | Collision event state, `collision_new`, `collision_event_count`, and optional collision actor/impulse or future traffic/lane events. |
| `expert_control` | Expert CARLA control when available; `null` for the rule-based smoke rollout. |

Inspect the first row with:

```bash
head -1 results/smoke_rollout/metadata.jsonl | python -m json.tool
```

## Troubleshooting

- **Timeout or connection refused:** start CARLA first with `make carla-start`, then rerun `make carla-status`. The rollout connects to port `2000` and does not start the server itself.
- **Port conflict:** stop the existing CARLA/UE process or verify which process owns port `2000` before starting another server.
- **No visible simulator window:** `make carla-start` is the detached/offscreen path. Use `make carla-window` for an interactive windowed server.
- **Import errors for `carla` or `vlm_driving`:** confirm `MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"`, use the `carla` environment, and reinstall the local package with `micromamba run -n carla pip install -e . --no-deps`.
- **`make smoke` appears green despite CARLA/Bench2Drive issues:** `scripts/run_smoke_tests.sh` currently appends `|| true` to `check_bench2drive.sh` and `check_carla.sh`. Treat `make smoke` as a broad convenience check, not a strict CARLA gate.
