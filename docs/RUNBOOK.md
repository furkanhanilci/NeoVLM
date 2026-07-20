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
| `make bc-smoke` | `scripts/run_bc_smoke.sh` | `micromamba run -n vlm` | Trains the tiny BC policy on the cached feature smoke set and writes `results/bc_smoke/bc_checkpoint.pt`. |
| `make bc-rollout-smoke` | `scripts/run_bc_rollout_smoke.sh` | unified CARLA + torch/VLM env required | Attempts learned-policy closed-loop rollout into `results/bc_rollout_smoke/`. Currently blocked in the split local env. |
| `make validate-carla-dataset` | `scripts/validate_carla_dataset.py` | script shebang/current Python | Validates `results/datasets/carla_il_smoke/episode_000`. |
| `make smoke` | `scripts/run_smoke_tests.sh` | mixed | Runs broad host/env checks, but some CARLA/Bench2Drive failures are intentionally masked with `|| true`. Use targeted checks for gating. |

## BC Policy Smoke

The offline IL warm-start smoke path is:

```bash
make feature-cache-smoke
make bc-smoke
```

This trains `resampler + FastPolicy` from cached frozen-VLM hidden states and writes a checkpoint under `results/bc_smoke/`, which is ignored by git.

Closed-loop learned-policy rollout is wired through `make bc-rollout-smoke`, but the current local environments are split: the `vlm` environment has torch/VLM and cannot import CARLA 0.9.15 PythonAPI, while the `carla` environment imports CARLA but does not have torch. Until a unified eval environment exists, use the CARLA-free `tests/test_bc_agent.py` coverage and `make bc-smoke` as the gated checks.

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

## `metadata.jsonl`

Each line is one `RolloutRecord` serialized as JSON. Top-level fields:

| Field | Meaning |
|-------|---------|
| `episode_id` | Episode id matching the manifest. |
| `step` | Zero-based rollout step. |
| `carla_frame` | CARLA simulator frame id. |
| `ego` | Ego pose, speed, acceleration, angular velocity, and timestamp. |
| `route` | Route command, target speed, optional progress/distance fields. |
| `action` | Normalized policy action: `steer` and `acceleration`, clipped to `[-1, 1]`. |
| `control` | CARLA control command: throttle, steer, brake, hand brake, reverse. |
| `camera` | RGB sensor metadata and relative saved frame path when saved. |
| `termination` | `done` flag and reason (`running`, `max_steps`, `collision`, `off_route`, `sensor_timeout`, `manual_stop`). |
| `policy` | Policy name, control mode, and whether it is expert/autopilot. |
| `events` | Collision state and optional collision actor/impulse. |
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
