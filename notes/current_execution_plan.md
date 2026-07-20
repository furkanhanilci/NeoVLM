# Current Execution Plan

Last updated: 2026-07-20

## Current State

The local research environment is ready: GPU/PyTorch, CARLA 0.9.15, Leaderboard, ScenarioRunner, Bench2Drive, thesis rendering, and the initial VLM driving model scaffold are in place. CARLA can run in offscreen mode or windowed mode. The current missing layer is the closed-loop CARLA interaction and data collection pipeline.

## Immediate Objective

Build the first runnable closed-loop CARLA smoke pipeline before adding Qwen, imitation learning, PPO, or full Bench2Drive evaluation.

The first pipeline should:

1. Connect to an already running CARLA server on port 2000.
2. Spawn an ego vehicle.
3. Attach an RGB camera sensor.
4. Tick the simulator for a short episode.
5. Apply a conservative rule-based control policy.
6. Record ego state, control, timestamps, and optional camera frames.
7. Write outputs under `results/smoke_rollout/`.
8. Expose a Makefile target so it can be rerun reliably.

## Milestone Order

### M1 - CARLA Closed-Loop Smoke

- `src/vlm_driving/carla/` package.
- Client/world helper.
- Ego vehicle spawn and cleanup.
- RGB camera queue.
- Simple lane-following or safe constant-throttle policy.
- JSONL rollout logger.
- `scripts/run_carla_rollout_smoke.sh`.
- `make carla-rollout-smoke`.

Success criterion: with CARLA running, the script completes a short rollout and writes `metadata.jsonl` plus frames.

### M2 - Observation And Action Interface

- Stable observation dataclasses.
- RGB image + ego kinematics + route command schema.
- Action normalization and CARLA VehicleControl mapping.
- Termination and reset conventions.

### M3 - Dataset Logger For IL

- Episode directory layout.
- Image frame storage.
- JSONL/Parquet metadata.
- Expert or autopilot action logging.
- Replay/inspection helper.

### M4 - VLM Token Provider

- Dummy token provider first.
- Frozen VLM loader later, gated by HF token/checkpoint availability.
- **Decision (2026-07-20): start with Qwen3-VL-2B, keep a config-swap upgrade path
  to 4B.** RTX 5060 has only 8 GB VRAM; an 8B VLM cannot coexist with CARLA render +
  PPO rollouts. Begin on 2B (fast iteration), upgrade to 4B only after the full
  IL->PPO->eval loop is green. `hidden_size` is read from the model config and
  `resampler.input_dim` derives from it, so upgrading is a config swap + retrain,
  not a code rewrite (T-009 + its "2B -> 4B" note). See `thesis_gap_analysis.md` G1.
- Consider **offline feature caching**: run the frozen VLM outside the loop and
  precompute query-resampled tokens to disk for IL/PPO to read (aligns with the
  async slow-fast cache thesis; reduces GPU contention).
- Query resampler integration.
- Async token cache integration + real slow-fast decoupled execution.
- Latency measurement (p50/p95).

### M5 - IL Warm-Start

- Dataset loader.
- Behavior cloning training loop.
- Policy checkpoint save/load.
- Closed-loop IL smoke evaluation.

### M6 - Bounded Residual PPO

- IL base policy loading.
- Residual action wrapper.
- Reward shaping hooks.
- PPO training loop and checkpointing.

### M7 - Benchmark Evaluation

- Bench2Drive/Leaderboard adapters.
- Route subset evaluation first.
- Full benchmark later.
- Metrics aggregation and failure-case reports.

## Thesis-Coverage Additions (added 2026-07-20)

These milestones close gaps between the fixed thesis idea / research matrix and the
original M1-M7 plan. Source: `notes/thesis_gap_analysis.md`. They are inserted into
the dependency order below, not appended as afterthoughts.

### M-mem - Temporal Memory Module (gap G2)
- Add a temporal/recurrent component to the fast policy with a config switch:
  {none, GRU, LSTM, temporal transformer}.
- Required by the "no temporal memory vs GRU vs LSTM vs temporal transformer"
  ablation axis. Currently `FastPolicy` is feedforward only.
- Depends on M4 (token provider); feeds M5 (IL) and M6 (PPO).

### M-data - Dataset Acquisition and Rationale Labels (gaps G3, G5)
- Scale up expert/autopilot demonstration collection for IL (build on M3 logger).
- Define the rationale label schema (meaning of the 4 risk classes / 6 meta-actions)
  and a label source: CoVLA acquisition and/or VLM auto-labeling of CARLA rollouts.
- Prerequisite for supervising the structured rationale head and for the
  "CoVLA only vs CARLA only vs mixed" comparison.
- Gated by the Scope Guard: do not start large dataset/model downloads until
  M4 (smaller VLM) is chosen and offline-feature-caching design is decided.

### M-rep - Representation Variants (gap G4)
- Implement the alternative representation paths for the ablation axis
  "generated text vs hidden state vs pooled embedding vs query-resampled".
- Only the query-resampled/pooled path exists today.

### M-eval - Evaluation Harness and Metrics (gap G7)
- Bench2Drive/Leaderboard adapter wiring the fast policy into scored routes.
- Metrics: Route Completion, Driving Score, Infraction Score, collision breakdown,
  red-light/stop-sign/lane violations, comfort/jerk, p50/p95 latency, sample
  efficiency, failure-case videos.
- Merges with / expands M7.

### M-exp - Experiment and Statistics Infrastructure (gap G6)
- Config system for the ~12-axis ablation grid (extend `configs/`).
- Multi-seed sweep runner (>=3, target 5), results aggregation, bootstrap CI,
  paired-route comparison, mean+-std reporting.
- Cross-cutting; needed before any thesis result table is produced.

### Revised High-Level Order
M1-M3 (done) -> M4 (VLM provider, smaller VLM) -> M-mem -> M5 (IL) ->
M-data (in parallel, gated) -> M6 (residual PPO) -> M-rep -> M-eval/M7 ->
M-exp (spans M5 onward). Staleness augmentation (gap G8) lands with M5/M6 training.

## Scope Guard

Do not start large model downloads, dataset downloads, or full PPO training until M1-M3 are stable. The thesis core remains fixed; new ideas belong in `notes/future_paper_ideas.md` unless they directly support these milestones.

## Progress Log

- 2026-07-20: Reproducibility baseline T-001..T-004 completed: git repository initialized, `.gitignore` added, editable package metadata added, micromamba lock exports generated, editable package leakage removed from locks, and pytest unit suite added.
- 2026-07-20: Documentation baseline T-005 completed: `docs/RUNBOOK.md` now documents CARLA server -> rollout smoke -> output inspection flow and the `run_smoke_tests.sh` `|| true` caveat.
- 2026-07-20: M1 CARLA closed-loop smoke implemented and verified. `make carla-rollout-smoke` produced 80 metadata rows and 16 camera frames under `results/smoke_rollout/`.
- 2026-07-20: M2 observation/action interface implemented and verified. Rollout metadata now includes stable ego, route, normalized action, control, camera, and termination schemas.
- 2026-07-20: M3 dataset logger/manifest implemented and verified. Autopilot expert dataset smoke produced 100 records and 20 frames; validator passes.
