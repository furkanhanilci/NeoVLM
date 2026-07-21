# Current Execution Plan

Last updated: 2026-07-21

## Current State (2026-07-21)

The local research environment is ready and the **full perception→policy→IL→live
closed-loop pipeline is implemented and verified**. Done: M1-M3 (rollout/obs/logger),
M4 (frozen Qwen3-VL-2B provider + offline feature cache), M5 IL (BC train + BC agent +
two-process live bridge — learned policy drove closed-loop in CARLA), and M-data scale
(T-016..T-019: 50 episode / 2000 frame dataset, collision-free per-episode cache,
BC train/val with **held-out val loss 0.0420→0.0100 at scale**, ~4.2x improvement).

**The current missing layer is thesis-grade closed-loop EVALUATION.** We measure val
loss (open-loop imitation error), but the thesis claim is judged by closed-loop driving
metrics (Route Completion, Driving Score, Infraction breakdown — gap G7). There is no
scored-eval harness yet; `carla/metrics.py` only computes basic rollout aggregates
(distance, collision count, mean speed). The thesis novelty components are also still
absent/stub: temporal memory (G2), rationale label source (G3), real async +
staleness augmentation (G8), residual PPO loop, VLM reward integration.

## Immediate Objective (revised 2026-07-21)

Build the **closed-loop evaluation harness and thesis metrics (M-eval / G7) NEXT**,
pulled ahead of M-mem/M6. Rationale: no component (memory, PPO, representation) can be
shown to help without scored closed-loop measurement; val loss is not the thesis
metric. M-eval converts the existing IL policy into the first genuine thesis result
(a Driving Score) and becomes the measuring stick for every later ablation.

## Historical Objective (M1, completed 2026-07-20)

The first pipeline built the runnable closed-loop CARLA smoke before adding Qwen, IL,
PPO, or Bench2Drive evaluation.

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
- Current scale decision (2026-07-21): after T-017's train/val gap, M-data scale
  is the next large direction. T-018 is CARLA-less preparation; T-019 is the live
  run. First target is `50 x 200 / save_every=5` (~2,000 saved frames, ~2.85 GiB
  feature cache), with a local first-pass ceiling of 5,000 saved frames (~7.1 GiB).
- Run `make dataset-stats` before feature-cache generation; QA reports action
  nonzero rate, double-pedal count, route/weather/town coverage, and train/val
  balance. It is a report gate, not an automatic reject gate.
- G3 rationale label source remains open; do not mix rationale-label scope into
  T-018/T-019 expert data scale unless that design is explicitly decided.

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

### Revised High-Level Order (re-ordered 2026-07-21)
M1-M3 (done) -> M4 (done: T-008..T-011) -> **M5 IL (done)** ->
**M-data scale (done: T-016..T-019)** -> **M-eval/G7 (NEXT: T-020 CARLA-free
scoring, T-021 live scored run)** -> M-mem (temporal memory ablation) -> M6
(residual PPO) -> M-rep -> M-exp (spans onward). Staleness augmentation (gap G8)
lands with M-mem/M6 training.

**Re-order decision (2026-07-21):** M-eval pulled BEFORE M-mem/M6. Prior order put
eval last (M-eval/M7 after M-rep). But a temporal-memory or PPO gain is unmeasurable
without a scored closed-loop harness, and val loss ≠ the thesis metric. Building the
Driving-Score/Route-Completion/Infraction harness now (a) yields the first real thesis
result from the existing IL policy, and (b) becomes the measuring stick every later
ablation requires. M-eval merges with/expands the original M7.

Reorder note (2026-07-20): M5 moved before M-mem. A memory module cannot be
validated without a training loop, so build the feedforward IL baseline first,
then add {GRU/LSTM/temporal transformer} as ablations on top. First IL run is a
loop-smoke on the small M3 dataset + T-011 cache (proves the loop, not driving
quality); a real IL result needs M-data.

## Scope Guard

Do not start large model downloads, dataset downloads, or full PPO training until M1-M3 are stable. The thesis core remains fixed; new ideas belong in `notes/future_paper_ideas.md` unless they directly support these milestones.

## Progress Log

- 2026-07-20: Reproducibility baseline T-001..T-004 completed: git repository initialized, `.gitignore` added, editable package metadata added, micromamba lock exports generated, editable package leakage removed from locks, and pytest unit suite added.
- 2026-07-20: Documentation baseline T-005 completed: `docs/RUNBOOK.md` now documents CARLA server -> rollout smoke -> output inspection flow and the `run_smoke_tests.sh` `|| true` caveat.
- 2026-07-20: M1 CARLA closed-loop smoke implemented and verified. `make carla-rollout-smoke` produced 80 metadata rows and 16 camera frames under `results/smoke_rollout/`.
- 2026-07-20: M2 observation/action interface implemented and verified. Rollout metadata now includes stable ego, route, normalized action, control, camera, and termination schemas.
- 2026-07-20: M3 dataset logger/manifest implemented and verified. Autopilot expert dataset smoke produced 100 records and 20 frames; validator passes.
- 2026-07-20: M4 done (T-008..T-011). Dummy token provider → frozen Qwen3-VL-2B (peak VRAM 4.17 GB, p50 98 ms/forward) → query resampler → offline hidden-state feature cache (numeric cache-vs-live equivalence).
- 2026-07-21: M5 IL done (T-012..T-015). Dataset loader + 64-dim obs, BC train loop (loss 0.316→0.0006 smoke), BCAgent, and two-process CARLA↔VLM policy-server bridge — learned policy drove closed-loop live. Architectural finding: CARLA (py3.7/8) + modern VLM (py3.9+) cannot share a process; cache mode cannot drive true closed-loop → bridge required.
- 2026-07-21: M-data done (T-016..T-019). control→action unified (throttle-brake), episode-level split, multi-episode collision-free per-episode feature cache, dataset-stats QA tool, feature-cache LRU. Live scale run: 50 episode / 2000 frame (Town10HD, autopilot), **held-out val loss 0.0420→0.0100 (~4.2x) — data scale confirmed as the lever**. Pipeline is end-to-end reproducible.
- 2026-07-21: Re-plan. Thesis-readiness review found the research is at proof-of-pipeline scale, not thesis-evidence scale: the thesis claim needs closed-loop scored metrics (G7), which do not exist. **M-eval (G7) pulled ahead of M-mem/M6** as the next milestone (T-020 CARLA-free scoring layer, T-021 live scored run).
