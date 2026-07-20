# Thesis Coverage Gap Analysis

Last updated: 2026-07-20

Purpose: map the fixed thesis core (`notes/thesis_core_idea.md`) and the research
matrix (`README.md`) against what actually exists in `src/` and what the execution
plan (`notes/current_execution_plan.md`) covers, then list the parts that are
missing or under-specified. This is a planning/review artifact, not a scope
change. The thesis core idea remains fixed.

## Status Legend

- DONE: implemented and verified.
- SCAFFOLD: module/interface exists, internals or integration incomplete.
- PLANNED: covered by an existing milestone, not yet built.
- MISSING: required by the thesis idea/research matrix but absent from both code
  and milestones.

## Fixed Component Mapping

| Thesis component | Code artifact | Status | Notes |
|---|---|---|---|
| Frozen Qwen3-VL backbone | (none) | PLANNED (M4) | Only config/processor load verified; no module runs the VLM to produce hidden states. |
| Query resampler | `models/query_resampler.py` | DONE | Cross-attention resampler; unit-tested. |
| Structured rationale head | `models/rationale_head.py` | SCAFFOLD | Predicts risk(4)/meta-action(6); **no ground-truth label source defined**. |
| Async slow-fast token cache | `cache.py` | SCAFFOLD | Cache primitive + age/freshness only; **no real async execution loop** (thread/process, slow-fast decoupling). |
| Staleness-aware fast policy | `models/policy.py` | SCAFFOLD | Consumes `token_age_s`; **no staleness augmentation at train time**; also feedforward only (see M-mem gap). |
| Imitation learning warm-start | (none) | PLANNED (M5) | No dataset loader / BC loop / expert data. |
| Bounded residual PPO | `models/policy.py` (ResidualActionHead + value) | SCAFFOLD | Residual + value heads exist; **no PPO loop / rollout buffer / advantage / env-reward wiring**. |
| VLM-guided reward shaping | `reward.py` | SCAFFOLD | `shaped_reward()` exists; depends on VLM rationale + `meta_action_match` reference (undefined source); not integrated. |
| CARLA/Leaderboard/Bench2Drive eval | `carla/rollout.py` (smoke) | PLANNED (M7) | Closed-loop smoke works (M1-M3); no eval adapter or metrics. |

## Gaps Required By The Thesis Idea But Not In Any Milestone

Ranked by impact on thesis feasibility.

### G1 — Hardware / VRAM feasibility — MEASURED / DE-RISKED (2026-07-20)

- **Measured (T-010, verified twice on RTX 5060 8 GB):** frozen Qwen3-VL-2B in
  bf16 → **peak VRAM 4.17 GB**, single forward p50 98 ms / p95 588 ms, real hidden
  states `[1, 366, 2048]`. No quantization needed; ~3 GB headroom for CARLA + policy.
- **Implication:** 2B is comfortably feasible on this hardware. The ~100 ms VLM
  forward also empirically motivates the thesis core: the VLM cannot run at control
  frequency (~20 Hz), so slow-fast async decoupling + staleness-aware token caching
  is not just an optimization but a necessity. This is thesis-relevant evidence.
- 4B outlook: 4B bf16 (~8-9 GB) will not fit; 4-bit 4B (~5-6 GB) may, tightly. The
  config-swap upgrade path (T-009) plus offline feature caching keeps 4B reachable.
- Original analysis retained below for context.

### G1 (original) — Hardware / VRAM feasibility
- Host GPU is RTX 5060, **8 GB VRAM** (`SETUP_STATUS.md`). Qwen3-VL-8B weights in
  fp16 are ~16 GB; even 4-bit (~5-6 GB) plus vision encoder, KV cache, the trained
  resampler/policy/rationale heads, and CARLA render + PPO rollouts on the same GPU
  is not viable.
- **Decision (2026-07-20, user):** move to a **smaller VLM (Qwen3-VL-2B/4B class)**.
  `configs/baseline.yaml` and `config.py` currently pin `Qwen/Qwen3-VL-8B-Instruct`
  and must be updated when M4 starts.
- Strongly compatible mitigation regardless of size: **offline feature caching** —
  run the frozen VLM outside the control loop, precompute query-resampled tokens to
  disk, and have IL/PPO read them. This aligns with the async slow-fast cache thesis
  and removes the VLM from the GPU-contended runtime loop. Keep as an option even
  with the smaller model.

### G2 (critical) — Temporal memory module absent
- Research matrix lists "no temporal memory vs GRU vs LSTM vs temporal transformer".
- `FastPolicy` is fully feedforward; there is **no recurrent/temporal component**.
- This is a named ablation axis with zero implementation. Needs a memory module and
  a config switch selecting {none, GRU, LSTM, temporal transformer}.

### G3 (critical) — Rationale label source undefined
- The rationale head predicts risk/meta-action classes but there is **no source of
  ground-truth labels** to supervise them.
- Research matrix implies CoVLA ("CoVLA only vs CARLA only vs mixed fine-tuning") but
  there is no CoVLA acquisition/loader, no VLM auto-labeling pipeline, and no defined
  label schema mapping (what the 4 risk classes / 6 meta-actions mean).

### G4 (high) — Representation axis "generated text vs hidden state vs pooled"
- Only the pooled/query-resampled path exists. The **generated-text rationale path**
  (decoding text from the VLM) and the raw-hidden-state path are not implemented.

### G5 (high) — Dataset acquisition
- `datasets/` is empty. Two data needs are unaddressed by milestones:
  1. IL expert demonstrations at scale (CARLA autopilot/expert rollouts, or
     Bench2Drive data) — M3 logger exists, but no large-scale collection run.
  2. Rationale-labeled data (CoVLA or VLM-labeled CARLA) for G3.

### G6 (high) — Experiment / seed / ablation infrastructure
- Thesis promises >=3-5 seeds, bootstrap CI, paired-route comparison, mean+-std.
- Only a single `configs/baseline.yaml` exists. No seed-sweep runner, no config
  system for the ~12-axis ablation grid, no results-aggregation/statistics code.

### G7 (high) — Evaluation harness and metrics
- No implementation for Route Completion, Driving Score, Infraction Score, collision
  breakdown, red-light/stop-sign/lane violations, comfort/jerk, p50/p95 latency,
  sample efficiency, failure-case videos, bootstrap CI, paired comparison.
- Leaderboard/Bench2Drive import but no adapter wires them to the fast policy.

### G8 (medium) — Real async execution and staleness augmentation
- The cache is a synchronous container. The thesis "asynchronous" claim needs an
  actual slow(VLM)/fast(control) decoupled execution path with measured latency, and
  training-time token-age augmentation ("no staleness augmentation vs staleness
  augmentation" axis).

## What Is Solid

- Module interfaces map cleanly onto the fixed thesis architecture
  (resampler -> cache -> staleness-aware policy -> bounded residual -> reward).
- M1-M3 closed-loop CARLA rollout + observation/action schema + dataset logger are
  implemented and verified.
- Reproducibility/packaging/test infrastructure (git, pyproject, env locks, pytest)
  is in place and reviewed (`AI_TASKS.md` T-001..T-004).

## Recommended Folding Into The Plan

These gaps are mapped into `notes/current_execution_plan.md` as milestone updates
and new milestones (M-mem, M-data, M-eval, M-exp) added 2026-07-20. No `AI_TASKS.md`
(Codex) tasks are opened from this document yet; infrastructure tasks T-005..T-007
continue first.
