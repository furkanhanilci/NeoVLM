# Fixed Doctoral Thesis Core Idea

Last updated: 2026-07-20

## Core Thesis Direction

This doctoral thesis keeps the original research idea fixed:

Camera and navigation command inputs are processed by a frozen Qwen3-VL-style vision-language model. A query resampler compresses VLM representations into compact driving tokens. A structured driving rationale head predicts safety- and maneuver-relevant semantic signals. These slow VLM-derived tokens are cached asynchronously and consumed by a staleness-aware fast policy. The fast policy is first warm-started with imitation learning, then improved through bounded residual PPO. VLM-guided reward shaping is used to bias reinforcement learning toward safer and semantically consistent driving. The complete system is evaluated in closed loop with CARLA, Leaderboard, and Bench2Drive.

## Fixed Main Components

- Frozen Qwen3-VL or equivalent large VLM backbone.
- Query resampler for compact visual-language driving tokens.
- Structured driving rationale head.
- Asynchronous slow-fast token cache.
- Staleness-aware fast driving policy.
- Imitation learning warm-start.
- Bounded residual PPO.
- VLM-guided reward shaping.
- Closed-loop CARLA / Leaderboard / Bench2Drive evaluation.

## Primary Thesis Claim

A frozen large VLM can be made practical for closed-loop autonomous driving by decoupling slow semantic reasoning from fast control through query-resampled cached tokens, explicit token staleness awareness, and bounded residual reinforcement learning.

## Scope Policy

This idea is considered fixed for the doctoral thesis. New literature trends and extra technical ideas should be treated as future-paper directions unless they directly support the above architecture, experiments, or evaluation.
