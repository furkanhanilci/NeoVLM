#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

micromamba run -n vlm python - <<'PY'
import torch

from vlm_driving.cache import AsyncTokenCache
from vlm_driving.config import ExperimentConfig
from vlm_driving.models import FastPolicy, QueryResampler, StructuredRationaleHead
from vlm_driving.reward import DrivingEvents, shaped_reward

cfg = ExperimentConfig()
batch = 2

resampler = QueryResampler(
    input_dim=cfg.resampler.input_dim,
    output_dim=cfg.resampler.output_dim,
    num_queries=cfg.resampler.num_queries,
    num_heads=cfg.resampler.num_heads,
    dropout=cfg.resampler.dropout,
)
vlm_tokens = torch.randn(batch, 17, cfg.resampler.input_dim)
compact = resampler(vlm_tokens)
assert compact.shape == (batch, cfg.resampler.num_queries, cfg.resampler.output_dim)

rationale = StructuredRationaleHead(
    token_dim=cfg.rationale.token_dim,
    hidden_dim=cfg.rationale.hidden_dim,
    risk_classes=cfg.rationale.risk_classes,
    meta_actions=cfg.rationale.meta_actions,
)
rationale_out = rationale(compact)
assert rationale_out.risk_logits.shape == (batch, cfg.rationale.risk_classes)
assert rationale_out.meta_action_logits.shape == (batch, cfg.rationale.meta_actions)

cache = AsyncTokenCache(max_age_s=cfg.policy.max_token_age_s)
cache.update(compact, timestamp_s=10.0)
cached, age_s, fresh = cache.read(now_s=10.2)
assert cached is not None and fresh and abs(age_s - 0.2) < 1e-6

policy = FastPolicy(
    obs_dim=cfg.policy.obs_dim,
    token_dim=cfg.policy.token_dim,
    hidden_dim=cfg.policy.hidden_dim,
    action_dim=cfg.policy.action_dim,
    residual_limits=cfg.policy.residual_limit,
)
policy_out = policy(
    observation=torch.randn(batch, cfg.policy.obs_dim),
    compact_tokens=cached,
    token_age_s=torch.full((batch,), age_s),
)
assert policy_out["action"].shape == (batch, cfg.policy.action_dim)
assert policy_out["value"].shape == (batch,)

events = DrivingEvents(
    progress=torch.ones(batch),
    collision=torch.zeros(batch),
    lane_violation=torch.zeros(batch),
    red_light=torch.zeros(batch),
)
reward = shaped_reward(
    events=events,
    risk_logits=rationale_out.risk_logits,
    meta_action_match=torch.ones(batch),
    progress_weight=cfg.reward.progress_weight,
    collision_penalty=cfg.reward.collision_penalty,
    lane_penalty=cfg.reward.lane_penalty,
    red_light_penalty=cfg.reward.red_light_penalty,
    risk_weight=cfg.reward.risk_weight,
    meta_action_weight=cfg.reward.meta_action_weight,
)
assert reward.shape == (batch,)
print("research code smoke ok")
PY
