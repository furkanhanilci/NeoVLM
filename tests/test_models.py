import pytest
import torch

from vlm_driving.models import FastPolicy, QueryResampler, StructuredRationaleHead
from vlm_driving.models.policy import ResidualActionHead


def test_query_resampler_output_shape_and_input_validation():
    resampler = QueryResampler(input_dim=16, output_dim=8, num_queries=4, num_heads=2)

    output = resampler(torch.randn(3, 5, 16))

    assert output.shape == (3, 4, 8)
    with pytest.raises(ValueError, match="tokens must have shape"):
        resampler(torch.randn(5, 16))


def test_structured_rationale_head_shapes_and_input_validation():
    head = StructuredRationaleHead(token_dim=8, hidden_dim=6, risk_classes=3, meta_actions=4)

    output = head(torch.randn(2, 4, 8))

    assert output.risk_logits.shape == (2, 3)
    assert output.meta_action_logits.shape == (2, 4)
    assert output.pooled.shape == (2, 6)
    with pytest.raises(ValueError, match="compact_tokens must have shape"):
        head(torch.randn(4, 8))


def test_fast_policy_shapes_and_action_bounds():
    policy = FastPolicy(obs_dim=5, token_dim=8, hidden_dim=12, action_dim=3, residual_limits=(0.1, 0.2, 0.3))

    output = policy(
        observation=torch.randn(2, 5),
        compact_tokens=torch.randn(2, 4, 8),
        token_age_s=torch.tensor([0.0, 0.5]),
    )

    assert output["il_action"].shape == (2, 3)
    assert output["residual"].shape == (2, 3)
    assert output["action"].shape == (2, 3)
    assert output["value"].shape == (2,)
    assert torch.all(output["action"] <= 1.0)
    assert torch.all(output["action"] >= -1.0)


def test_residual_action_head_validates_limit_count():
    with pytest.raises(ValueError, match="limits length"):
        ResidualActionHead(input_dim=4, action_dim=3, limits=(0.1, 0.2))
