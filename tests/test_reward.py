import torch

from vlm_driving.reward import DrivingEvents, shaped_reward


def test_shaped_reward_combines_progress_penalties_and_meta_bonus():
    events = DrivingEvents(
        progress=torch.tensor([1.0]),
        collision=torch.tensor([0.0]),
        lane_violation=torch.tensor([1.0]),
        red_light=torch.tensor([0.0]),
    )
    risk_logits = torch.tensor([[0.0, 0.0, 0.0]])

    reward = shaped_reward(
        events=events,
        risk_logits=risk_logits,
        meta_action_match=torch.tensor([1.0]),
        progress_weight=2.0,
        collision_penalty=5.0,
        lane_penalty=0.5,
        red_light_penalty=2.0,
        risk_weight=0.2,
        meta_action_weight=0.3,
    )

    assert torch.allclose(reward, torch.tensor([1.7]))


def test_shaped_reward_penalizes_collision_and_high_risk():
    events = DrivingEvents(
        progress=torch.tensor([0.0]),
        collision=torch.tensor([1.0]),
        lane_violation=torch.tensor([0.0]),
        red_light=torch.tensor([1.0]),
    )
    risk_logits = torch.tensor([[-20.0, -20.0, 20.0]])

    reward = shaped_reward(
        events=events,
        risk_logits=risk_logits,
        meta_action_match=torch.tensor([0.0]),
        progress_weight=1.0,
        collision_penalty=5.0,
        lane_penalty=1.0,
        red_light_penalty=2.0,
        risk_weight=0.5,
        meta_action_weight=0.0,
    )

    assert reward.item() < -7.49
