import pytest


@pytest.mark.carla
def test_carla_rollout_import_is_marked():
    from vlm_driving.carla.rollout import RolloutConfig

    assert RolloutConfig().port == 2000
