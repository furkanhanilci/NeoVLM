from vlm_driving.config import ExperimentConfig, ResamplerConfig, VLMConfig


def test_experiment_config_uses_qwen3_vl_2b_hidden_size():
    config = ExperimentConfig()

    assert config.vlm.model_id == "Qwen/Qwen3-VL-2B-Instruct"
    assert config.vlm.hidden_size == 2048
    assert config.resampler.input_dim == config.vlm.hidden_size
    assert config.vlm.token_dim == 512
    assert config.policy.token_dim == 512


def test_experiment_config_forces_resampler_input_dim_from_vlm_hidden_size():
    config = ExperimentConfig(
        vlm=VLMConfig(hidden_size=1234),
        resampler=ResamplerConfig(input_dim=999),
    )

    assert config.resampler.input_dim == 1234
