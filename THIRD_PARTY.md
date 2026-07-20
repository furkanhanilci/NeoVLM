# Third-Party Notices

This repository's original code is licensed under the MIT License. Third-party simulators, benchmark tools, datasets, model weights, and generated assets remain under their own licenses and terms. This notice is a project-level index, not a replacement for the upstream license files.

## Local Third-Party Checkouts

| Component | Local path | Version/source | License/terms observed locally | Notes |
|-----------|------------|----------------|--------------------------------|-------|
| CARLA 0.9.15 | `third_party/CARLA_0.9.15` | CARLA 0.9.15 Linux package | MIT License (`third_party/CARLA_0.9.15/LICENSE`) | Includes CARLA simulator binaries and PythonAPI assets. |
| CARLA Leaderboard | `third_party/leaderboard` | `leaderboard-2.1` @ `cfecdc8` | MIT License (`third_party/leaderboard/LICENSE`) | Used for benchmark/evaluation integration. |
| ScenarioRunner | `third_party/scenario_runner` | `leaderboard-2.1` @ `d7bcaf0` | MIT License (`third_party/scenario_runner/LICENSE`) | Used by CARLA/Leaderboard scenarios. |
| Bench2Drive | `third_party/Bench2Drive` | `main` @ `2645714` | CC-BY-NC-ND 4.0 (`third_party/Bench2Drive/LICENSE`) unless specified otherwise by upstream | README states assets and code are under CC-BY-NC-ND unless specified otherwise. This is not MIT-compatible for unrestricted redistribution; keep upstream terms attached. |

## Model And Dataset Dependencies

| Component | Source | License/terms note | Project status |
|-----------|--------|--------------------|----------------|
| Qwen3-VL | `Qwen/Qwen3-VL-8B-Instruct` model card | Apache-2.0 is shown on the current Hugging Face model card (checked 2026-07-20). Verify the exact model card/license again before downloading or redistributing weights. | Config/processor loading was tested earlier; full weights are not downloaded into this repo. |
| Bench2Drive datasets/assets | Upstream Bench2Drive distribution | Dataset access may require user acceptance and may be subject to non-commercial/no-derivatives terms. | Full benchmark datasets/assets are not downloaded. |
| Hugging Face model cache | User-local cache, outside repo | Governed by each model/dataset card. | Not committed. |

## Attribution Guidance

- Preserve upstream `LICENSE`, `NOTICE`, citation, and model/dataset card files when distributing any third-party component or derivative artifact.
- Cite CARLA, Leaderboard/ScenarioRunner, Bench2Drive, and Qwen/Qwen3-VL in thesis text and papers when their code, assets, model outputs, or benchmark protocols are used.
- Do not assume this repository's MIT license grants rights to redistribute third-party datasets, model weights, simulator binaries, or benchmark assets.
- Keep large third-party assets out of git; document reproducible version references in `SETUP_STATUS.md` and environment lock files.

## Local License Files Checked

- `third_party/CARLA_0.9.15/LICENSE`
- `third_party/leaderboard/LICENSE`
- `third_party/scenario_runner/LICENSE`
- `third_party/Bench2Drive/LICENSE`
- `third_party/Bench2Drive/README.md`
