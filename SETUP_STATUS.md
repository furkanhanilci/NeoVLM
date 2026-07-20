# Setup Status

Last updated: 2026-07-20

## Host

- OS: Ubuntu 24.04.4 LTS
- Kernel: 6.17.0-35-generic
- GPU: NVIDIA GeForce RTX 5060, 8 GB VRAM
- NVIDIA driver: 595.71.05
- NVIDIA reported CUDA: 13.2
- RAM: 31 GiB
- Root disk free at start: about 212 GiB

## Decisions

- CARLA compatibility is treated as an isolated environment problem. Official CARLA package docs target Ubuntu 20.04/22.04, while this host is Ubuntu 24.04.
- Bench2Drive expects CARLA 0.9.15 and Python 3.7/3.8, so the `carla` environment uses Python 3.8.
- VLM work is separated into `vlm` to avoid CARLA dependency pin conflicts.
- Analysis and thesis tooling are separated into `analysis`.

## Installed Without Sudo

- Project folder structure.
- Micromamba at `~/bin/micromamba`.
- Conda environments:
  - `carla` with Python 3.8, CARLA PythonAPI, Leaderboard/ScenarioRunner dependencies.
  - `vlm` with PyTorch CUDA, Transformers, Qwen utilities, RL packages, W&B/MLflow.
  - `analysis` with JupyterLab, Quarto, Pandoc, Typst, plotting/statistics packages, uv, pipx.
- CARLA 0.9.15 Linux package under `third_party/CARLA_0.9.15`.
- CARLA PythonAPI `agents` path available from `third_party/CARLA_0.9.15/PythonAPI/carla`.
- Repositories:
  - `third_party/leaderboard` on `leaderboard-2.1`.
  - `third_party/scenario_runner` on `leaderboard-2.1`.
  - `third_party/Bench2Drive`.
- Thesis starter files.

## Verified

- Initial research code scaffold under `src/vlm_driving` is in place.
- `make check-research` passes on CPU in the `vlm` environment.
- `nvidia-smi` works.
- `torch.cuda.is_available()` is `True` in `vlm`.
- Torch CUDA tensor smoke test succeeded on NVIDIA GeForce RTX 5060.
- CARLA PythonAPI import works.
- CARLA server started once with `-RenderOffScreen -nosound`.
- CARLA client connected to `Carla/Maps/Town10HD_Opt`.
- Minimal ego vehicle spawn/destroy smoke test succeeded.
- Leaderboard 2.1 imports work with CARLA PythonAPI path.
- ScenarioRunner imports work with CARLA PythonAPI path.
- Bench2Drive leaderboard/scenario_runner imports work with CARLA PythonAPI path.
- Qwen3-VL config and processor load from `Qwen/Qwen3-VL-8B-Instruct` without model weights.
- Quarto rendered thesis DOCX and Typst-based PDF.
- `check_gpu.sh`, `check_torch.sh`, `check_bench2drive.sh`, and `check_carla.sh` passed when CARLA server was running.
- Docker service is active.
- Docker Compose is installed.
- NVIDIA Container Toolkit is installed; Docker reports `nvidia` runtime and CDI GPU devices.
- Docker GPU smoke test passed with `nvidia/cuda:13.0.2-base-ubuntu22.04`.
- VS Code CLI `code` is installed.
- Zotero is installed under `/opt/zotero` with `/usr/local/bin/zotero`.
- System XeLaTeX PDF rendering passed after setting Quarto `pdf-engine: xelatex`.
- M1 CARLA closed-loop smoke is implemented and verified: `make carla-rollout-smoke` produced `results/smoke_rollout/metadata.jsonl`, `manifest.json`, and RGB frames.
- M2 observation/action interface is implemented and verified: rollout metadata includes stable ego, route, normalized action, CARLA control, camera, termination, policy, event, and optional expert control schemas.
- M3 dataset logger/manifest path is implemented and verified: IL dataset smoke produced `results/datasets/carla_il_smoke/episode_000` and `make validate-carla-dataset` passed.
- Reproducibility baseline tasks T-001..T-005 are complete: git repository initialized, `.gitignore` added, `pyproject.toml` packaging added, `environments/*.lock.yml` exported without local `prefix:`, pytest unit tests added, and `docs/RUNBOOK.md` created.
- Pytest unit suite passes in `vlm`: 12 CARLA-free tests pass by default; CARLA-dependent tests are marked with `@pytest.mark.carla` and excluded from the default run.

## Third-Party Version Pins

The `third_party/` repositories are intentionally ignored by the project git repository, so these commit hashes are the reproducibility reference for the current local checkout:

| Component | Branch/tag/source | Local revision |
|-----------|-------------------|----------------|
| `third_party/leaderboard` | `leaderboard-2.1` | `cfecdc8` |
| `third_party/scenario_runner` | `leaderboard-2.1` | `d7bcaf0` |
| `third_party/Bench2Drive` | `main` | `2645714` |
| `third_party/CARLA_0.9.15` | CARLA 0.9.15 Linux package | downloaded asset, not a git checkout |

## Blocked

- None for the core local environment.

## Remaining Research Assets

- `datasets/` has no full benchmark dataset assets. Bench2Drive route/config files are present, but full benchmark datasets/assets are not downloaded.
- `checkpoints/` is currently empty. Qwen3-VL config/processor loading is verified, but full model weights are not downloaded.
- `results/` contains smoke outputs (`results/smoke_rollout/` and dataset smoke output), but no training runs, benchmark evaluations, or thesis result tables exist yet.
- M4 has not started. Per the current plan, choose a smaller Qwen3-VL 2B/4B-class model or an offline feature-caching path before downloading large model weights.
- `vlm` OpenCV import issue was fixed by pinning `libjxl=0.11.*` in `environments/vlm.yml`.

## Needs User Token/Login Later

- Hugging Face token is optional for public Qwen3-VL config/processor but recommended for rate limits and full model downloads.
- W&B login is needed only if W&B tracking is used.
- Dataset license/Bench2Drive asset access may require user acceptance before downloading benchmark datasets.

## Logs

Setup logs are under `setup_logs/`.
