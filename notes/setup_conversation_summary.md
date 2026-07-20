# Setup Conversation Summary

Date: 2026-07-03
Machine: native Ubuntu 24.04.4 LTS, Europe/Istanbul timezone
Project root: `/home/mehmet-g-rkem/research/vlm-driving-thesis`

## Goal

Prepare a full research/thesis environment for closed-loop autonomous driving research:

Camera + navigation command -> frozen Qwen3-VL -> query resampler -> structured driving rationale head -> async token cache -> staleness-aware fast policy -> IL policy -> bounded residual PPO -> VLM-guided reward -> CARLA / Leaderboard / Bench2Drive closed-loop evaluation.

The user asked not to reduce the benchmark/test/ablation scope and to log all setup decisions under `setup_logs/`.

## Important Security/Access Notes

- The user configured passwordless sudo during setup so system-level installation could proceed.
- Do not store or reuse any password from chat.
- Hugging Face token is not configured.
- W&B login is not configured.
- Bench2Drive/dataset license acceptance or external login may still be needed.

## Created Project Layout

Root: `/home/mehmet-g-rkem/research/vlm-driving-thesis`

Key folders:

- `thesis/`
- `thesis/chapters/`
- `references/`
- `papers/`
- `notes/`
- `figures/`
- `experiments/`
- `src/`
- `configs/`
- `results/`
- `logs/`
- `setup_logs/`
- `scripts/`
- `datasets/`
- `checkpoints/`
- `third_party/`

## Host Findings

Initial system inventory:

- OS: Ubuntu 24.04.4 LTS
- Kernel: 6.17.0-35-generic
- GPU: NVIDIA GeForce RTX 5060, 8 GB VRAM
- NVIDIA driver: 595.71.05
- NVIDIA-reported CUDA: 13.2
- RAM: 31 GiB
- Disk after setup: about 164 GiB free on `/`

## Installed System Packages

Installed via apt after passwordless sudo was configured:

- build tools: `build-essential`, `cmake`, `ninja-build`, `pkg-config`
- git tools: `git`, `git-lfs`
- utilities: `curl`, `wget`, `unzip`, `p7zip-full`, `bzip2`, `htop`, `nvtop`, `tmux`, `tree`, `xclip`
- media/graphics: `ffmpeg`, `mesa-utils`, `vulkan-tools`, `graphviz`
- Docker: `docker.io`, `docker-compose-v2`
- Python helpers: `python3-venv`, `python3-pip`, `pipx`
- writing tools: `pandoc`, `texlive-xetex`, `texlive-fonts-recommended`, `texlive-latex-extra`, `latexmk`, `libreoffice`
- VS Code CLI from Microsoft apt repo: `code`
- Zotero installed under `/opt/zotero`, symlinked at `/usr/local/bin/zotero`

## NVIDIA / Docker Status

Installed NVIDIA Container Toolkit from NVIDIA repo:

- `nvidia-container-cli` version: 1.19.1
- Docker reports `nvidia` runtime and CDI GPU devices.
- Docker GPU smoke test passed with:

```bash
sudo docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu22.04 nvidia-smi
```

The user was added to the `docker` group. A new login/terminal may be needed for sudo-less docker access. The project script falls back to `sudo docker` when necessary.

## Micromamba / Conda Environments

Micromamba installed at:

```bash
/home/mehmet-g-rkem/bin/micromamba
```

Environments:

- `carla`: Python 3.8, CARLA/Leaderboard/ScenarioRunner dependencies.
- `vlm`: Python 3.11, PyTorch CUDA, Transformers, Qwen utilities, RL packages, W&B/MLflow.
- `analysis`: Python 3.11, JupyterLab, Quarto, Pandoc, Typst, plotting/statistics packages, uv, pipx.

Environment files:

- `environments/carla.yml`
- `environments/vlm.yml`
- `environments/analysis.yml`

Important fix applied:

- `vlm` OpenCV initially failed with `libjxl.so.0.11` missing.
- Fixed by pinning `libjxl=0.11.*` in `environments/vlm.yml`.
- Final `vlm` import check passed: `torch`, `transformers`, `cv2`, `gymnasium`, `stable_baselines3`.

## CARLA / Leaderboard / Bench2Drive

Installed/downloaded:

- CARLA 0.9.15 Linux package under:
  `/home/mehmet-g-rkem/research/vlm-driving-thesis/third_party/CARLA_0.9.15`
- CARLA archive retained at:
  `third_party/CARLA_0.9.15.tar.gz`
- CARLA PythonAPI agents path:
  `third_party/CARLA_0.9.15/PythonAPI/carla`
- Leaderboard cloned at:
  `third_party/leaderboard`, branch `leaderboard-2.1`
- ScenarioRunner cloned at:
  `third_party/scenario_runner`, branch `leaderboard-2.1`
- Bench2Drive cloned at:
  `third_party/Bench2Drive`

Verified:

- CARLA PythonAPI import works.
- CARLA agents import works.
- Leaderboard evaluator import works with PYTHONPATH.
- ScenarioRunner import works with PYTHONPATH.
- Bench2Drive leaderboard/scenario_runner imports work.
- CARLA server started with offscreen rendering.
- CARLA client connected to `Carla/Maps/Town10HD_Opt`.
- Minimal ego vehicle spawn/destroy test passed.

## VLM / Qwen Status

Verified without downloading full model weights:

- `Qwen/Qwen3-VL-8B-Instruct` config loads.
- `Qwen3VLProcessor` loads.
- Tokenizer vocab reported.

Not yet done:

- Full Qwen3-VL weights are not downloaded.
- `checkpoints/` is empty.
- HF token is not configured.

## Thesis Writing Status

Created:

- `thesis/main.qmd`
- `thesis/references.bib`
- `thesis/chapters/01_introduction.md`
- `thesis/chapters/02_related_work.md`
- `thesis/chapters/03_method.md`
- `thesis/chapters/04_experiments.md`
- `thesis/chapters/05_results.md`
- `thesis/chapters/06_discussion.md`
- `thesis/chapters/07_conclusion.md`

Verified:

- Quarto DOCX render passed.
- Quarto PDF render passed after setting `pdf-engine: xelatex`.
- Earlier Typst PDF render also passed.

Outputs:

- `thesis/main.pdf`
- `thesis/main.docx`

## Scripts / Makefile

Created scripts:

- `scripts/check_gpu.sh`
- `scripts/check_torch.sh`
- `scripts/check_carla.sh`
- `scripts/check_bench2drive.sh`
- `scripts/check_docker_gpu.sh`
- `scripts/run_carla_server.sh`
- `scripts/run_smoke_tests.sh`
- `scripts/install_system_packages.sh`
- `scripts/install_nvidia_container_toolkit.sh`

Make targets:

```bash
make check-gpu
make check-torch
make check-docker-gpu
make check-carla
make check-bench2drive
make smoke
make carla-server
make thesis-pdf
```

Notes:

- `check_carla.sh` imports CARLA/agents and tries server connection.
- If server is not running, it prints `server unavailable` and exits cleanly by default.
- To require a live CARLA server:

```bash
REQUIRE_CARLA_SERVER=1 ./scripts/check_carla.sh
```

## Desktop Shortcuts

Desktop path: `/home/mehmet-g-rkem/Masaüstü`

Created folder:

- `/home/mehmet-g-rkem/Masaüstü/VLM-Driving-Thesis`

Inside it:

- `Proje Klasörü` symlink to project root
- `README.txt`

Created desktop launchers:

- `VS Code - Tez Projesi.desktop`
- `Terminal - Tez Klasörü.desktop`
- `CARLA Server - Tez.desktop`
- `JupyterLab - Analysis.desktop`
- `Tez Proje Klasörü.desktop`
- `Zotero.desktop`
- `LibreOffice Writer.desktop`
- `LibreOffice Draw.desktop`

GNOME may require “Allow Launching” on first launch.

## Logs

Setup logs are under:

```bash
/home/mehmet-g-rkem/research/vlm-driving-thesis/setup_logs/
```

Important log groups:

- system inventory: `00_system_info.log`
- environment creation: `05_env_carla_create.log`, `07_env_vlm_create.log`, `11_env_analysis_create.log`
- GPU/Torch/Qwen: `09_torch_cuda_test.log`, `36_qwen3vl_processor_config_test.log`
- CARLA tests: `32_carla_client_world_test.log`, `33_carla_spawn_test.log`, `91_audit_check_carla_strict.log`, `92_audit_carla_spawn_repeat.log`
- Docker GPU: `60_check_docker_gpu_cuda13_retry.log`
- final audits: `106_final_todo_smoke.log`, `107_final_todo_audit.log`, `108_final_login_audit.log`

## Current Remaining Work

These are not local setup failures; they are research asset/login tasks:

1. Hugging Face login/token:
   - Needed or recommended for full Qwen3-VL model downloads and rate limits.
   - Token currently not configured.

2. W&B login:
   - Needed only if W&B experiment tracking is used.
   - Currently not logged in.

3. Dataset/assets:
   - `datasets/` is empty.
   - Bench2Drive route/config files are present in the repo, but full dataset/benchmark assets are not downloaded.
   - Dataset license/terms may require user acceptance.

4. Checkpoints:
   - `checkpoints/` is empty.
   - Qwen3-VL full model weights are not downloaded.

5. Research implementation:
   - Architecture code for frozen Qwen3-VL, query resampler, structured rationale head, async token cache, staleness-aware fast policy, IL warm-start, bounded residual PPO, VLM reward shaping still needs to be implemented under `src/`, `configs/`, `experiments/`.

## How To Resume In A New Chat

Tell the next Codex agent:

> Continue from `/home/mehmet-g-rkem/research/vlm-driving-thesis/notes/setup_conversation_summary.md`. First read that file and `SETUP_STATUS.md`, then continue with the next research setup step. Do not redo completed installation unless a check fails.

Recommended first commands in the new session:

```bash
cd ~/research/vlm-driving-thesis
make smoke
sed -n '1,220p' notes/setup_conversation_summary.md
sed -n '1,180p' SETUP_STATUS.md
```

If starting live CARLA work:

```bash
make carla-server
# In another terminal:
cd ~/research/vlm-driving-thesis
REQUIRE_CARLA_SERVER=1 ./scripts/check_carla.sh
```

