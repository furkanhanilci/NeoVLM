# VLM Driving Thesis Environment

This workspace targets closed-loop autonomous driving research with frozen Qwen3-VL, query resampling, structured driving rationales, async slow-fast token caching, staleness-aware fast policies, IL warm-start, bounded residual PPO, and VLM-guided reward shaping on CARLA / Leaderboard / Bench2Drive.

## Layout

- `thesis/`: Quarto thesis draft and chapter markdown files.
- `configs/`: experiment and environment configs.
- `scripts/`: setup and smoke-test scripts.
- `third_party/`: external repos such as CARLA tooling, ScenarioRunner, Leaderboard, Bench2Drive.
- `datasets/`: local datasets and benchmark assets.
- `checkpoints/`: model checkpoints and downloaded weights.
- `results/`, `logs/`, `setup_logs/`: experiment outputs and setup logs.

## Shell Setup

Micromamba is installed at `~/bin/micromamba`. Add it to the shell path if needed:

```bash
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
```

Create environments:

```bash
micromamba env create -f environments/carla.yml
micromamba env create -f environments/vlm.yml
micromamba env create -f environments/analysis.yml
```

For reproducible environment recreation, prefer the resolved lock exports after
the initial setup has been validated:

```bash
micromamba env create -f environments/carla.lock.yml
micromamba env create -f environments/vlm.lock.yml
micromamba env create -f environments/analysis.lock.yml
```

The lock files are generated with `micromamba env export`; build strings and pip
package versions may still be platform-specific. The machine-local `prefix:`
entry is intentionally removed.

Run checks:

```bash
make check-gpu
make check-torch
make smoke
```

Note: `make smoke` is a convenience check. `scripts/run_smoke_tests.sh` currently masks `check_bench2drive.sh` and `check_carla.sh` failures with `|| true`, so use targeted commands such as `make carla-status`, `make carla-rollout-smoke`, and `make test` for gating.

CARLA and Bench2Drive require external assets and may require license or dataset access. Put CARLA under `third_party/CARLA_0.9.15` unless a script states otherwise.


## Runbook

For the CARLA server -> rollout smoke -> output inspection flow, see [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Research Matrix

Core comparisons:

- pooled VLM vs query resampler
- hidden state vs generated text vs pooled embedding
- no temporal memory vs GRU vs LSTM vs temporal transformer
- no-CoT vs structured rationale
- risk-only vs meta-action-only vs risk+meta-action
- sync vs async
- no token age vs token age
- no staleness augmentation vs staleness augmentation
- IL-only vs PPO from scratch vs full fine-tune PPO vs residual PPO
- VLM reward off vs risk penalty vs meta-action consistency vs combined
- CoVLA only vs CARLA only vs mixed fine-tuning
- minimum 3 seeds, preferably 5

## Research Matrix To Milestones

This table maps the thesis comparison axes to the implementation milestones in `notes/current_execution_plan.md`; it documents scope only and does not change the code target.

| Research comparison axis | Milestone path | Current state |
|--------------------------|----------------|---------------|
| pooled VLM vs query resampler | M4, M-rep | Scaffolded model components; real VLM provider not started. |
| hidden state vs generated text vs pooled embedding | M-rep | Planned. |
| no temporal memory vs GRU vs LSTM vs temporal transformer | M-mem | Planned; current fast policy is feedforward. |
| no-CoT vs structured rationale | M4, M-data, M5 | Structured rationale head scaffold exists; labels/data pending. |
| risk-only vs meta-action-only vs risk+meta-action | M5, M6 | Reward/scaffold exists; training/evaluation pending. |
| sync vs async, token age, staleness augmentation | M4, M5, M6 | Async cache scaffold tested; real slow-fast loop pending. |
| IL-only vs PPO from scratch vs full fine-tune PPO vs residual PPO | M5, M6 | Planned after dataset logger and VLM provider. |
| VLM reward off vs risk penalty vs meta-action consistency vs combined | M6 | Reward shaping scaffold tested; benchmark use pending. |
| CoVLA only vs CARLA only vs mixed fine-tuning | M-data, M5 | Dataset acquisition pending; large downloads gated. |
| 3-5 seeds, bootstrap CI, paired route comparison | M-exp | Planned before thesis result tables. |

Primary metrics:

- Route Completion
- Driving Score
- Infraction Score
- collision breakdown
- red light / stop sign / lane violation
- comfort / jerk
- p50/p95 latency
- sample efficiency
- failure-case videos
- bootstrap confidence intervals
- paired route comparison
- mean +- std

