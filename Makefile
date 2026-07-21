SHELL := /usr/bin/env bash

.PHONY: check-gpu check-torch check-carla check-bench2drive check-docker-gpu check-research test token-smoke vlm-smoke feature-cache-smoke feature-cache-dataset dataset-stats eval-report bc-smoke bc-train bc-rollout-smoke bc-bridge-smoke smoke carla-server carla-start carla-window carla-status carla-rollout-smoke carla-dataset-smoke carla-dataset-collect validate-carla-dataset thesis-pdf

check-gpu:
	./scripts/check_gpu.sh

check-torch:
	./scripts/check_torch.sh

check-carla:
	./scripts/check_carla.sh

check-bench2drive:
	./scripts/check_bench2drive.sh

check-docker-gpu:
	./scripts/check_docker_gpu.sh

check-research:
	./scripts/check_research_code.sh

test:
	micromamba run -n vlm pytest tests -q

token-smoke:
	./scripts/run_token_pipeline_smoke.sh

vlm-smoke:
	./scripts/run_vlm_provider_smoke.sh

feature-cache-smoke:
	./scripts/run_feature_cache_smoke.sh

feature-cache-dataset:
	./scripts/build_feature_cache_dataset.sh

dataset-stats:
	./scripts/run_dataset_stats.sh

eval-report:
	./scripts/run_eval_report.sh

bc-smoke:
	./scripts/run_bc_smoke.sh

bc-train:
	./scripts/run_bc_train.sh

bc-rollout-smoke:
	./scripts/run_bc_rollout_smoke.sh

bc-bridge-smoke:
	./scripts/run_bc_bridge_smoke.sh

smoke:
	./scripts/run_smoke_tests.sh

carla-server:
	./scripts/run_carla_server.sh

carla-start:
	./scripts/start_carla_server_detached.sh

carla-window:
	./scripts/start_carla_window_detached.sh

carla-status:
	./scripts/carla_status.sh

carla-rollout-smoke:
	./scripts/run_carla_rollout_smoke.sh

carla-dataset-smoke:
	./scripts/run_carla_dataset_smoke.sh

carla-dataset-collect:
	./scripts/run_carla_dataset_collect.sh

validate-carla-dataset:
	./scripts/validate_carla_dataset.py results/datasets/carla_il_smoke/episode_000

thesis-pdf:
	micromamba run -n analysis quarto render thesis/main.qmd --to pdf
