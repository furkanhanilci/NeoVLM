#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
LOG_DIR="$ROOT/setup_logs"
mkdir -p "$LOG_DIR"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update 2>&1 | tee "$LOG_DIR/nvidia_container_toolkit_apt_update.log"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit \
  2>&1 | tee "$LOG_DIR/nvidia_container_toolkit_install.log"
sudo nvidia-ctk runtime configure --runtime=docker \
  2>&1 | tee "$LOG_DIR/nvidia_ctk_runtime_configure.log"
sudo systemctl restart docker \
  2>&1 | tee "$LOG_DIR/docker_restart_after_nvidia_ctk.log"

