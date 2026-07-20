#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/research/vlm-driving-thesis}"
LOG_DIR="$ROOT/setup_logs"
mkdir -p "$LOG_DIR"
sudo apt-get update 2>&1 | tee "$LOG_DIR/apt_update.log"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake ninja-build git git-lfs curl wget unzip p7zip-full \
  pkg-config htop nvtop tmux tree ffmpeg mesa-utils vulkan-tools \
  ca-certificates gnupg2 lsb-release software-properties-common \
  docker.io docker-compose-v2 python3-venv python3-pip pipx \
  pandoc texlive-xetex texlive-fonts-recommended texlive-latex-extra latexmk \
  libreoffice graphviz xclip \
  2>&1 | tee "$LOG_DIR/apt_install_base.log"
git lfs install 2>&1 | tee "$LOG_DIR/git_lfs_install.log"

