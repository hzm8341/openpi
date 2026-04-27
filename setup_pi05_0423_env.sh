#!/usr/bin/env bash
set -euo pipefail

# Reproducible environment setup for pi0.5 0423 right-arm fine-tuning.
#
# Intended usage on the GPU server from the repository root:
#   ./setup_pi05_0423_env.sh
#
# Optional:
#   SKIP_SYNC=1 ./setup_pi05_0423_env.sh      # only run checks
#   INSTALL_UV=0 ./setup_pi05_0423_env.sh     # require uv to already exist

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
INSTALL_UV="${INSTALL_UV:-1}"
SKIP_SYNC="${SKIP_SYNC:-0}"

echo "==> Checking repository root"
if [[ ! -f "pyproject.toml" || ! -f "uv.lock" ]]; then
  echo "Run this script from the openpi repository root." >&2
  exit 1
fi

echo "==> Checking NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi; then
    echo "nvidia-smi exists but cannot communicate with the NVIDIA driver." >&2
    echo "Fix the server NVIDIA driver before training." >&2
    exit 1
  fi
else
  echo "nvidia-smi not found. Install NVIDIA driver on the server before training." >&2
  exit 1
fi

echo "==> Ensuring uv is available"
if ! command -v uv >/dev/null 2>&1; then
  if [[ "${INSTALL_UV}" != "1" ]]; then
    echo "uv not found and INSTALL_UV=0. Install uv first: https://docs.astral.sh/uv/" >&2
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

echo "==> Ensuring git-lfs is available"
if ! command -v git-lfs >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    conda install -y -c conda-forge git-lfs
  elif command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y git-lfs
  else
    echo "git-lfs not found, and neither conda nor sudo+apt-get is available." >&2
    echo "Install one of:" >&2
    echo "  conda install -y -c conda-forge git-lfs" >&2
    echo "  sudo apt-get update && sudo apt-get install -y git-lfs" >&2
    echo "  ask the server admin to install git-lfs system-wide" >&2
    exit 1
  fi
fi
git-lfs version

echo "==> Initializing git submodules"
git submodule update --init --recursive

if [[ "${SKIP_SYNC}" == "1" ]]; then
  echo "SKIP_SYNC=1, skipping uv environment sync."
  exit 0
fi

echo "==> Installing Python ${PYTHON_VERSION} and syncing uv environment"
uv python install "${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}"

# Required by openpi README for pulling LeRobot as a dependency without LFS smudge.
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo "==> Verifying training config import"
uv run python - <<'PY'
from openpi.training import config as _config

cfg = _config.get_config("pi05_0423_right8_chest_wrist_finetune")
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
print("config:", cfg.name)
print("action_dim:", cfg.model.action_dim)
print("action_horizon:", cfg.model.action_horizon)
print("repo_id:", data_cfg.repo_id)
print("asset_id:", data_cfg.asset_id)
print("norm_stats:", sorted(data_cfg.norm_stats.keys()) if data_cfg.norm_stats else None)
PY

echo "==> Environment setup complete"
echo "Next: make sure the dataset and assets are present, then run:"
echo "  ./run_train_pi05_0423_right8.sh"
