#!/usr/bin/env bash
# Conda environment for SUFLECA.
# Usage: bash scripts/create_env.sh [env_name]   (default: sufleca)
set -euo pipefail

# Run from the repo root so the relative install paths below resolve.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENV_NAME="${1:-sufleca}"

conda create -n "$ENV_NAME" python=3.10 -y
conda install -n "$ENV_NAME" -c conda-forge cmake pybind11 eigen boost compilers -y

conda run -n "$ENV_NAME" pip install --upgrade pip
# PyTorch for CUDA 12.8 (required for Blackwell / RTX 50-series GPUs).
conda run -n "$ENV_NAME" pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
# Install the native extension into this environment. An editable native build
# lives in the shared source tree and can be silently reused by another Conda
# environment with a different ABI.
conda run -n "$ENV_NAME" pip install --no-build-isolation third_party/superansac

# Install SUFLECA and its dependencies (notebook kernel, UI, SAM2, etc.).
# Disabling build isolation makes SAM2 reuse the pinned CUDA PyTorch above
# instead of downloading a second build.
conda run -n "$ENV_NAME" pip install --no-build-isolation -e .

echo "Done. Activate with: conda activate $ENV_NAME"
