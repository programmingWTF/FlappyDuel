#!/usr/bin/env bash
# Create the project conda environment INSIDE this folder (./env) and install
# GPU-enabled PyTorch. Nothing is installed into the base env.
# Run from the project root:  bash setup_env.sh
set -euo pipefail

ENV_PREFIX="$(cd "$(dirname "$0")" && pwd)/env"

if [ ! -d "$ENV_PREFIX" ]; then
  echo "Creating conda env at $ENV_PREFIX ..."
  conda create --prefix "$ENV_PREFIX" python=3.11 -y
fi

source "$ENV_PREFIX/bin/activate"

python -m pip install --upgrade pip
echo "Installing PyTorch (CUDA 12.8, Blackwell sm_120 support) ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo "Installing remaining deps ..."
pip install pygame tensorboard numpy

echo
echo "Done. Activate with:  source $ENV_PREFIX/bin/activate"
