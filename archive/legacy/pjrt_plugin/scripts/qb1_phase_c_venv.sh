#!/usr/bin/env bash
# Phase C — Install uv and set up project venv.
# Run via: ssh qb1 'bash -s' < pjrt_plugin/scripts/qb1_phase_c_venv.sh
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

# Install uv if not present
if ! command -v uv &>/dev/null; then
  echo "[phase-c] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv --version

# Create project venv with Python 3.10 (default on Ubuntu 22.04)
# uv will fetch a managed Python if 3.10 not present locally.
cd ~/tt-xla 2>/dev/null || mkdir -p ~/tt-xla
cd ~/tt-xla

if [ ! -d .venv ]; then
  echo "[phase-c] Creating venv at ~/tt-xla/.venv..."
  uv venv .venv --python 3.10
fi

# Install deps — pin to known-good versions for PJRT API v0.70
# jax 0.6.2 / jaxlib 0.6.2 ships PJRT_Api_STRUCT_SIZE = 944 (v0.70)
echo "[phase-c] Installing Python deps via uv..."
uv pip install --python .venv/bin/python \
  numpy \
  pytest \
  torch \
  jax==0.6.2 \
  jaxlib==0.6.2

echo "[phase-c] Verification:"
.venv/bin/python -c "import jax, jaxlib, numpy, torch, pytest; print(f'jax={jax.__version__} jaxlib={jaxlib.__version__} numpy={numpy.__version__} torch={torch.__version__}')"

echo "[phase-c] Done. Venv at ~/tt-xla/.venv"
