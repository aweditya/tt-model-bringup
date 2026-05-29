#!/usr/bin/env bash
# Phase G — Install model bringup deps + set up HuggingFace cache.
# Required by experiments/ scripts that load Llama/Qwen/etc.
# Run via: ssh qb1 'bash -s' < pjrt_plugin/scripts/qb1_phase_g_model_deps.sh
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd ~/tt-xla

echo "[phase-g] Installing transformers, safetensors, huggingface_hub..."
# Pin to versions known good from REPRODUCE.md (transformers 5.5.x, etc.)
# Note: we don't need a specific transformers version for model loading via
# safetensors — most ports load weights directly without transformers' classes.
# But many experiments import transformers for tokenizers. Latest is fine.
uv pip install --python .venv/bin/python \
  safetensors \
  huggingface_hub \
  transformers \
  tokenizers \
  sentencepiece

echo "[phase-g] Setting up HF cache under project (no /tmp)..."
mkdir -p ~/tt-xla/.cache/hf

# Add HF_HOME to bashrc if not already there
if ! grep -q "HF_HOME=" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<'EOF'

# HuggingFace cache: keep model weights in-project, never in /tmp or ~/.cache
export HF_HOME="$HOME/tt-xla/.cache/hf"
export HUGGINGFACE_HUB_CACHE="$HOME/tt-xla/.cache/hf/hub"
EOF
  echo "[phase-g] HF_HOME exported in ~/.bashrc"
fi

echo "[phase-g] Verification:"
.venv/bin/python - <<'PY'
import safetensors, huggingface_hub, transformers
print(f"  safetensors      = {safetensors.__version__}")
print(f"  huggingface_hub  = {huggingface_hub.__version__}")
print(f"  transformers     = {transformers.__version__}")
PY

echo "[phase-g] Done."
