#!/usr/bin/env bash
# One-shot setup of a Gemma 4 12B dev venv on the Tenstorrent host.
#
# qb1's main .venv ships transformers 5.9.0 which does not recognize
# the `gemma4_unified` model type (released 2026-06-03). We keep the
# main .venv pristine for 27B/35B prod and install Gemma 4's deps in
# a sibling venv so the HF reference oracle + tokenizer can load
# google/gemma-4-12B.
#
# This script is idempotent: re-running it upgrades transformers in
# place and verifies the Gemma 4 config loads at the end.
#
# Usage:  bash scripts/setup_venv_gemma4.sh [host]
# Host defaults to qb1; pass qb2 to set up there too.
#
# Memory cross-reference: research/gemma4_12b_bringup_plan.md §0 +
# user mandate "Create a separate venv for Gemma 4 work" 2026-06-03.

set -euo pipefail
HOST="${1:-qb1}"
VENV_DIR=".venv-gemma4"

ssh "$HOST" bash <<REMOTE
set -e
cd ~/tt-xla

if [[ ! -d $VENV_DIR ]]; then
  echo "[setup] creating $VENV_DIR ..."
  python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

echo "[setup] upgrading pip ..."
pip install --quiet --upgrade pip

echo "[setup] installing transformers from git (required for gemma4_unified) ..."
pip install --quiet --upgrade \
  "git+https://github.com/huggingface/transformers.git" \
  torch numpy safetensors huggingface_hub accelerate sentencepiece

echo "[setup] verifying Gemma 4 12B config loads ..."
python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('google/gemma-4-12B')
print('  model_type:', c.model_type)
print('  text_model_type:', c.text_config.model_type)
print('  num_hidden_layers:', c.text_config.num_hidden_layers)
print('  hidden_size:', c.text_config.hidden_size)
print('  vocab_size:', c.text_config.vocab_size)
"

echo "[setup] DONE. Activate with:  source ~/tt-xla/$VENV_DIR/bin/activate"
REMOTE
