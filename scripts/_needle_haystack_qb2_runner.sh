#!/usr/bin/env bash
# One-shot runner used by the gm4 needle-haystack quick-gate (2026-06-05).
# Sets the ttnn env + resets devices, then exec's the probe with all args
# forwarded. Lives under scripts/ so it survives across sessions and is
# rsynced to qb2 via deploy.sh-style flows. Forks the env block from
# scripts/run_remote_qb2.sh (run-locally script that ssh's into qb2);
# this version is meant to be exec'd ON qb2 from inside a tmux session.
set -euo pipefail
cd "$HOME/tt-xla"
tt-smi -r 0,1,2,3 >/dev/null 2>&1 || true
export TT_METAL_HOME="$HOME/tenstorrent/tt-metal"
export TT_BUILD_DIR="$HOME/tenstorrent/tt-metal/build"
export ARCH_NAME=blackhole
export PYTHONPATH="$HOME/tenstorrent/tt-metal/ttnn"
export LD_LIBRARY_PATH="$HOME/tenstorrent/tt-metal/ttnn/ttnn:$HOME/tenstorrent/tt-metal/build/ttnn:$HOME/tenstorrent/tt-metal/build/lib"
export HF_HUB_OFFLINE=1
export TT_GEMMA4_VARIANT="${TT_GEMMA4_VARIANT:-it}"
# Round 9 ablation: pass through optional output-subdir override so per-variant
# needle results land in research/.../needle_haystack/<TT_GM4_NEEDLE_OUT_SUBDIR>/.
export TT_GM4_NEEDLE_OUT_SUBDIR="${TT_GM4_NEEDLE_OUT_SUBDIR:-}"
# Round 9 ablation: dtype gates on MLP + lm_head upload (bf16 default).
# Set to "bfp8" to re-enable the Round 8 shape on a per-piece basis.
export TT_GM4_MLP_DTYPE="${TT_GM4_MLP_DTYPE:-bf16}"
export TT_GM4_LM_HEAD_DTYPE="${TT_GM4_LM_HEAD_DTYPE:-bf16}"
exec .venv/bin/python -u experiments/cb/isolate/gm4_v04_needle_haystack_traced.py "$@"
