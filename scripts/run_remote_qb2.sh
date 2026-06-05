#!/usr/bin/env bash
# Run a Python script on qb2 with the ttnn environment.
#
# qb2 builds tt-metal at ~/tenstorrent/tt-metal/build (NOT build_Release like qb1),
# so we cannot use scripts/run_remote.sh verbatim (it hardcodes TT_BUILD_DIR=build_Release).
# This script is a fork of run_remote.sh with TT_BUILD_DIR=build for qb2.
#
#   scripts/run_remote_qb2.sh [--no-reset] <path-relative-to-repo-root> [args...]
#
# Forks: scripts/run_remote.sh:1-29 (same arg parsing + env structure, only
# TT_BUILD_DIR differs because qb2's build layout differs from qb1).
set -euo pipefail

HOST="qb2"
reset=1
if [[ "${1:-}" == "--no-reset" ]]; then reset=0; shift; fi
script="${1:?usage: run_remote_qb2.sh [--no-reset] <script.py> [args...]}"; shift || true

reset_cmd=""
[[ "$reset" == 1 ]] && reset_cmd="tt-smi -r 0,1,2,3 >/dev/null 2>&1 &&"

# shellcheck disable=SC2029  # env vars intentionally expand on the remote host
ssh "$HOST" "cd ~/tt-xla && ${reset_cmd} \
  TT_METAL_HOME=\$HOME/tenstorrent/tt-metal \
  TT_BUILD_DIR=\$HOME/tenstorrent/tt-metal/build \
  ARCH_NAME=blackhole \
  PYTHONPATH=\$HOME/tenstorrent/tt-metal/ttnn \
  LD_LIBRARY_PATH=\$HOME/tenstorrent/tt-metal/ttnn/ttnn:\$HOME/tenstorrent/tt-metal/build/ttnn:\$HOME/tenstorrent/tt-metal/build/lib \
  HF_HUB_OFFLINE=1 \
  TT_GEMMA4_VARIANT=\${TT_GEMMA4_VARIANT:-it} \
  .venv/bin/python -u ${script} $*"
