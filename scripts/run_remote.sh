#!/usr/bin/env bash
# Run a Python script on the Tenstorrent host with the ttnn environment.
#
# Single source of truth for the device-run incantation: the ttnn env block +
# the tt-smi mesh reset. Replaces hand-typing the 5-line env block everywhere.
#
#   scripts/run_remote.sh [--no-reset] <path-relative-to-repo-root> [args...]
#
# Host defaults to qb1; override with TT_HOST=qb2. Assumes the file is already
# synced to the host (use scripts/deploy.sh first, or `make dr`).
set -euo pipefail

HOST="${TT_HOST:-qb1}"
reset=1
if [[ "${1:-}" == "--no-reset" ]]; then reset=0; shift; fi
script="${1:?usage: run_remote.sh [--no-reset] <script.py> [args...]}"; shift || true

reset_cmd=""
[[ "$reset" == 1 ]] && reset_cmd="tt-smi -r 0,1,2,3 >/dev/null 2>&1 &&"

# shellcheck disable=SC2029  # env vars intentionally expand on the remote host
ssh "$HOST" "cd ~/tt-xla && ${reset_cmd} \
  TT_METAL_HOME=\$HOME/tenstorrent/tt-metal \
  TT_BUILD_DIR=\$HOME/tenstorrent/tt-metal/build_Release \
  ARCH_NAME=blackhole \
  PYTHONPATH=\$HOME/tenstorrent/tt-metal/ttnn \
  LD_LIBRARY_PATH=\$HOME/tenstorrent/tt-metal/ttnn/ttnn:\$HOME/tenstorrent/tt-metal/build_Release/ttnn:\$HOME/tenstorrent/tt-metal/build_Release/lib \
  .venv/bin/python -u ${script} $*"
