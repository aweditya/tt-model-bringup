#!/usr/bin/env bash
# Run a command with the TT-Metal env vars set, the same way
# experiments/serve/scripts/serve_cb.sh does. Lets isolation probes
# load `ttnn` correctly without each one duplicating the env block.
#
# Why: a fresh `ssh qb1 'python -c "import ttnn"'` fails with
# `AttributeError: ttnn.WormholeComputeKernelConfig` (or similar) when
# TT_METAL_HOME / LD_LIBRARY_PATH / ARCH_NAME aren't set, because the
# system-installed ttnn binary differs from the one the build expects.
#
# Usage:
#   bash scripts/run_with_tt_env.sh <command> [args...]
# Or, on the remote host:
#   ssh qb1 'cd ~/tt-xla && bash scripts/run_with_tt_env.sh \
#       .venv/bin/python -u experiments/cb/isolate/<probe>.py'
#
# Per-host build dir override: TT_BUILD_DIR can be set; default is
# build_Release (qb1) — set TT_BUILD_DIR=build for qb2.
set -euo pipefail

# Auto-detect build dir if not provided.
if [ -z "${TT_BUILD_DIR:-}" ]; then
    if [ -d "$HOME/tenstorrent/tt-metal/build_Release" ]; then
        TT_BUILD_DIR="$HOME/tenstorrent/tt-metal/build_Release"
    elif [ -d "$HOME/tenstorrent/tt-metal/build" ]; then
        TT_BUILD_DIR="$HOME/tenstorrent/tt-metal/build"
    else
        echo "FATAL: neither build_Release nor build found under" \
             "$HOME/tenstorrent/tt-metal" >&2
        exit 2
    fi
fi

export TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tenstorrent/tt-metal}"
export TT_BUILD_DIR
export ARCH_NAME="${ARCH_NAME:-blackhole}"
export PYTHONPATH="$TT_METAL_HOME/ttnn:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"

exec "$@"
