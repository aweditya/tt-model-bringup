#!/usr/bin/env bash
# Launch a dev harness inside a detached tmux session on the remote
# Tenstorrent host. tmux survives SSH disconnects properly; nohup/setsid in
# our environment do NOT (the python process gets killed mid-bootstrap).
#
# Usage:
#   bash scripts/run_harness_tmux.sh [model] [host]
# model: cb35 (default) | gm4
# host:  qb1 (default)
#
# To attach:    ssh <host> tmux attach -t <model>
# To kill:      ssh <host> tmux kill-session -t <model>
# Log file:     ~/tt-xla/.cache/<model>_runtime/harness.log on the host

set -euo pipefail
MODEL="${1:-cb35}"
HOST="${2:-qb1}"

case "$MODEL" in
  cb35) HARNESS_PATH="experiments/cb/dev/cb35_dev_harness.py"; RUNTIME_DIR="cb35_runtime";;
  gm4)  HARNESS_PATH="experiments/cb/dev/gm4_dev_harness.py";  RUNTIME_DIR="gm4_runtime";;
  nm3)  HARNESS_PATH="experiments/cb/dev/nm3_dev_harness.py";  RUNTIME_DIR="nm3_runtime";;
  *) echo "unknown model: $MODEL (expected cb35|gm4|nm3)"; exit 1;;
esac

# Propagate select env vars into the tmux session so model-variant
# switches don't need a code edit. Add new vars here as bringups need them.
PASS_THROUGH_ENV=""
for var in TT_GEMMA4_VARIANT TT_GEMMA4_MODEL_ID TT_CB_SLOTS TT_CB_TOPK_K \
           NEMOTRON3_UPLOAD_LAYERS NEMOTRON3_MOE_MODE \
           NM3_ROUTER_ON_DEVICE NM3_NEEDLE_MAX_NEW; do
  if [ -n "${!var:-}" ]; then
    PASS_THROUGH_ENV="$PASS_THROUGH_ENV $var=${!var}"
  fi
done

# qb1 ships `build_Release`; qb2 ships `build` (different cmake preset). Allow
# the caller to override; default per-host so callers don't have to remember.
# (See research/gemma4_perf_qb2_2026-06-05/log.md §"qb2 tt-metal builds".)
if [ -z "${TT_BUILD_NAME:-}" ]; then
  case "$HOST" in
    qb2) TT_BUILD_NAME="build";;
    *)   TT_BUILD_NAME="build_Release";;
  esac
fi

ssh "$HOST" bash <<REMOTE
set -e
pgrep -f $HARNESS_PATH | xargs -r kill -9 2>/dev/null || true
tmux kill-session -t $MODEL 2>/dev/null || true
sleep 1
mkdir -p ~/tt-xla/.cache/$RUNTIME_DIR/trig
rm -f ~/tt-xla/.cache/$RUNTIME_DIR/harness.log
# Clear stale triggers (especially _exit from a previous shutdown) so the
# fresh harness doesn't immediately exit or run an old test.
rm -f ~/tt-xla/.cache/$RUNTIME_DIR/trig/*
cd ~/tt-xla
tmux new-session -d -s $MODEL \\
  "cd ~/tt-xla && \\
   export TT_METAL_HOME=\\\$HOME/tenstorrent/tt-metal && \\
   export TT_BUILD_DIR=\\\$TT_METAL_HOME/$TT_BUILD_NAME && \\
   export ARCH_NAME=blackhole && \\
   export PYTHONPATH=\\\$TT_METAL_HOME/ttnn && \\
   export LD_LIBRARY_PATH=\\\$TT_METAL_HOME/ttnn/ttnn:\\\$TT_BUILD_DIR/ttnn:\\\$TT_BUILD_DIR/lib && \\
   $PASS_THROUGH_ENV exec .venv/bin/python -u $HARNESS_PATH"
sleep 2
echo "=== tmux sessions ==="
tmux ls 2>&1
echo "=== harness pid ==="
pgrep -af "python.*$(basename $HARNESS_PATH)" | head -2
echo "=== env passed ==="
echo "$PASS_THROUGH_ENV"
REMOTE
