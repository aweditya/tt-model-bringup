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
  *) echo "unknown model: $MODEL (expected cb35|gm4)"; exit 1;;
esac

ssh "$HOST" bash <<REMOTE
set -e
pgrep -f $HARNESS_PATH | xargs -r kill -9 2>/dev/null || true
tmux kill-session -t $MODEL 2>/dev/null || true
sleep 1
mkdir -p ~/tt-xla/.cache/$RUNTIME_DIR/trig
rm -f ~/tt-xla/.cache/$RUNTIME_DIR/harness.log
cd ~/tt-xla
tmux new-session -d -s $MODEL \\
  "cd ~/tt-xla && \\
   export TT_METAL_HOME=\\\$HOME/tenstorrent/tt-metal && \\
   export TT_BUILD_DIR=\\\$TT_METAL_HOME/build_Release && \\
   export ARCH_NAME=blackhole && \\
   export PYTHONPATH=\\\$TT_METAL_HOME/ttnn && \\
   export LD_LIBRARY_PATH=\\\$TT_METAL_HOME/ttnn/ttnn:\\\$TT_BUILD_DIR/ttnn:\\\$TT_BUILD_DIR/lib && \\
   exec .venv/bin/python -u $HARNESS_PATH"
sleep 2
echo "=== tmux sessions ==="
tmux ls 2>&1
echo "=== harness pid ==="
pgrep -af "python.*$(basename $HARNESS_PATH)" | head -2
REMOTE
