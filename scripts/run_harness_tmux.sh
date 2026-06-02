#!/usr/bin/env bash
# Launch the cb35 dev harness inside a detached tmux session on the remote
# Tenstorrent host. tmux survives SSH disconnects properly; nohup/setsid in
# our environment do NOT (the python process gets killed mid-bootstrap).
#
# Usage:
#   bash scripts/run_harness_tmux.sh [host]
# Default host: qb1
#
# To attach:    ssh <host> tmux attach -t cb35
# To kill:      ssh <host> tmux kill-session -t cb35
# Log file:     ~/tt-xla/.cache/cb35_runtime/harness.log on the host

set -euo pipefail
HOST="${1:-qb1}"

ssh "$HOST" bash <<'REMOTE'
set -e
pgrep -f experiments/cb/dev/cb35_dev_harness | xargs -r kill -9 2>/dev/null || true
tmux kill-session -t cb35 2>/dev/null || true
sleep 1
mkdir -p ~/tt-xla/.cache/cb35_runtime/trig
rm -f ~/tt-xla/.cache/cb35_runtime/harness.log
cd ~/tt-xla
tmux new-session -d -s cb35 \
  "cd ~/tt-xla && \
   export TT_METAL_HOME=\$HOME/tenstorrent/tt-metal && \
   export TT_BUILD_DIR=\$TT_METAL_HOME/build_Release && \
   export ARCH_NAME=blackhole && \
   export PYTHONPATH=\$TT_METAL_HOME/ttnn && \
   export LD_LIBRARY_PATH=\$TT_METAL_HOME/ttnn/ttnn:\$TT_BUILD_DIR/ttnn:\$TT_BUILD_DIR/lib && \
   exec .venv/bin/python -u experiments/cb/dev/cb35_dev_harness.py 2>&1 | tee .cache/cb35_runtime/harness.log"
sleep 2
echo "=== tmux sessions ==="
tmux ls 2>&1
echo "=== harness pid ==="
pgrep -af "python.*cb35_dev_harness" | head -2
REMOTE
