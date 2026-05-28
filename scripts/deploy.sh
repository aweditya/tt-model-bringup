#!/usr/bin/env bash
# Sync local files to the Tenstorrent host (~/tt-xla), preserving repo-relative
# paths so `scripts/run_remote.sh <path>` finds them.
#
#   scripts/deploy.sh <path> [<path> ...]
#
# With no args, syncs the active serving + validation/bench code. Host defaults
# to qb1 (TT_HOST=qb2 to override).
set -euo pipefail

HOST="${TT_HOST:-qb1}"
if [[ "$#" -eq 0 ]]; then
  set -- experiments/serve/server_tp.py experiments/serve/server_tp_cb.py \
         experiments/serve/cb_scheduler.py experiments/cb_validate_27b.py \
         experiments/cb_bench_trace.py experiments/cb_needle.py
fi
for p in "$@"; do
  rsync -az "$p" "$HOST:~/tt-xla/$p"
done
echo "deployed ${#} path(s) to $HOST:~/tt-xla/"
