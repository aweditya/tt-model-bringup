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
         experiments/serve/ondevice_27b.py experiments/serve/generate_27b.py \
         experiments/serve/cb_scheduler.py experiments/cb/validate/forward.py \
         experiments/cb/bench/trace.py experiments/cb/needle.py \
         experiments/utils/full_layer_tp_probe.py \
         experiments/utils/tp_attn_traced_probe.py
fi
# -R (relative) recreates the repo-relative dir structure on the host, so newly
# nested paths (e.g. experiments/cb/validate/forward.py) don't need the remote
# dirs to pre-exist.
for p in "$@"; do
  rsync -azR "$p" "$HOST:~/tt-xla/"
done
echo "deployed ${#} path(s) to $HOST:~/tt-xla/"
