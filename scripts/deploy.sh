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
  # Glob the whole active surface so a new file under serve/, cb/, or the
  # serve scripts dir doesn't silently fall off `make dr`. Pre-glob caught us
  # twice (full_layer_tp_probe, then the CB stack); the audit added a static
  # check (scripts/ci_check_deploy_sync.py) but glob is the real fix.
  set -- experiments/serve/*.py \
         experiments/serve/scripts/*.sh \
         experiments/cb/needle.py \
         experiments/cb/validate/*.py \
         experiments/cb/bench/*.py \
         experiments/cb/isolate/*.py \
         experiments/cb/load/*.py \
         experiments/cb/profile/*.py \
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
