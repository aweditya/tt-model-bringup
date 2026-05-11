#!/usr/bin/env bash
# Phase E — Sync local tt-xla code to qb1.
# Run LOCALLY: bash pjrt_plugin/scripts/qb1_phase_e_sync.sh
set -euo pipefail

# Sync project files. Excludes:
# - .venv/ (created on remote, not synced)
# - .cache/ (host-specific)
# - __pycache__/ (regenerated)
# - .git/ (use git pull on remote if needed; here we sync files directly)
# - *.so / build/ (rebuilt on remote)
# - tenstorrent/ (the worktree dir picked up by `?? tenstorrent` accidentally — not part of project)

rsync -avz --delete \
  --exclude='.venv/' \
  --exclude='.cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.git/' \
  --exclude='build/' \
  --exclude='*.so' \
  --exclude='tenstorrent/' \
  --exclude='.pytest_cache/' \
  ./ qb1:~/tt-xla/

echo "[phase-e] Sync complete."
ssh qb1 'ls ~/tt-xla/'
