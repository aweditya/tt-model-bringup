#!/usr/bin/env bash
# Phase A — Filesystem prep on qb1.
# Run via: ssh qb1 'bash -s' < pjrt_plugin/scripts/qb1_phase_a_prep.sh
set -euo pipefail

echo "[phase-a] Creating directories..."
mkdir -p ~/tenstorrent
mkdir -p ~/tt-xla/.cache/ttnn
mkdir -p ~/tt-xla/.cache/build

# Redirect TTNN cache away from /tmp.
# Set the env vars for the current shell + persist to ~/.bashrc if not already there.
if ! grep -q "TTNN_CACHE_DIR" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<'EOF'

# tt-xla project: redirect caches away from /tmp
export TTNN_CACHE_DIR="$HOME/tt-xla/.cache/ttnn"
export TT_METAL_HOME="$HOME/tenstorrent/tt-metal"
export PATH="$HOME/.local/bin:$PATH"
EOF
  echo "[phase-a] Added env vars to ~/.bashrc"
fi

echo "[phase-a] Done. Directories:"
ls -la ~/tenstorrent ~/tt-xla/.cache
