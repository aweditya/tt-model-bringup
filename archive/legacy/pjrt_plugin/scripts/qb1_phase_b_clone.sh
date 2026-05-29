#!/usr/bin/env bash
# Phase B — Clone Tenstorrent repos on qb1.
# Run via: ssh qb1 'bash -s' < pjrt_plugin/scripts/qb1_phase_b_clone.sh
set -euo pipefail

cd ~/tenstorrent

# tt-metal: pin to v0.69.0 (latest stable). Submodules needed (umd, llk-bh, etc.)
# We use --depth=1 and --shallow-submodules to avoid pulling years of history.
if [ ! -d tt-metal ]; then
  echo "[phase-b] Cloning tt-metal v0.69.0 (with submodules, shallow)..."
  git clone --depth=1 --branch v0.69.0 --recursive --shallow-submodules \
    https://github.com/tenstorrent/tt-metal.git
else
  echo "[phase-b] tt-metal already present, skipping"
fi

# tt-mlir: official PJRT/MLIR compiler — reference for our PJRT plugin
if [ ! -d tt-mlir ]; then
  echo "[phase-b] Cloning tt-mlir..."
  git clone --depth=1 --recursive --shallow-submodules \
    https://github.com/tenstorrent/tt-mlir.git &
fi

# tt-llk: standalone low-level kernels
[ ! -d tt-llk ] && git clone --depth=1 https://github.com/tenstorrent/tt-llk.git &

# Lightweight clones (no submodules)
for repo in \
  tt-lang \
  tt-isa-documentation \
  tt-umd \
  tt-smi \
  tt-exalens \
  tt-forge-models \
  tt-vscode-toolkit \
  tt-system-firmware \
  tt-buda \
  tt-inference-server \
  sfpi \
  sfpi-gcc \
  polaris; do
  if [ ! -d "$repo" ]; then
    echo "[phase-b] Cloning $repo..."
    git clone --depth=1 "https://github.com/tenstorrent/$repo.git" &
  fi
done

# Official tt-xla — rename to avoid collision with our ~/tt-xla project
if [ ! -d tt-xla-official ]; then
  echo "[phase-b] Cloning tt-xla (official, renamed to tt-xla-official)..."
  git clone --depth=1 https://github.com/tenstorrent/tt-xla.git tt-xla-official &
fi

wait

echo "[phase-b] All clones complete. Sizes:"
du -sh ~/tenstorrent/* | sort -h
