#!/bin/bash
# Build the PJRT plugin on the remote Tenstorrent host.
# Phase 1: no ttnn dependency, just the PJRT skeleton.
#
# Usage: cd ~/tt-xla/pjrt_plugin && bash scripts/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PLUGIN_DIR/build"

echo "=== Building PJRT Plugin (Phase 1) ==="
echo "Source: $PLUGIN_DIR"
echo "Build:  $BUILD_DIR"

cmake --version || { echo "ERROR: cmake not found"; exit 1; }

echo ""
echo "=== Configuring ==="
cmake -B "$BUILD_DIR" -S "$PLUGIN_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo ""
echo "=== Building ==="
cmake --build "$BUILD_DIR" -j$(nproc)

echo ""
echo "=== Result ==="
ls -la "$BUILD_DIR"/libpjrt_plugin_tt.so

echo ""
echo "=== Checking exported symbols ==="
nm -D "$BUILD_DIR"/libpjrt_plugin_tt.so | grep GetPjrtApi || echo "WARNING: GetPjrtApi not found!"
