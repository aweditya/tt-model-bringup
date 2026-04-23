#!/bin/bash
# Build the PJRT plugin on the remote Tenstorrent host.
#
# Prerequisites:
#   - TT_METAL_HOME is set
#   - CMake >= 3.20
#   - GCC or Clang with C++17 support
#
# Usage: ./scripts/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PLUGIN_DIR/build"

echo "=== Environment ==="
echo "TT_METAL_HOME: ${TT_METAL_HOME:-not set}"
echo "Plugin dir: $PLUGIN_DIR"
echo "Build dir: $BUILD_DIR"

# Verify TT_METAL_HOME
if [ -z "${TT_METAL_HOME:-}" ]; then
    echo "ERROR: TT_METAL_HOME not set"
    echo "Set it to the tt-metal installation directory, e.g.:"
    echo "  export TT_METAL_HOME=/opt/tt-metal"
    exit 1
fi

# Verify cmake
cmake --version || { echo "ERROR: cmake not found"; exit 1; }

echo ""
echo "=== Configuring ==="
cmake -B "$BUILD_DIR" -S "$PLUGIN_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo

echo ""
echo "=== Building ==="
cmake --build "$BUILD_DIR" -j$(nproc)

echo ""
echo "=== Result ==="
ls -la "$BUILD_DIR"/libpjrt_plugin_tt.so 2>/dev/null && echo "Build succeeded!" || echo "Build failed!"
