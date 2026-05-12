#!/bin/bash
# Run a Python command using the Tracy-enabled ttnn build at
# ~/tenstorrent/tt-metal (the source-built variant), instead of the
# stock no-Tracy wheel installed in ~/tt-xla/.venv.
#
# Mechanism:
#   - PYTHONPATH prepends $TT_METAL_HOME/ttnn so Python's import system
#     finds the source-tree ttnn package BEFORE the installed wheel
#   - LD_LIBRARY_PATH prepends the build's lib/ so the shared objects
#     (libtt_metal.so, _ttnn.so, libTracyClient.so) resolve to the
#     Tracy-enabled build
#   - TT_METAL_HOME points ttnn at the source-tree data files
#     (pre-compiled firmware, hw configs, etc.)
#
# Usage:
#   ./run_with_tracy_build.sh /path/to/python script.py [args...]
#
# Enable per-op device profiling:
#   TT_METAL_DEVICE_PROFILER=1 ./run_with_tracy_build.sh \
#       ~/tt-xla/.venv/bin/python experiments/utils/tracy_smoke_test.py
#
# To revert to the stock wheel, just don't use this wrapper.

set -e

TT_METAL_BUILD_DIR=${TT_METAL_BUILD_DIR:-~/tenstorrent/tt-metal}

if [ ! -d "$TT_METAL_BUILD_DIR/build_Release" ]; then
    echo "ERROR: $TT_METAL_BUILD_DIR/build_Release does not exist."
    echo "       Build first:  cd $TT_METAL_BUILD_DIR && ./build_metal.sh --release"
    exit 1
fi

if [ ! -f "$TT_METAL_BUILD_DIR/ttnn/ttnn/_ttnn.so" ]; then
    echo "ERROR: $TT_METAL_BUILD_DIR/ttnn/ttnn/_ttnn.so missing."
    echo "       Build may be incomplete."
    exit 1
fi

export TT_METAL_HOME="$TT_METAL_BUILD_DIR"
export ARCH_NAME=blackhole
export PYTHONPATH="$TT_METAL_BUILD_DIR/ttnn:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TT_METAL_BUILD_DIR/build_Release/lib:${LD_LIBRARY_PATH:-}"

if [ -z "$*" ]; then
    echo "Usage: $0 <python-binary> <script.py> [args...]"
    echo
    echo "Current env:"
    echo "  TT_METAL_HOME    = $TT_METAL_HOME"
    echo "  PYTHONPATH       = $PYTHONPATH"
    echo "  LD_LIBRARY_PATH  = $LD_LIBRARY_PATH"
    echo "  TT_METAL_DEVICE_PROFILER = ${TT_METAL_DEVICE_PROFILER:-<unset>}"
    exit 0
fi

exec "$@"
