#!/usr/bin/env bash
# Regression sweep for the owned Mamba2 SSD decode kernel.
#
# Runs the full set of correctness gates the kernel has so far:
#   - Single-step smoke for modes 2, 3, 4, 5 (mamba2_kernel_mode3_smoke.py)
#   - 8-step multi-step replay (mamba2_multi_step_replay.py)
#
# Each test resets the device first (tt-smi -r 0,1,2,3). Run on the
# QuietBox; not portable to other hosts. Exit non-zero on any failure.
#
# Usage (from project root):
#   ssh $TT_HOST 'cd ~/tt-xla && bash experiments/cb/isolate/mamba2_regression_sweep.sh'
#
# Or run locally if you've sourced the qb env:
#   bash experiments/cb/isolate/mamba2_regression_sweep.sh

set -e

export TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tenstorrent/tt-metal}"
export TT_BUILD_DIR="${TT_BUILD_DIR:-$TT_METAL_HOME/build_Release}"
export ARCH_NAME="${ARCH_NAME:-blackhole}"
export PYTHONPATH="${PYTHONPATH:-$TT_METAL_HOME/ttnn}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib}"

PYTHON="${PYTHON:-.venv/bin/python}"
SMOKE="experiments/cb/isolate/mamba2_kernel_mode3_smoke.py"
REPLAY="experiments/cb/isolate/mamba2_multi_step_replay.py"
G2_SMOKE="experiments/cb/isolate/mamba2_g2_multihead_smoke.py"

reset_device() {
    tt-smi -r 0,1,2,3 >/dev/null 2>&1 || true
    sleep 3
}

run_mode() {
    local mode=$1
    echo "── mode=$mode ──"
    reset_device
    timeout 60 $PYTHON -u $SMOKE --mode $mode 2>&1 | \
        grep -E "PASS|FAIL|state_out vs|y_out vs" | head -4
    local result=${PIPESTATUS[0]}
    if [[ $result -ne 0 ]]; then
        echo "✗ FAIL (mode=$mode exit $result)"
        return 1
    fi
}

run_replay() {
    echo "── 8-step multi-step replay (single-head recurrence) ──"
    reset_device
    timeout 120 $PYTHON -u $REPLAY --n-steps 8 2>&1 | \
        grep -E "PASS|FAIL|Mamba2 multi-step" | head -4
    local result=${PIPESTATUS[0]}
    if [[ $result -ne 0 ]]; then
        echo "✗ FAIL (replay exit $result)"
        return 1
    fi
}

run_g2() {
    echo "── G2 64-head multi-core smoke (full Nemotron shapes) ──"
    reset_device
    timeout 180 $PYTHON -u $G2_SMOKE --num-heads 64 --n-groups 8 2>&1 | \
        grep -E "PASS|FAIL|overall cos|per-head cos: min" | head -10
    local result=${PIPESTATUS[0]}
    if [[ $result -ne 0 ]]; then
        echo "✗ FAIL (G2 exit $result)"
        return 1
    fi
}

echo "════════════════════════════════════════════════════"
echo "  Mamba2 SSD decode kernel — regression sweep"
echo "════════════════════════════════════════════════════"

# Single-step correctness gates (per debug_mode).
for mode in 2 3 4 5; do
    run_mode $mode || exit 1
done

# Multi-step recurrence gate.
run_replay || exit 1

# G2 full 64-head multi-core gate.
run_g2 || exit 1

echo ""
echo "════════════════════════════════════════════════════"
echo "  ✓ ALL REGRESSION TESTS PASSED"
echo "════════════════════════════════════════════════════"
