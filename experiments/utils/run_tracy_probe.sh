#!/usr/bin/env bash
# Run a Tracy-instrumented probe on qb1 and dump the headline ops-perf report.
#
# Usage (from ~/tt-xla on qb1):
#   bash experiments/utils/run_tracy_probe.sh <probe.py> [out_dir_name]
#
# Example:
#   bash experiments/utils/run_tracy_probe.sh \
#       experiments/utils/tracy_profile_one_moe.py tracy_one_moe
#
# Assumes one-time setup already done (see archive/superseded_research_2026-06-04/profiling-cheatsheet.md):
#   - pipx install tt-perf-report
#   - python3 experiments/utils/_patch_tracy_assertion.py
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <probe.py> [out_dir_name]" >&2
    exit 1
fi

PROBE_SCRIPT="$1"
OUT_DIR="${2:-tracy_run_$(date +%Y%m%d_%H%M%S)}"

if [ ! -f "$PROBE_SCRIPT" ]; then
    echo "probe script not found: $PROBE_SCRIPT" >&2
    exit 1
fi

# Cold-reset all four chips before any TT script (mandatory).
tt-smi -r 0,1,2,3

# Standard ttnn env (matches every other script in experiments/).
export TT_METAL_HOME=$HOME/tenstorrent/tt-metal
export TT_BUILD_DIR=$TT_METAL_HOME/build_Release
export ARCH_NAME=blackhole
export PYTHONPATH=$TT_METAL_HOME/ttnn:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}

# Tracy spawns a subprocess that calls `python3 -m tracy <script>`; put the
# venv bin dir first so the subprocess uses the venv python (where the `tracy`
# module is installed), not /usr/bin/python3.
export PATH=/home/aditya/tt-xla/.venv/bin:/home/aditya/.local/bin:$PATH

OUT_PATH=".cache/perf_logs/$OUT_DIR"
mkdir -p "$OUT_PATH"

echo "[run_tracy_probe] probe   = $PROBE_SCRIPT"
echo "[run_tracy_probe] out_dir = $OUT_PATH"

python -m tracy -r -p -v -o "$OUT_PATH" "$PROBE_SCRIPT"

# Find the generated ops_perf_results CSV (tracy writes under reports/<run_id>/).
CSV_PATH=$(find "$OUT_PATH" -name 'ops_perf_results_*.csv' -print -quit)
if [ -z "$CSV_PATH" ]; then
    echo "[run_tracy_probe] ERROR: no ops_perf_results_*.csv produced under $OUT_PATH" >&2
    exit 2
fi

echo
echo "=============================================================="
echo "[run_tracy_probe] analyze_ops_perf_results.py on:"
echo "  $CSV_PATH"
echo "=============================================================="
.venv/bin/python experiments/utils/analyze_ops_perf_results.py "$CSV_PATH"

echo
echo "[run_tracy_probe] CSV path (for manual tt-perf-report):"
echo "$CSV_PATH"
