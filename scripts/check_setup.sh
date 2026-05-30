#!/usr/bin/env bash
# check_setup.sh — sanity-check a device host BEFORE booting the server.
#
# Each check is independent and prints PASS/FAIL with a one-line hint. Does
# NOT open the device (no `ttnn.open_mesh_device` — that lives in the server
# bootstrap). Exits non-zero if any check fails so CI can gate on it.
#
#   bash scripts/check_setup.sh                  # full check
#   bash scripts/check_setup.sh --skip-hf        # skip the HF auth probe
#
# Run this on qb1/qb2 after `uv sync && bash scripts/install_ttnn.sh` and
# before `bash experiments/serve/scripts/serve_cb.sh start`.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tenstorrent/tt-metal}"
TT_BUILD_DIR="${TT_BUILD_DIR:-$TT_METAL_HOME/build_Release}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"
SKIP_HF=0
[[ "${1:-}" == "--skip-hf" ]] && SKIP_HF=1

fail=0
check() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "  ✓ $name"
    else
        echo "  ✗ $name"
        "$@" 2>&1 | sed 's/^/      /' | head -5
        fail=$((fail + 1))
    fi
}

echo "=== check_setup.sh ($(date '+%Y-%m-%d %H:%M:%S')) ==="

echo "[1] venv + uv installed"
check "venv python exists ($VENV_PY)"     test -x "$VENV_PY"
check "uv installed at ~/.local/bin/uv"   test -x "$HOME/.local/bin/uv"

echo "[2] tt-metal source tree"
check "TT_METAL_HOME exists ($TT_METAL_HOME)"    test -d "$TT_METAL_HOME"
check "TT_BUILD_DIR exists ($TT_BUILD_DIR)"       test -d "$TT_BUILD_DIR"
check "tt-metal SHA matches tt-metal-sha.txt" bash -c '
    pinned=$(grep -E "^[0-9a-f]{40}$" '"$PROJECT_ROOT"'/tt-metal-sha.txt | head -1)
    actual=$(git -C '"$TT_METAL_HOME"' rev-parse HEAD 2>/dev/null)
    [ "$pinned" = "$actual" ]
'

echo "[3] ttnn + owned kernels resolve"
export TT_METAL_HOME TT_BUILD_DIR
export ARCH_NAME="${ARCH_NAME:-blackhole}"
export PYTHONPATH="$TT_METAL_HOME/ttnn:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
check "import ttnn"                          "$VENV_PY" -c 'import ttnn'
check "ttnn.experimental.qwen36_gdn_decode_owned" \
    "$VENV_PY" -c 'import ttnn; assert hasattr(ttnn.experimental, "qwen36_gdn_decode_owned"), "missing — run scripts/build_owned_ops.sh"'
check "ttnn.experimental.qwen36_decay_gate_decode_owned" \
    "$VENV_PY" -c 'import ttnn; assert hasattr(ttnn.experimental, "qwen36_decay_gate_decode_owned"), "missing — run scripts/build_owned_ops.sh"'

echo "[4] device visible to tt-smi"
check "tt-smi available"        command -v tt-smi
# `tt-smi -s` emits JSON; count `device_info[]` rather than grep an unstable string.
check "tt-smi reports >0 devices" bash -c '
    n=$(tt-smi -s 2>/dev/null | "'"$VENV_PY"'" -c "import sys,json; print(len(json.load(sys.stdin).get(\"device_info\", [])))")
    [ "$n" -gt 0 ]
'

if [ "$SKIP_HF" = 0 ]; then
    echo "[5] HuggingFace access (Qwen3.6 weights)"
    # The real gate is "can we fetch the config.json?"; that subsumes the token
    # check (anonymous fetch works for currently-public models, gated → 401).
    # We surface token presence as INFO, not a fail.
    if [ -n "${HF_TOKEN:-}" ] || [ -s "$HOME/.cache/huggingface/token" ]; then
        echo "  ✓ HF token present (env or ~/.cache/huggingface/token)"
    else
        echo "  i  HF token not configured — anonymous fetches only (rate-limited; gated models will 401)."
        echo "       Fix: \`uv run hf auth login\` (one-time)."
    fi
    check "Qwen/Qwen3.6-27B config.json reachable" \
        "$VENV_PY" -c 'from huggingface_hub import hf_hub_download; hf_hub_download("Qwen/Qwen3.6-27B", "config.json")'
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "=== ALL CHECKS PASSED ==="
    exit 0
else
    echo "=== $fail CHECK(S) FAILED ==="
    exit 1
fi
