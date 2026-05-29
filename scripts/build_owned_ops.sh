#!/usr/bin/env bash
# Install the owned TT-NN ops into a tt-metal checkout and rebuild ttnn.
# RUN THIS ON THE TT HOST (qb1/qb2) — it compiles tt-metal; it does not open a
# device. One-time setup after cloning + building tt-metal. See
# experiments/owned_ops/README.md for which ops are which.
#
#   scripts/build_owned_ops.sh [--all] [--dry-run] [--no-build] [op ...]
#
#   (no args)   install the production 27B ops (gdn_decode_owned, decay_gate)
#   --all       install every op under experiments/owned_ops/
#   <op ...>    install just the named op dir(s)
#   --dry-run   show what integrate would do; skip the rebuild
#   --no-build  integrate sources only; skip the ttnn rebuild + .so copy
#
# Env: TT_METAL (default ~/tenstorrent/tt-metal), BUILD_DIR
#      (default $TT_METAL/build_Release — the dir scripts/run_remote.sh uses).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPS_DIR="$REPO_ROOT/experiments/owned_ops"
TT_METAL="${TT_METAL:-$HOME/tenstorrent/tt-metal}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL/build_Release}"

PROD_OPS=(qwen36_gdn_decode_owned qwen36_decay_gate_decode_owned)

dry_run=0; no_build=0; all=0; ops=()
for a in "$@"; do
  case "$a" in
    --dry-run)  dry_run=1 ;;
    --no-build) no_build=1 ;;
    --all)      all=1 ;;
    --*)        echo "unknown flag: $a" >&2; exit 2 ;;
    *)          ops+=("$a") ;;
  esac
done

if [[ "$all" == 1 ]]; then
  ops=()
  for d in "$OPS_DIR"/*/; do [[ -f "$d/integrate_into_ttmetal.py" ]] && ops+=("$(basename "$d")"); done
elif [[ "${#ops[@]}" -eq 0 ]]; then
  ops=("${PROD_OPS[@]}")
fi

[[ -d "$TT_METAL" ]] || { echo "tt-metal not found at $TT_METAL (set TT_METAL=)" >&2; exit 1; }

echo "tt-metal: $TT_METAL"
echo "ops:      ${ops[*]}"
dr=(); [[ "$dry_run" == 1 ]] && dr=(--dry-run)
for op in "${ops[@]}"; do
  script="$OPS_DIR/$op/integrate_into_ttmetal.py"
  [[ -f "$script" ]] || { echo "no such op: $op ($script)" >&2; exit 1; }
  echo "=== integrating $op ==="
  python3 "$script" --tt-metal "$TT_METAL" "${dr[@]}"
done

if [[ "$dry_run" == 1 || "$no_build" == 1 ]]; then
  echo "skipping ttnn rebuild ($([[ $dry_run == 1 ]] && echo --dry-run || echo --no-build))"
  exit 0
fi

echo "=== rebuilding ttnn ($BUILD_DIR) ==="
cmake --build "$BUILD_DIR" --target ttnn -j"$(nproc 2>/dev/null || echo 8)"
# Refresh the source-package extensions so `PYTHONPATH=$TT_METAL/ttnn` picks up
# the rebuilt op (otherwise a stale wheel can shadow it).
cp "$BUILD_DIR/ttnn/_ttnn.so"    "$TT_METAL/ttnn/ttnn/_ttnn.so"
cp "$BUILD_DIR/ttnn/_ttnncpp.so" "$TT_METAL/ttnn/ttnn/_ttnncpp.so"
echo "done — ${#ops[@]} op(s) installed + ttnn rebuilt"
