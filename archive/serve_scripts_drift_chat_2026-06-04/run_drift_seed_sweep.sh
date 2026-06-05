#!/usr/bin/env bash
# Cross-seed comparison: baseline (temp+top-p) vs DRY greedy.
# Validates DRY winning across multiple seeds, not just one.
set -u
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/tt-xla}"
OUT_DIR="$PROJECT_ROOT/experiments/serve/scripts/drift_runs/seeds"
PROMPT="Implement a JSON parser combinator in Rust"
COMMON="--prompt $(printf '%q' "$PROMPT") --max-pos 512 --max-tokens 400"
PY="$PROJECT_ROOT/.venv/bin/python -m experiments.serve.client generate_long"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT" || exit 1

run() {
    local name="$1"; shift
    local out="$OUT_DIR/$name.txt"
    echo "RUN: $name (args: $*)"
    bash -c "$PY $COMMON $* > $out 2>&1" || true
    tail -1 "$out"
}

# 3 seeds × 2 configs.
for seed in 7 13 99; do
    run "baseline_s${seed}" --temperature 0.7 --top-p 0.9 --seed $seed
    run "dry08_g_s${seed}" --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 --seed $seed
done

echo ""
echo "===================== ANALYSIS ====================="
python3 experiments/serve/scripts/count_coherent.py "$OUT_DIR"/*.txt
