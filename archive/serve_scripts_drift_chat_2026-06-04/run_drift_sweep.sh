#!/usr/bin/env bash
# Sweep new long-context samplers against the standard drift prompt.
# Runs against the persistent server on qb1.
#
# Usage:
#   bash experiments/serve/scripts/run_drift_sweep.sh
#
# Output: experiments/serve/scripts/drift_runs/<run_name>.txt for each config,
# then prints count_coherent analysis at the end.
set -u
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/tt-xla}"
OUT_DIR="$PROJECT_ROOT/experiments/serve/scripts/drift_runs"
PROMPT="Implement a JSON parser combinator in Rust"
COMMON="--prompt $(printf '%q' "$PROMPT") --max-pos 512 --max-tokens 400 --seed 42"
PY="$PROJECT_ROOT/.venv/bin/python -m experiments.serve.client generate_long"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT" || exit 1

run() {
    local name="$1"; shift
    local out="$OUT_DIR/$name.txt"
    echo "=========================================================="
    echo "RUN: $name"
    echo "  args: $*"
    echo "  out:  $out"
    echo "=========================================================="
    bash -c "$PY $COMMON $* > $out 2>&1" || true
    tail -3 "$out"
    echo ""
}

# Test A: greedy + n-gram only (most surgical)
run "ngram4_greedy" --no-repeat-ngram-size 4

# Test B: temp+top-p + n-gram (combined with prior baseline)
run "ngram4_temp_topp" --no-repeat-ngram-size 4 --temperature 0.7 --top-p 0.9

# Test C: greedy + DRY only
run "dry08_greedy" --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2

# Test D: kitchen-sink — temp+top-p + n-gram + DRY
run "all_combined" --no-repeat-ngram-size 4 --dry-multiplier 0.8 \
                   --temperature 0.7 --top-p 0.9

echo ""
echo "==================== ANALYSIS ===================="
python3 experiments/serve/scripts/count_coherent.py \
    "$OUT_DIR/ngram4_greedy.txt" \
    "$OUT_DIR/ngram4_temp_topp.txt" \
    "$OUT_DIR/dry08_greedy.txt" \
    "$OUT_DIR/all_combined.txt"
