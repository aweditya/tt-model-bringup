#!/usr/bin/env bash
# Quick chat-vs-raw comparison: 3 configs, max-tokens 300, max-pos 512.
# Faster than run_chat_vs_raw.sh; for first-cut signal.
set -u
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/tt-xla}"
OUT_DIR="$PROJECT_ROOT/experiments/serve/scripts/drift_runs/chat_vs_raw"
PROMPT="Implement a JSON parser combinator in Rust"
COMMON="--prompt $(printf '%q' "$PROMPT") --max-pos 512 --max-tokens 300 --seed 42"
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

# Baseline raw greedy
run "Q1_raw_greedy"
# Baseline raw + best prior combo
run "Q2_raw_dry08_rp11" --dry-multiplier 0.8 --repetition-penalty 1.1
# NEW: chat greedy (the core hypothesis)
run "Q3_chat_greedy" --chat

echo ""
echo "==================== ANALYSIS ===================="
python3 experiments/serve/scripts/count_coherent.py \
    "$OUT_DIR/Q1_raw_greedy.txt" \
    "$OUT_DIR/Q2_raw_dry08_rp11.txt" \
    "$OUT_DIR/Q3_chat_greedy.txt"
