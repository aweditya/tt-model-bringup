#!/usr/bin/env bash
# Quick chat-vs-raw comparison: 4 configs, max-tokens 600, max-pos 1024.
# Long enough that raw-mode drift hits.
set -u
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/tt-xla}"
OUT_DIR="$PROJECT_ROOT/experiments/serve/scripts/drift_runs/chat_vs_raw"
PROMPT="Implement a JSON parser combinator in Rust"
COMMON="--prompt $(printf '%q' "$PROMPT") --max-pos 1024 --max-tokens 600 --seed 42"
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

# A. Baseline raw greedy (will drift ~step 130)
run "Q1_raw_greedy"
# B. Best prior raw config (DRY 0.8 + rep_pen 1.1; ~200 coherent tokens)
run "Q2_raw_dry08_rp11" --dry-multiplier 0.8 --repetition-penalty 1.1
# C. NEW: chat greedy (the core hypothesis)
run "Q3_chat_greedy" --chat
# D. NEW: chat + DRY + rp (combining both)
run "Q4_chat_dry08_rp11" --chat --dry-multiplier 0.8 --repetition-penalty 1.1

echo ""
echo "==================== ANALYSIS ===================="
python3 experiments/serve/scripts/count_coherent.py \
    "$OUT_DIR/Q1_raw_greedy.txt" \
    "$OUT_DIR/Q2_raw_dry08_rp11.txt" \
    "$OUT_DIR/Q3_chat_greedy.txt" \
    "$OUT_DIR/Q4_chat_dry08_rp11.txt"
