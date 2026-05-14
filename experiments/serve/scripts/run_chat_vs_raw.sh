#!/usr/bin/env bash
# Compare chat-templated prompt vs raw prompt on the long-context drift case.
# Runs four configurations against the persistent server on qb1.
#
# Usage:
#   bash experiments/serve/scripts/run_chat_vs_raw.sh
#
# Output: experiments/serve/scripts/drift_runs/chat_vs_raw/<name>.txt
# Then prints count_coherent analysis.
set -u
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/tt-xla}"
OUT_DIR="$PROJECT_ROOT/experiments/serve/scripts/drift_runs/chat_vs_raw"
PROMPT="Implement a JSON parser combinator in Rust"
# Reuse the same prompt the prior drift sweep used so coherent-char counts
# are directly comparable to drift_runs/{baseline_temp_topp,dry08_greedy,
# all_combined,_final_winner}.txt
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

# A. Baseline: raw prompt, greedy (matches our worst prior baseline)
run "A_raw_greedy"

# B. Raw prompt + best prior config (DRY 0.8 + rep_pen 1.1)
run "B_raw_dry08_rp11" --dry-multiplier 0.8 --repetition-penalty 1.1

# C. NEW: chat template, greedy
run "C_chat_greedy" --chat

# D. NEW: chat template + HF-recommended thinking sampling
#    (temperature=1.0 top_p=0.95 per HF model card for thinking mode)
run "D_chat_thinking_sampler" --chat --temperature 1.0 --top-p 0.95

# E. NEW: chat template + DRY+rp combo (drift-resistant + proper instruct framing)
run "E_chat_dry08_rp11" --chat --dry-multiplier 0.8 --repetition-penalty 1.1

# F. NEW: chat template + non-thinking sampler (HF-recommended for instruct)
#    The model still defaults to thinking mode due to chat template forcing <think>\n,
#    but the sampling profile is the non-thinking one (presence_penalty=1.5 mapped
#    to repetition_penalty since our sampler doesn't have presence)
run "F_chat_instruct_sampler" --chat --temperature 0.7 --top-p 0.8 --repetition-penalty 1.1

echo ""
echo "==================== ANALYSIS ===================="
python3 experiments/serve/scripts/count_coherent.py \
    "$OUT_DIR/A_raw_greedy.txt" \
    "$OUT_DIR/B_raw_dry08_rp11.txt" \
    "$OUT_DIR/C_chat_greedy.txt" \
    "$OUT_DIR/D_chat_thinking_sampler.txt" \
    "$OUT_DIR/E_chat_dry08_rp11.txt" \
    "$OUT_DIR/F_chat_instruct_sampler.txt"
