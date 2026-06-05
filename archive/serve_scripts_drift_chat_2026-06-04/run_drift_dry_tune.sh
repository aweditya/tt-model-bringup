#!/usr/bin/env bash
# Tune DRY parameters to push collapse later.
set -u
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/tt-xla}"
OUT_DIR="$PROJECT_ROOT/experiments/serve/scripts/drift_runs/dry_tune"
PROMPT="Implement a JSON parser combinator in Rust"
COMMON="--prompt $(printf '%q' "$PROMPT") --max-pos 512 --max-tokens 400 --seed 42"
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

# Allowed length tweaks — lower = bans shorter repeats
run "dry_m08_a2" --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2  # baseline winner
run "dry_m08_a3" --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 3
run "dry_m15_a2" --dry-multiplier 1.5 --dry-base 1.75 --dry-allowed-length 2  # stronger
run "dry_m05_a2" --dry-multiplier 0.5 --dry-base 1.75 --dry-allowed-length 2  # weaker

# Combine: DRY + repetition_penalty (the canonical pair from llama.cpp default chain)
run "dry_m08_rep11" --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 --repetition-penalty 1.1
run "dry_m08_rep12" --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 --repetition-penalty 1.2

echo ""
echo "===================== ANALYSIS ====================="
python3 experiments/serve/scripts/count_coherent.py "$OUT_DIR"/*.txt
