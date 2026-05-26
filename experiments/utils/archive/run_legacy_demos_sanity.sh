#!/usr/bin/env bash
# Run the 6 legacy 8B-era demos in sequence, capturing each to its own
# log under .cache/sanity_<DATE>/. Designed for fresh-clone sanity-checking
# on qb1 or qb2 (single-chip path, uses device 0 only — requires that any
# resident server holding the chip is stopped first).
#
# Usage:
#   bash experiments/utils/run_legacy_demos_sanity.sh
# Output:
#   .cache/sanity_<DATE>/{demoNN.log, summary.txt}
#
# Permanent (per CLAUDE.md: no inline scripts). Re-runnable from any
# fresh clone where `uv` + `ttnn` are installed.

set -u
shopt -s nullglob

DATE="${SANITY_DATE:-$(date +%Y_%m_%d)}"
LOG_DIR=".cache/sanity_${DATE}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"

DEMOS=(
  "60_native_rope_decode.py"
  "64_llama32_1b_port.py"
  "67_llama32_3b_port.py"
  "73_llama8b_instruct.py"
  "76b_8b_correctness_check.py"
  "80_8b_diverse_qa_demo.py"
)

{
  echo "Legacy 6 demos sanity run — $(date -u +%FT%TZ)"
  echo "Host: $(hostname)"
  echo "Repo HEAD: $(git rev-parse --short HEAD)  ($(git log -1 --format=%s))"
  echo "ttnn version: editable from $TT_METAL_HOME"
  echo "Log dir: $LOG_DIR"
  echo "---"
} | tee "$SUMMARY"

for demo in "${DEMOS[@]}"; do
  log="$LOG_DIR/${demo%.py}.log"
  start=$(date +%s)
  echo "=== $demo (start $(date -u +%H:%M:%SZ)) ===" | tee -a "$SUMMARY"
  if uv run python "experiments/$demo" >"$log" 2>&1; then
    status="PASS"
  else
    status="FAIL"
  fi
  elapsed=$(( $(date +%s) - start ))
  # surface the tok/s and a one-line summary if present
  perf=$(grep -hE "tok/s|tok/sec|tokens/s|tok per second|cosine|cos " "$log" | tail -3 | tr '\n' ' | ')
  echo "  $status in ${elapsed}s — $perf" | tee -a "$SUMMARY"
done

echo "---" | tee -a "$SUMMARY"
echo "DONE — $(date -u +%FT%TZ)" | tee -a "$SUMMARY"
