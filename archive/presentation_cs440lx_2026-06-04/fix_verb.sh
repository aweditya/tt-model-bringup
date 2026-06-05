#!/bin/bash
# Replace \verb|...| with \code{...} in poster.tex
# Some \verb snippets contain | (none currently), so this pipes match the simple case.
set -e
f="/Users/adityasriram/Labs/stanford/cs440lx/tt-model-bringup/presentation/poster.tex"
# Use perl for non-greedy match on \verb|...|
perl -i -pe 's/\\verb\|([^|]+)\|/\\code{$1}/g' "$f"
echo "done; remaining \\verb count:"
grep -c '\\verb|' "$f" || true
