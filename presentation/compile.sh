#!/bin/bash
# Compile poster using lualatex (gemini theme requires fontspec)
set -e
cd "$(dirname "$0")"
export PATH="/Library/TeX/texbin:$PATH"
export TEXINPUTS=".:"
lualatex -interaction=nonstopmode poster.tex
lualatex -interaction=nonstopmode poster.tex
