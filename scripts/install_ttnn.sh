#!/usr/bin/env bash
# install_ttnn.sh — editable install of ttnn from a local tt-metal checkout.
#
# Run on a device host after `uv sync` (which provisions the venv + pyproject
# deps, including ttnn's pure-Python runtime deps). This script registers the
# `ttnn` Python package against $TT_METAL_HOME so `import ttnn` resolves to the
# rebuilt source-package — and installs the vendored tracy/ tools module that
# ttnn imports at startup.
#
# Crucially passes `--no-deps` so tt-metal's pyproject does NOT pull a fresh
# torch/transformers/etc. (which would shadow the pyproject.toml/uv.lock pins
# this repo runs against — pre-fix, this was the most likely root cause of the
# "specific versions needed" half of the clone-and-run report).
#
# Re-run this after rebuilding tt-metal (or after each `uv sync`, which prunes
# unmanaged packages).
#
#   bash scripts/install_ttnn.sh
#
# Env: TT_METAL_HOME (default ~/tenstorrent/tt-metal).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tenstorrent/tt-metal}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"

[ -d "$TT_METAL_HOME" ] || { echo "TT_METAL_HOME=$TT_METAL_HOME not a directory"; exit 1; }
[ -x "$VENV_PY" ]      || { echo "venv python not found at $VENV_PY (run uv sync first)"; exit 1; }
UV="${UV:-$HOME/.local/bin/uv}"
[ -x "$UV" ]           || { echo "uv not found at $UV"; exit 1; }

echo ">>> uv pip install setuptools_scm (ttnn's setup.py needs it)…"
"$UV" pip install --python "$VENV_PY" setuptools_scm

echo ">>> uv pip install -e $TT_METAL_HOME --no-build-isolation --no-deps"
echo "    (--no-deps preserves the torch/transformers pins from pyproject.toml + uv.lock)"
"$UV" pip install --python "$VENV_PY" --no-build-isolation --no-deps -e "$TT_METAL_HOME"

# ttnn imports `tracy.ttnn_profiler_wrapper` at startup; that submodule lives
# under tt-metal/tools/tracy and is not packaged on PyPI. Copy it into
# site-packages. (uv sync would strip it again; re-run this script after sync.)
echo ">>> copying vendored tracy tools into site-packages…"
SITE_PKGS="$("$VENV_PY" -c 'import site; print(site.getsitepackages()[0])')"
mkdir -p "$SITE_PKGS/tracy"
cp "$TT_METAL_HOME/tools/tracy/"*.py "$SITE_PKGS/tracy/"
echo "    copied $(ls "$TT_METAL_HOME/tools/tracy/"*.py | wc -l) files to $SITE_PKGS/tracy/"

echo ">>> verifying import…"
"$VENV_PY" -c "import ttnn; print(f'  ttnn at: {ttnn.__file__}'); \
  print(f'  qwen36_gdn_decode_owned available:', hasattr(ttnn.experimental, 'qwen36_gdn_decode_owned')); \
  print(f'  qwen36_decay_gate_decode_owned available:', hasattr(ttnn.experimental, 'qwen36_decay_gate_decode_owned'))"

echo ">>> install_ttnn.sh OK"
