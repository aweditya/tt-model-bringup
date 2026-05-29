#!/usr/bin/env bash
# Phase D (revised) — Install ttnn from prebuilt wheel on PyPI.
# tt-metal v0.69.0 publishes ttnn==0.69.0 on PyPI, so we skip the source build.
# The tt-metal source tree remains at ~/tenstorrent/tt-metal for reference / models.
# Run via: ssh qb1 'bash -s' < pjrt_plugin/scripts/qb1_phase_d_install_ttnn.sh
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

cd ~/tt-xla

echo "[phase-d] Installing ttnn==0.69.0 from PyPI..."
uv pip install --python .venv/bin/python ttnn==0.69.0

echo "[phase-d] Verification — import ttnn, JAX, and basic sanity:"
.venv/bin/python - <<'PY'
import sys, importlib
mods = ['ttnn', 'jax', 'jaxlib', 'numpy', 'torch']
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, '__version__', 'n/a')
        print(f"  {m:10s} = {ver}")
    except Exception as e:
        print(f"  {m:10s} FAILED: {e}")
PY

echo "[phase-d] Done."
