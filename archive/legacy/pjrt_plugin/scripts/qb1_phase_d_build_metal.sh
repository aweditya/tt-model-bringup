#!/usr/bin/env bash
# Phase D — Build tt-metal v0.69.0 (LONG: 30-60 min).
# Run via: ssh qb1 'bash -s' < pjrt_plugin/scripts/qb1_phase_d_build_metal.sh
# Or for background: ssh qb1 'nohup bash ~/tt-xla/pjrt_plugin/scripts/qb1_phase_d_build_metal.sh > ~/tt-xla/.cache/build_metal.log 2>&1 &'
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export TT_METAL_HOME="$HOME/tenstorrent/tt-metal"

mkdir -p ~/tt-xla/.cache

cd "$TT_METAL_HOME"

# Try the simpler build path first. tt-metal v0.69.0 should have build_metal.sh
if [ -x ./build_metal.sh ]; then
  echo "[phase-d] Running ./build_metal.sh (logs → ~/tt-xla/.cache/build_metal.log)"
  ./build_metal.sh 2>&1 | tee ~/tt-xla/.cache/build_metal.log
else
  echo "[phase-d] No build_metal.sh; trying CMake directly..."
  cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ 2>&1 | tee ~/tt-xla/.cache/build_metal.log
  cmake --build build 2>&1 | tee -a ~/tt-xla/.cache/build_metal.log
fi

# After build, install ttnn into our project venv via a .pth file pointing at metal's python module
PROJECT_VENV="$HOME/tt-xla/.venv"
SITE_PACKAGES=$("$PROJECT_VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")

echo "[phase-d] Linking tt-metal's ttnn into project venv site-packages..."
cat > "$SITE_PACKAGES/tt_metal.pth" <<EOF
$TT_METAL_HOME
$TT_METAL_HOME/ttnn
EOF

# Sanity check: can we import ttnn from the project venv?
echo "[phase-d] Verifying ttnn import..."
"$PROJECT_VENV/bin/python" -c "
import sys
sys.path.insert(0, '$TT_METAL_HOME')
sys.path.insert(0, '$TT_METAL_HOME/ttnn')
import ttnn
print('ttnn imported OK; module path:', ttnn.__file__)
" || echo "[phase-d] WARNING: ttnn import failed — see log"

echo "[phase-d] Done."
