#!/usr/bin/env python3
"""Tracy probe for Gemma 4 forward at REDUCED layer count.

The full 48-layer v2 capture (tracy_profile_one_gemma4_layer_v2.py)
overflows tracy's 12k-marker DRAM buffer, leaving cpp_device_perf_report.csv
with blank OP NAME columns — we end up only seeing host wall.

This probe sets GM4_NUM_LAYERS_OVERRIDE=N (default 4: ~2 sliding + ~2
global, or whatever the first N layer_types are) so the marker count
fits and device kernel time-per-op is recoverable. Per-layer cost should
be close enough to the full-fwd cost for #289 decisions.

Env:
    TT_GEMMA4_VARIANT={base, it}        default base
    GM4_NUM_LAYERS_OVERRIDE=N           default 4

Run (qb1, no /tmp, permanent .cache/perf_logs/):
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
  export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \\
  export TT_BUILD_DIR=$TT_METAL_HOME/build_tracy && \\
  export ARCH_NAME=blackhole && \\
  export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \\
  export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \\
  export PATH=$HOME/tt-xla/.venv/bin:$PATH && \\
  HF_HUB_OFFLINE=1 GM4_NUM_LAYERS_OVERRIDE=4 \\
    .venv/bin/python -m tracy -r -p -v \\
      -o .cache/perf_logs/tracy_gemma4_subset \\
      experiments/utils/tracy_profile_gemma4_subset.py

Then on qb1:
  .venv/bin/python experiments/utils/tracy_top_ops.py \\
      .cache/perf_logs/tracy_gemma4_subset/.logs/
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_gemma4_unified_ttnn as srv  # noqa: E402
import ttnn  # noqa: E402
import tracy  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    n_layers = int(os.environ.get("GM4_NUM_LAYERS_OVERRIDE", "4"))
    log(f"NUM_LAYERS override → {n_layers} (bootstrap loads {n_layers} layers' "
        f"weights; forward iterates {n_layers}; output is NOT correctness-valid, "
        f"this is for perf marker-fit only)")

    log("bootstrap…")
    state = srv.State()
    srv.bootstrap(state, log)

    # Single signposted forward — no warmup. Per the v2 docstring, marker
    # count stays well under 12k if NUM_LAYERS <= 8, so we get clean device
    # timing without JIT noise polluting the kernel timestamps.
    log("signposted forward…")
    srv.update_input_buffers(state, token_id=2, cur_pos=0)
    ttnn.synchronize_device(state.mesh)
    tracy.signpost("Performance pass start")
    t0 = time.time()
    out = srv.forward_token_gm4_inner(state)
    ttnn.synchronize_device(state.mesh)
    elapsed = (time.time() - t0) * 1000.0
    tracy.signpost("Performance pass end")
    log(f"one forward (eager, {n_layers} layers): {elapsed:.1f} ms")
    ttnn.deallocate(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
