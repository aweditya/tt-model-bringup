#!/usr/bin/env python3
"""Tracy-instrumented profile of a single MoE call (Pattern A looped).

Why: profiling the full 5120-op forward overflows the Tracy DRAM marker
buffer (12000 markers per RISCV per core). The MoE structure repeats
identically across all 40 layers, so one MoE call gives us the same
information about WHICH ops eat time.

Captures:
  - Pattern A MoE on layer 0's weights with synthetic input
  - Signposts around the single MoE call
  - tt-perf-report on the output answers: are matmuls dispatch-bound
    (high op-to-op gap, low kernel time) or BW/compute-bound?

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
  export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \\
  export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \\
  export ARCH_NAME=blackhole && \\
  export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \\
  export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \\
  export PATH=/home/aditya/tt-xla/.venv/bin:$PATH && \\
  .venv/bin/python -m tracy -r -p -v -o .cache/perf_logs/tracy_one_moe \\
      experiments/utils/tracy_profile_one_moe.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402
import ttnn  # noqa: E402
import tracy  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("bootstrap (moe_mode=pattern_a)…")
    state = srv.State()
    state.moe_mode = "pattern_a"
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    # Synthesize input h: shape [1, HIDDEN] replicated, post-rms-norm magnitude.
    rng = np.random.default_rng(0)
    h_np = rng.normal(0, 5.0, size=(1, srv.HIDDEN)).astype(np.float32)
    h_tt = srv.np_to_replicated(h_np, state.mesh)
    log(f"h_tt shape={list(h_tt.shape)} dtype={h_tt.dtype}")

    # Profile the production MoE path. Swap moe_fn between the looped
    # (moe_forward_ttnn_pattern_a) and batched (moe_forward_ttnn_pattern_a_batched)
    # variants depending on which one you're optimizing.
    moe_fn = srv.moe_forward_ttnn_pattern_a_batched

    log("warmup 2 MoE calls…")
    for _ in range(2):
        out = moe_fn(h_tt, state.per_layer_tt[0], state.mesh)
        ttnn.deallocate(out)
    ttnn.synchronize_device(state.mesh)

    log("signposted MoE call…")
    ttnn.synchronize_device(state.mesh)
    tracy.signpost("Performance pass start")
    t0 = time.time()
    out = moe_fn(h_tt, state.per_layer_tt[0], state.mesh)
    ttnn.synchronize_device(state.mesh)
    elapsed = (time.time() - t0) * 1000.0
    tracy.signpost("Performance pass end")
    log(f"one MoE call: {elapsed:.1f} ms")
    ttnn.deallocate(out)
    ttnn.deallocate(h_tt)


if __name__ == "__main__":
    main()
