#!/usr/bin/env python3
"""B17 — try to capture a full step_forward_ttnn trace as-is.

Expectation: this will likely FAIL or capture wrong things because:
  - step_forward_ttnn has host writes per call (tok_id upload, cos/sin
    upload, MoE top_idxs/weights readback)
  - KV cache concat changes shape per position
  - MoE Python loop branches on host-read indices

But it's useful to confirm what specifically breaks so we know the
scope of the B17 refactor.

Method:
  1. Bootstrap
  2. Warmup 2 eager step_forward_ttnn calls (JIT compile)
  3. Try begin_trace_capture → step_forward_ttnn → end_trace_capture
  4. If capture succeeds: try execute_trace; compare results
  5. Otherwise: report which op / line breaks

Run (qb1):
  cd ~/tt-xla && tt-smi -r && \\
    export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \\
    export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \\
    export ARCH_NAME=blackhole && \\
    export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \\
    export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \\
    .venv/bin/python -u experiments/utils/trace_demo_full_step.py
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402
import ttnn  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("bootstrap…")
    state = srv.State()
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    prompt_ids = state.tokenizer.encode("The capital of France is")
    tok_id = prompt_ids[0]

    log("eager warmup 2 forwards…")
    state.reset_caches_ttnn()
    next_id = srv.step_forward_ttnn(state, tok_id, 0)
    log(f"  warmup 1 → next_id={next_id}")
    state.reset_caches_ttnn()
    next_id = srv.step_forward_ttnn(state, tok_id, 0)
    log(f"  warmup 2 → next_id={next_id}")

    # Reset caches before trace capture so state is clean
    state.reset_caches_ttnn()
    log("attempting begin_trace_capture → step_forward_ttnn → end_trace_capture…")
    try:
        trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
        next_id_traced = srv.step_forward_ttnn(state, tok_id, 0)
        ttnn.end_trace_capture(state.mesh, trace_id, cq_id=0)
        log(f"  ✓ trace capture succeeded; next_id during capture = {next_id_traced}")
        log(f"  trace_id = {trace_id}")
    except Exception as e:
        log(f"  ✗ trace capture FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    # Try to replay
    log("attempting execute_trace…")
    try:
        state.reset_caches_ttnn()
        t0 = time.time()
        ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=True)
        elapsed = (time.time() - t0) * 1000.0
        log(f"  ✓ execute_trace succeeded in {elapsed:.1f} ms")
    except Exception as e:
        log(f"  ✗ execute_trace FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
