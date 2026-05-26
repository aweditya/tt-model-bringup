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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe-mode", choices=["pattern_a_batched"],
                    default="pattern_a_batched",
                    help="Trace requires no host readback in step; pattern_a_batched "
                         "is the only trace-clean MoE path.")
    ap.add_argument("--owned-gdn", action="store_true",
                    help="Use the fused qwen36_gdn_decode_owned kernel for DN "
                         "recurrence (requires tt-metal rebuilt with the kernel).")
    args = ap.parse_args()
    log(f"bootstrap (moe_mode={args.moe_mode}, owned_gdn={args.owned_gdn})…")
    state = srv.State()
    state.moe_mode = args.moe_mode
    state.dn_owned_gdn = bool(args.owned_gdn)
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    prompt_ids = state.tokenizer.encode("The capital of France is")
    tok_id = prompt_ids[0]

    # Eager warmup: 2 forwards. JIT-during-capture hangs on Blackhole
    # (feedback_c4v4_validated), so we MUST run eager calls first to populate
    # the kernel cache.
    log("eager warmup 2 forwards…")
    next_id = srv.step_forward_ttnn(state, tok_id, 0)
    log(f"  warmup 1 → next_id={next_id}")
    state.reset_caches_ttnn()
    next_id = srv.step_forward_ttnn(state, tok_id, 0)
    log(f"  warmup 2 → next_id={next_id}")

    # Reset caches before trace capture so state is clean
    state.reset_caches_ttnn()
    # Update input buffers OUTSIDE trace capture (no host writes inside trace).
    srv.update_input_buffers(state, tok_id, 0)
    log("attempting begin_trace_capture → step_forward_inner → end_trace_capture…")
    try:
        trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
        argmax_tt = srv.step_forward_inner(state)
        ttnn.end_trace_capture(state.mesh, trace_id, cq_id=0)
        log(f"  ✓ trace capture succeeded; trace_id = {trace_id}")
        log(f"  argmax_tt shape={list(argmax_tt.shape)} dtype={argmax_tt.dtype}")
    except Exception as e:
        log(f"  ✗ trace capture FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    # Try to replay: reset state, fill input buffers, execute_trace, read argmax.
    log("attempting execute_trace…")
    try:
        state.reset_caches_ttnn()
        srv.update_input_buffers(state, tok_id, 0)
        # Time the synchronous execute_trace.
        ttnn.synchronize_device(state.mesh)
        t0 = time.time()
        ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(state.mesh)
        elapsed = (time.time() - t0) * 1000.0
        # Read argmax outside the trace.
        next_id_t = ttnn.to_torch(
            argmax_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        )
        next_id_traced = int(next_id_t.flatten()[0].item())
        log(f"  ✓ execute_trace succeeded in {elapsed:.1f} ms; next_id_traced={next_id_traced}")
        # Compare to eager argmax (from warmup 2 above which used reset state):
        log(f"  eager next_id={next_id}  traced next_id={next_id_traced}  match={'Y' if next_id == next_id_traced else 'N'}")
        # Run a few more executions to measure steady-state time.
        timings_ms = []
        for _ in range(10):
            state.reset_caches_ttnn()
            srv.update_input_buffers(state, tok_id, 0)
            ttnn.synchronize_device(state.mesh)
            t0 = time.time()
            ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(state.mesh)
            timings_ms.append((time.time() - t0) * 1000.0)
        import numpy as _np
        log(f"  execute_trace 10× iters: min {min(timings_ms):.1f} med {_np.median(timings_ms):.1f} max {max(timings_ms):.1f} ms")
    except Exception as e:
        log(f"  ✗ execute_trace FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
