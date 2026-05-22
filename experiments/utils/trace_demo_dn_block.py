#!/usr/bin/env python3
"""B17 trace validation — capture and replay ONE DN block forward.

Why DN: it has no data-dependent control flow (unlike MoE's Python loop
over TOP_K=8 expert indices) and uses fixed-shape state (conv_state +
recurrent_state). So it's the cleanest target to validate trace infra
on our (1,4) mesh.

Method:
  1. Bootstrap the on-device server
  2. Pre-allocate input h_tt + reuse the layer 0 DN cache
  3. Warmup 2 eager dn_forward_ttnn (per 27B note: JIT during capture hangs)
  4. begin_trace_capture → dn_forward_ttnn(h_tt, w_L0, ...) → end_trace_capture
  5. Benchmark eager vs traced over N iters
  6. Report speedup

Expectation from 27B (commit C'7.6.1, see feedback_c761_tp_trace_wins_big.md):
  eager TP 7.01 ms/block → traced TP 1.34 ms/block = 5.23× speedup

Run (qb1):
  cd ~/tt-xla && tt-smi -r && \\
    export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \\
    export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \\
    export ARCH_NAME=blackhole && \\
    export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \\
    export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \\
    .venv/bin/python -u experiments/utils/trace_demo_dn_block.py
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

N_ITERS = 50


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("bootstrap…")
    state = srv.State()
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    # Build a real h_tt input (use the prompt's first-token embed via embed lookup)
    prompt_ids = state.tokenizer.encode("The capital of France is")
    tok_id = prompt_ids[0]
    tok_idx_tt = ttnn.from_torch(
        torch.from_numpy(np.array([[tok_id]], dtype=np.int32)),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    embed_out = ttnn.embedding(tok_idx_tt, state.embed_tt)
    ttnn.deallocate(tok_idx_tt)
    h_tt = ttnn.to_layout(embed_out, ttnn.TILE_LAYOUT)
    ttnn.deallocate(embed_out)
    h_norm_1 = ttnn.rms_norm(h_tt, weight=state.per_layer_tt[0]["input_layernorm"],
                              epsilon=srv.EPS)
    ttnn.deallocate(h_tt)

    log(f"input h_norm_1 ready: shape={list(h_norm_1.shape)}")

    # ── Eager benchmark ────────────────────────────────────────────────
    # Warmup
    log("eager warmup 3 iters (JIT compile)…")
    dn_state = state.dn_caches_tt[0]
    for _ in range(3):
        out, new_conv, new_rec = srv.dn_forward_ttnn(
            h_norm_1, state.per_layer_tt[0], state.mesh, dn_state
        )
        ttnn.deallocate(out)
        # Don't update caches — keep input the same; the new state tensors leak but only ~50× iters
        if new_conv is not dn_state[0]:
            ttnn.deallocate(new_conv)
        if new_rec is not dn_state[1]:
            ttnn.deallocate(new_rec)
    ttnn.synchronize_device(state.mesh)
    log("  warmup done.")

    log(f"eager benchmark {N_ITERS} iters…")
    eager_ms = []
    for i in range(N_ITERS):
        ttnn.synchronize_device(state.mesh)
        t0 = time.time()
        out, new_conv, new_rec = srv.dn_forward_ttnn(
            h_norm_1, state.per_layer_tt[0], state.mesh, dn_state
        )
        ttnn.synchronize_device(state.mesh)
        eager_ms.append((time.time() - t0) * 1000.0)
        ttnn.deallocate(out)
        if new_conv is not dn_state[0]:
            ttnn.deallocate(new_conv)
        if new_rec is not dn_state[1]:
            ttnn.deallocate(new_rec)
    eager_med = float(np.median(eager_ms))
    log(f"  eager: median {eager_med:.2f} ms  (min {min(eager_ms):.2f}, max {max(eager_ms):.2f})")

    # ── Trace capture ──────────────────────────────────────────────────
    log("trace capture: begin_trace_capture → dn_forward_ttnn → end_trace_capture…")
    t0 = time.time()
    trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
    # IMPORTANT: the captured op should write into PRE-ALLOCATED output slots.
    # For this demo, we don't reuse the output, so just capture the call.
    out_traced, new_conv_traced, new_rec_traced = srv.dn_forward_ttnn(
        h_norm_1, state.per_layer_tt[0], state.mesh, dn_state
    )
    ttnn.end_trace_capture(state.mesh, trace_id, cq_id=0)
    log(f"  capture took {(time.time() - t0) * 1000:.0f} ms, trace_id={trace_id}")

    # ── Traced benchmark ───────────────────────────────────────────────
    log(f"traced benchmark {N_ITERS} iters (execute_trace)…")
    traced_ms = []
    for i in range(N_ITERS):
        ttnn.synchronize_device(state.mesh)
        t0 = time.time()
        ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(state.mesh)
        traced_ms.append((time.time() - t0) * 1000.0)
    traced_med = float(np.median(traced_ms))
    log(f"  traced: median {traced_med:.2f} ms  (min {min(traced_ms):.2f}, max {max(traced_ms):.2f})")

    # ── Summary ────────────────────────────────────────────────────────
    speedup = eager_med / traced_med
    log("")
    log(f"=== B17 trace validation result ===")
    log(f"  eager DN block:  {eager_med:.2f} ms")
    log(f"  traced DN block: {traced_med:.2f} ms")
    log(f"  speedup:         {speedup:.2f}×")
    log(f"  (27B reference: 5.23× per feedback_c761_tp_trace_wins_big.md)")

    # Cleanup
    ttnn.release_trace(state.mesh, trace_id)
    ttnn.close_mesh_device(state.mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
