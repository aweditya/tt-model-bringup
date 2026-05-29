#!/usr/bin/env python3
"""CB4 — TRACED batched decode: correctness + throughput at fixed B.

Eager batching is ~free (cb_bench_throughput.py: B=32 step only +2.5% vs B=1),
but eager is dispatch-bound (~252 ms/step). Tracing amortizes per-op Python
dispatch so per-step time drops toward pure compute. This captures one decode
trace of the batched forward at a fixed B (vLLM CUDA-graph pattern) and:

  1. CORRECTNESS — traced B=1, teacher-forced, argmax must match the production
     B=1 reference (proves execute_trace threads the autoregressive DN/KV state).
  2. THROUGHPUT — execute_trace ms/step + aggregate tok/s at B in {1,8,32}.

Trace pattern mirrors production server_tp._ensure_decode_trace: warmup eager
forwards (JIT all kernels — capturing during JIT hangs on Blackhole), then
begin/end_trace_capture around forward_batch_tp_inner (reads only pre-allocated
buffers). Per step: update_input_buffers_batched (host, OUTSIDE trace) +
execute_trace. State (cb_dn ssm/conv via ttnn.copy, cb_kv via paged_update_cache)
mutates in-place so the trace threads it.

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb_bench_trace.py --batches 1,8,32 --steps 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base       # noqa: E402
import server_tp_cb as cb      # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prod_ref(state, prompt_ids):
    """Production B=1 argmax sequence (the correctness reference)."""
    import ttnn
    ids = []
    for pos, t in enumerate(prompt_ids):
        base.update_input_buffers(state, int(t), pos)
        am = base.forward_token_tp_inner(state)
        ids.append(int(ttnn.to_torch(
            am, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)).flatten()[0]))
        ttnn.deallocate(am)
    return ids


def capture_trace(state, B, blocks_per_seq):
    """setup_cb_state(B) + JIT warmup + capture forward_batch_tp_inner trace.
    Returns (trace_id, argmax_handle). Leaves cb state dirty (warmup ran)."""
    import ttnn
    cb.setup_cb_state(state, B, blocks_per_seq=blocks_per_seq)
    cb.cb_reset_states(state)
    tok = 760
    for i in range(2):  # JIT all kernels eagerly first (capture-during-JIT hangs)
        cb.update_input_buffers_batched(state, [tok] * B, [i] * B)
        am = cb.forward_batch_tp_inner(state); ttnn.deallocate(am)
    ttnn.synchronize_device(state.mesh)
    cb.update_input_buffers_batched(state, [tok] * B, [2] * B)
    tid = ttnn.begin_trace_capture(state.mesh, cq_id=0)
    argmax_handle = cb.forward_batch_tp_inner(state)
    ttnn.end_trace_capture(state.mesh, tid, cq_id=0)
    return tid, argmax_handle


def traced_correctness(state, prompt_ids, ref, blocks_per_seq):
    """Capture B=1 trace, then teacher-force the prompt via execute_trace and
    compare argmax to the production reference (validates state threading)."""
    import ttnn
    tid, am = capture_trace(state, 1, blocks_per_seq)
    cb.cb_reset_states(state)  # fresh autoregressive state for the real run
    got = []
    for pos, t in enumerate(prompt_ids):
        cb.update_input_buffers_batched(state, [int(t)], [pos])
        ttnn.execute_trace(state.mesh, tid, cq_id=0, blocking=False)
        got.append(int(ttnn.to_torch(
            am, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)).flatten()[0]))
    ttnn.synchronize_device(state.mesh)
    ttnn.release_trace(state.mesh, tid)
    ok = (got == ref)
    log(f"  traced B=1 argmax: {got}")
    log(f"  prod   B=1 argmax: {ref}")
    log(f"  traced-correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def bench_traced(state, B, steps, warmup, blocks_per_seq):
    import ttnn
    import time as _t
    tid, am = capture_trace(state, B, blocks_per_seq)
    tok = 760
    # warm the trace
    for i in range(warmup):
        cb.update_input_buffers_batched(state, [tok] * B, [3 + i] * B)
        ttnn.execute_trace(state.mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(state.mesh)

    # execute_trace only (amortized compute; inputs fixed)
    t0 = _t.perf_counter()
    for _ in range(steps):
        ttnn.execute_trace(state.mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(state.mesh)
    exec_ms = (_t.perf_counter() - t0) / steps * 1000.0

    # full step: host buffer update + execute_trace (the production timed region)
    t0 = _t.perf_counter()
    for i in range(steps):
        cb.update_input_buffers_batched(state, [tok] * B, [3 + warmup + i] * B)
        ttnn.execute_trace(state.mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(state.mesh)
    full_ms = (_t.perf_counter() - t0) / steps * 1000.0

    ttnn.release_trace(state.mesh, tid)
    for li in list(state.cb_kv.keys()):  # free big KV pool before next B
        for k in ('kc', 'vc'):
            try: ttnn.deallocate(state.cb_kv[li][k])
            except Exception: pass
    return exec_ms, full_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--blocks-per-seq", type=int, default=8)
    ap.add_argument("--owned-gdn", action="store_true",
                    help="use the batched owned_gdn DN recurrence kernel")
    ap.add_argument("--shiftacc", action="store_true",
                    help="use the shift-accumulate conv1d (DNK-G4)")
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]

    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    state.deltanet_recurrence_mode = "manual"
    state.deltanet_decay_gate_mode = "manual"
    state.deltanet_decay_mode = "native_softplus"
    state.cb_dn_recurrence_mode = "owned_gdn" if args.owned_gdn else "manual"
    state.cb_conv_mode = "shiftacc" if args.shiftacc else "kdim"
    log(f"DN recurrence: {state.cb_dn_recurrence_mode}; conv: {state.cb_conv_mode}")

    prompt_ids = state.tok.encode("The capital of France is the city of")[:6]
    log("=== production B=1 reference ===")
    ref = prod_ref(state, prompt_ids)

    log("=== traced B=1 correctness (state threading under execute_trace) ===")
    ok = traced_correctness(state, prompt_ids, ref, args.blocks_per_seq)

    log(f"=== traced throughput (steps={args.steps}, warmup={args.warmup}) ===")
    base_exec = None
    rows = []
    for B in batches:
        e_ms, f_ms = bench_traced(state, B, args.steps, args.warmup, args.blocks_per_seq)
        if base_exec is None and B == 1:
            base_exec = e_ms
        agg = B * 1000.0 / e_ms
        rows.append((B, e_ms, f_ms, agg))
        log(f"  B={B:3d}: execute {e_ms:7.2f} ms  full-step {f_ms:7.2f} ms  "
            f"agg {agg:8.2f} tok/s"
            + (f"  ({e_ms/base_exec:.2f}x B=1 exec, {agg/(1000.0/base_exec):.1f}x tput)"
               if base_exec else ""))
    log("=== summary (TRACED; DN=manual, owned_gdn would speed B=1 further) ===")
    log(f"  traced-correctness: {'PASS' if ok else 'FAIL'}")
    for B, e_ms, f_ms, agg in rows:
        log(f"  B={B:3d}  execute {e_ms:7.2f} ms/step  agg {agg:8.2f} tok/s")


if __name__ == "__main__":
    main()
