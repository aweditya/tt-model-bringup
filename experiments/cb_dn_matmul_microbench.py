#!/usr/bin/env python3
"""DNK-G3 profile — isolate the DN in_proj / out_proj matmuls (traced).

After owned_gdn fused the recurrence, the DN block is still ~148 ms of the 190 ms
B=32 step (slope 3.61 ms/seq). The recurrence is now small, so the remainder is
the surrounding ops — chiefly the two matmuls (in_proj [B,HIDDEN]×W_in and
out_proj [B,VAL_DIM_CHIP]×W_out + all_reduce), whose FLOPs ∝ B. This microbench
times those two matmuls in isolation (traced) at B=1/32/64 with the REAL DN
weights, to see their contribution to the slope and whether an explicit
core_grid (the A004 lever: −30 ms on the 35B MoE matmul) would help.

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb_dn_matmul_microbench.py --batches 1,32,64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base       # noqa: E402
import server_tp_cb as cb      # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def repl(mesh, shape):
    import ttnn, torch
    return ttnn.from_torch(torch.randn(*shape) * 0.1, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=mesh,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


def time_traced(state, build_fn, steps, warmup):
    """build_fn() runs the op chain once (reads fixed inputs). Trace + time."""
    import ttnn, time as _t
    for _ in range(2):
        out = build_fn(); ttnn.synchronize_device(state.mesh)
        try: ttnn.deallocate(out)
        except Exception: pass
    tid = ttnn.begin_trace_capture(state.mesh, cq_id=0)
    out = build_fn()
    ttnn.end_trace_capture(state.mesh, tid, cq_id=0)
    for _ in range(warmup):
        ttnn.execute_trace(state.mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(state.mesh)
    t0 = _t.perf_counter()
    for _ in range(steps):
        ttnn.execute_trace(state.mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(state.mesh)
    ms = (_t.perf_counter() - t0) / steps * 1000.0
    ttnn.release_trace(state.mesh, tid)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,32,64")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]

    import ttnn
    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    from full_layer_tp_probe import VAL_DIM_CHIP
    HIDDEN = state.cfg['hidden']
    dn = next(l['dn'] for l in state.layers if l['type'] == 'linear_attention')
    w_in, w_out = dn['w_in'], dn['w_out']
    log(f"HIDDEN={HIDDEN} VAL_DIM_CHIP={VAL_DIM_CHIP}")
    log(f"w_in.shape={list(w_in.shape)}  w_out.shape={list(w_out.shape)}")

    log(f"=== traced DN matmul timing (steps={args.steps}) ===")
    rows = []
    for B in batches:
        h = repl(state.mesh, (B, HIDDEN))
        g = repl(state.mesh, (B, VAL_DIM_CHIP))
        in_ms = time_traced(state, lambda: ttnn.linear(h, w_in), args.steps, args.warmup)
        out_ms = time_traced(state, lambda: cb._tp_all_reduce(state, ttnn.linear(g, w_out)),
                             args.steps, args.warmup)
        rows.append((B, in_ms, out_ms))
        log(f"  B={B:3d}: in_proj {in_ms:6.3f} ms   out_proj+AR {out_ms:6.3f} ms   "
            f"sum {in_ms+out_ms:6.3f} ms")
        ttnn.deallocate(h); ttnn.deallocate(g)

    if len(batches) >= 2:
        b0, b1 = batches[0], batches[-1]
        d_in = (rows[-1][1] - rows[0][1]) / (b1 - b0)
        d_out = (rows[-1][2] - rows[0][2]) / (b1 - b0)
        log(f"=== slope {b0}→{b1} (ms/seq) ===")
        log(f"  in_proj: {d_in:.3f}   out_proj+AR: {d_out:.3f}   "
            f"both: {d_in+d_out:.3f}  (DN total slope ~3.61 with owned_gdn)")
        log("  if in_proj+out_proj are a large share of 3.61 ms/seq → matmul "
            "core_grid tuning (A004 lever) is the next optimization.")


if __name__ == "__main__":
    main()
