#!/usr/bin/env python3
"""DNK-G4 isolation — kill the conv1d K=4→32 tile-padding tax.

DNK-G3b: conv1d is 71.8% of the DN cost. The current impl builds
conv_input [B, C, K=4] and (concat/mul/sum/silu/slice) it; in TILE layout K=4
pads to 32 → ~8× wasted traffic. This isolates a **shift-accumulate**
reformulation that never materialises a K dim:

  out = silu( w0·s0 + w1·s1 + w2·s2 + w3·cur )

with the conv state held as 3 SEPARATE [B,C] columns (s0,s1,s2 = the last 3
inputs) and per-tap weights w_k as [1,C] (pre-transposed w_conv). All ops are on
[B,C] tiles (C a whole-tile multiple → no padding). State update = shift
(s0←s1, s1←s2, s2←cur).

Compares correctness (vs numpy) and traced timing at B=32: current [B,C,K] vs
shift-accumulate. If shift-accum is correct AND faster, it's a no-custom-kernel
win for the 72% DN lever.

Run on qb1:
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb_conv_reform_isolation.py --batch 32
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

C = 2560   # CONV_DIM_CHIP (27B per-chip): 2*KEY_DIM_CHIP + VAL_DIM_CHIP
CONV_K = 4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1); b = np.asarray(b, np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def silu_np(x):
    return x / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    B = args.batch

    import ttnn
    log(f"opening device {args.device_id}  B={B} C={C} K={CONV_K}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        rng = np.random.default_rng(0)
        w = rng.normal(0, 0.5, (C, CONV_K)).astype(np.float32)       # per-channel taps
        state3 = rng.normal(0, 1.0, (B, C, CONV_K - 1)).astype(np.float32)  # s0,s1,s2
        cur = rng.normal(0, 1.0, (B, C)).astype(np.float32)

        # numpy reference: window = [s0,s1,s2,cur]; out = silu(sum_k w[:,k]*window[:,:,k])
        window = np.concatenate([state3, cur[:, :, None]], axis=2)   # [B,C,4]
        out_ref = silu_np((window * w[None, :, :]).sum(axis=2))      # [B,C]

        def tt(x, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x.astype(np.float32))),
                                   dtype=ttnn.bfloat16, layout=layout, device=device)

        # --- current [B,C,K] approach ---
        st_tt = tt(state3); cur_tt = tt(cur); w_bck = tt(w.reshape(1, C, CONV_K))
        def conv_current():
            cur_col = ttnn.reshape(cur_tt, [B, C, 1])
            ci = ttnn.concat([st_tt, cur_col], dim=-1)        # [B,C,4]
            prod = ttnn.mul(ci, w_bck)
            o = ttnn.silu(ttnn.sum(prod, dim=-1))             # [B,C]
            ttnn.deallocate(cur_col); ttnn.deallocate(ci); ttnn.deallocate(prod)
            return o
        o_cur = conv_current()
        cos_cur = cos(ttnn.to_torch(o_cur).float().numpy().reshape(B, C), out_ref)
        ttnn.deallocate(o_cur)

        # --- shift-accumulate: per-tap [1,C] weights, state as 3 [B,C] columns ---
        # w_col[k] = w[:,k] as [1,C]
        w_cols = [tt(w[:, k].reshape(1, C)) for k in range(CONV_K)]
        s_cols = [tt(state3[:, :, j]) for j in range(CONV_K - 1)]   # [B,C] each
        cur_sa = tt(cur)
        def conv_shiftacc():
            # out = w0*s0 + w1*s1 + w2*s2 + w3*cur, then silu — all [B,C]
            acc = ttnn.mul(s_cols[0], w_cols[0])
            for j in range(1, CONV_K - 1):
                t = ttnn.mul(s_cols[j], w_cols[j])
                acc2 = ttnn.add(acc, t)
                ttnn.deallocate(acc); ttnn.deallocate(t); acc = acc2
            t = ttnn.mul(cur_sa, w_cols[CONV_K - 1])
            acc2 = ttnn.add(acc, t); ttnn.deallocate(acc); ttnn.deallocate(t)
            o = ttnn.silu(acc2); ttnn.deallocate(acc2)
            return o
        o_sa = conv_shiftacc()
        cos_sa = cos(ttnn.to_torch(o_sa).float().numpy().reshape(B, C), out_ref)
        ttnn.deallocate(o_sa)

        log(f"correctness vs numpy: current cos={cos_cur:.6f}  shift-accum cos={cos_sa:.6f}")

        def time_traced(build_fn):
            for _ in range(2):
                o = build_fn(); ttnn.synchronize_device(device); ttnn.deallocate(o)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            o = build_fn()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            for _ in range(args.warmup):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(args.steps):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            ms = (time.perf_counter() - t0) / args.steps * 1000.0
            ttnn.release_trace(device, tid)
            return ms

        ms_cur = time_traced(conv_current)
        ms_sa = time_traced(conv_shiftacc)
        log(f"traced per-call: current {ms_cur*1000:.2f} us   shift-accum {ms_sa*1000:.2f} us   "
            f"speedup {ms_cur/ms_sa:.2f}x")
        ok = (cos_cur >= 0.99 and cos_sa >= 0.99)
        log(f"VERDICT: {'shift-accum CORRECT' if cos_sa >= 0.99 else 'shift-accum WRONG'}; "
            f"{ms_cur/ms_sa:.2f}x faster → "
            f"{'WORTH integrating (×48 DN layers)' if ms_sa < ms_cur and cos_sa >= 0.99 else 'not a win'}")
        if not ok:
            raise SystemExit(1)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
