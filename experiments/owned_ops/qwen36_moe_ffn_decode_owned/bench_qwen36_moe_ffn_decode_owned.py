#!/usr/bin/env python3
"""Isolated perf comparison: G1b fused kernel vs 3-op ttnn chain.

Both paths run on identical [H], [E,H,2I], [E,I,H], [E] inputs. Reports:
  - mean kernel ms/call (sync-bounded, warmup + N timed iters)
  - pcc between the two outputs (correctness sanity)
  - pcc of each vs bf16 numpy oracle

The 3-op chain mirrors what server_35b_ttnn's Pattern A batched MoE does
in trace: bmm(h_repl, W1) -> silu(gate)*up -> bmm(mid, W2) -> rw*eo sum.
This is the kernel-replacement comparison; not the user-visible decode
time (that's gated on G4 integration).

Run on qb1 with TT_METAL_HOME etc. set — see test_qwen36_moe_ffn_decode_owned.py
header for the full env block.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import moe_ffn_kernel_oracle as oracle  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

TILE = 32


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_ttnn(arr_np, device, dtype=None, layout=None):
    import ttnn
    if dtype is None: dtype = ttnn.bfloat16
    if layout is None: layout = ttnn.TILE_LAYOUT
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(arr_np.astype(np.float32))),
        dtype=dtype, layout=layout, device=device,
    )


def to_host(t):
    import ttnn
    return ttnn.to_torch(t).float().numpy()


def broadcast_rw_to_tiles(rw_1d):
    rw = rw_1d.astype(np.float32).reshape(-1, 1, 1)
    return np.broadcast_to(rw, (rw.shape[0], TILE, TILE)).copy()


def run_kernel_path(h_tt, W1_tt, W2_tt, rw_bcast_tt, device, n_warmup, n_iters):
    """Time the fused G1b kernel."""
    import ttnn
    # warmup (JIT cache + first-call overhead)
    for _ in range(n_warmup):
        out = ttnn.experimental.qwen36_moe_ffn_decode_owned(h_tt, W1_tt, W2_tt, rw_bcast_tt)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    # batched timed iters
    ts = []
    for _ in range(n_iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = ttnn.experimental.qwen36_moe_ffn_decode_owned(h_tt, W1_tt, W2_tt, rw_bcast_tt)
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
        last_out = out
    out_np = to_host(last_out)[0]
    ttnn.deallocate(last_out)
    return np.array(ts), out_np


def run_threeop_path(h_E_tt, W1_tt, W2_tt, rw_E_tt, device, n_warmup, n_iters, E, H, I):
    """Time the 3-op chain that the kernel replaces.

      gate_up = h_E @ W1                  # bmm [E,1,H] x [E,H,2I] -> [E,1,2I]
      gate, up = gate_up[..., :I], gate_up[..., I:]
      mid = silu(gate) * up               # [E,1,I]
      eo = mid @ W2                       # bmm [E,1,I] x [E,I,H] -> [E,1,H]
      routed = sum_e (rw[e] * eo[e])      # [1,H]
    """
    import ttnn

    def forward():
        gate_up = ttnn.matmul(h_E_tt, W1_tt)               # [E,1,2I]
        # Split via slice. With logical [E,1,2I], slice on last dim.
        gate = ttnn.slice(gate_up, [0, 0, 0],    [E, 1, I])
        up   = ttnn.slice(gate_up, [0, 0, I],    [E, 1, 2 * I])
        ttnn.deallocate(gate_up)
        silu_gate = ttnn.silu(gate)
        ttnn.deallocate(gate)
        mid = ttnn.multiply(silu_gate, up)                 # [E,1,I]
        ttnn.deallocate(silu_gate); ttnn.deallocate(up)
        eo = ttnn.matmul(mid, W2_tt)                       # [E,1,H]
        ttnn.deallocate(mid)
        # rw[e] * eo[e], summed over experts.
        # rw_E_tt shape [E,1,1] in tile-padded layout; broadcast against [E,1,H]
        scaled = ttnn.multiply(eo, rw_E_tt)                # [E,1,H]
        ttnn.deallocate(eo)
        # sum over expert dim → [1, H]. ttnn.sum on dim 0.
        routed = ttnn.sum(scaled, dim=0, keepdim=False)    # [1, H]
        ttnn.deallocate(scaled)
        return routed

    for _ in range(n_warmup):
        out = forward()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    ts = []
    for _ in range(n_iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = forward()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
        last_out = out
    out_np = to_host(last_out)[0]
    ttnn.deallocate(last_out)
    # Squeeze leading singleton if needed.
    if out_np.ndim == 2 and out_np.shape[0] == 1:
        out_np = out_np[0]
    elif out_np.ndim == 1:
        pass
    return np.array(ts), out_np


def run_threeop_traced(h_E_tt, W1_tt, W2_tt, rw_E_tt, device, n_warmup, n_iters, E, H, I):
    """Capture the 3-op chain in a trace, then time execute_trace.

    Trace mode amortizes dispatch overhead — the question is whether the
    dispatch tax visible in eager mode survives trace.
    """
    import ttnn

    # Run forward once eagerly so the kernel JIT cache is warm and the
    # captured trace doesn't include compilation.
    def forward():
        gate_up = ttnn.matmul(h_E_tt, W1_tt)
        gate = ttnn.slice(gate_up, [0, 0, 0], [E, 1, I])
        up   = ttnn.slice(gate_up, [0, 0, I], [E, 1, 2 * I])
        silu_gate = ttnn.silu(gate)
        mid = ttnn.multiply(silu_gate, up)
        eo = ttnn.matmul(mid, W2_tt)
        scaled = ttnn.multiply(eo, rw_E_tt)
        routed = ttnn.sum(scaled, dim=0, keepdim=False)
        return routed

    out_warm = forward()
    ttnn.synchronize_device(device)
    ttnn.deallocate(out_warm)

    # Capture the trace.
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = forward()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    # Warmup execute_trace calls.
    for _ in range(n_warmup):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    # Timed iters.
    ts = []
    for _ in range(n_iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1000.0)

    out_np = to_host(out_trace)
    if out_np.ndim == 2 and out_np.shape[0] == 1:
        out_np = out_np[0]

    ttnn.release_trace(device, trace_id)
    return np.array(ts), out_np


def run_threeop_breakdown(h_E_tt, W1_tt, W2_tt, rw_E_tt, device, n_warmup, n_iters, E, H, I):
    """Time EACH ttnn op in the 3-op chain separately. Sync-bounded per op."""
    import ttnn

    def time_op(fn, n_w, n_i):
        for _ in range(n_w):
            fn().__class__  # warmup; result discarded by caller via deallocate
        ttnn.synchronize_device(device)
        ts = []
        results = []
        for _ in range(n_i):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(device)
            ts.append((time.perf_counter() - t0) * 1000.0)
            results.append(out)
        return np.array(ts), results

    # ---- bmm1: gate_up = h @ W1 ----
    bmm1_ts, gate_up_list = time_op(
        lambda: ttnn.matmul(h_E_tt, W1_tt), n_warmup, n_iters)
    gate_up = gate_up_list[-1]
    for x in gate_up_list[:-1]:
        ttnn.deallocate(x)

    # ---- slice gate/up ----
    slice_ts, _ = time_op(
        lambda: (ttnn.slice(gate_up, [0, 0, 0], [E, 1, I]),
                 ttnn.slice(gate_up, [0, 0, I], [E, 1, 2 * I])),
        n_warmup, n_iters)
    # Get a real gate/up for next steps.
    gate = ttnn.slice(gate_up, [0, 0, 0], [E, 1, I])
    up   = ttnn.slice(gate_up, [0, 0, I], [E, 1, 2 * I])
    ttnn.deallocate(gate_up)

    # ---- silu + mul (treated as fused step in our path) ----
    def silu_mul():
        s = ttnn.silu(gate)
        m = ttnn.multiply(s, up)
        ttnn.deallocate(s)
        return m
    silumul_ts, mid_list = time_op(silu_mul, n_warmup, n_iters)
    mid = mid_list[-1]
    for x in mid_list[:-1]:
        ttnn.deallocate(x)

    # ---- bmm2: eo = mid @ W2 ----
    bmm2_ts, eo_list = time_op(
        lambda: ttnn.matmul(mid, W2_tt), n_warmup, n_iters)
    eo = eo_list[-1]
    for x in eo_list[:-1]:
        ttnn.deallocate(x)

    # ---- rw * eo, sum over experts ----
    def rw_sum():
        scaled = ttnn.multiply(eo, rw_E_tt)
        routed = ttnn.sum(scaled, dim=0, keepdim=False)
        ttnn.deallocate(scaled)
        return routed
    rwsum_ts, _ = time_op(rw_sum, n_warmup, n_iters)

    ttnn.deallocate(gate); ttnn.deallocate(up); ttnn.deallocate(mid); ttnn.deallocate(eo)
    return {
        "bmm1 h@W1":   bmm1_ts,
        "slice gate/up": slice_ts,
        "silu*up":     silumul_ts,
        "bmm2 mid@W2": bmm2_ts,
        "rw·eo + sum": rwsum_ts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--moe-inter", type=int, default=512)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-iters", type=int, default=20)
    ap.add_argument("--breakdown", action="store_true",
                    help="Also report per-op breakdown of the 3-op chain.")
    ap.add_argument("--traced", action="store_true",
                    help="Also report trace-mode timing for the 3-op chain.")
    args = ap.parse_args()

    if args.hidden % TILE or args.moe_inter % TILE:
        raise SystemExit("--hidden and --moe-inter must be multiples of TILE=32")

    import ttnn
    if not hasattr(ttnn.experimental, "qwen36_moe_ffn_decode_owned"):
        raise SystemExit("ttnn.experimental.qwen36_moe_ffn_decode_owned not registered")

    H, I, E = args.hidden, args.moe_inter, args.experts
    log(f"bench shape: H={H} I={I} E={E}  warmup={args.n_warmup} iters={args.n_iters}")
    log(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        fx = oracle.make_fixture(E=E, HIDDEN=H, MOE_INTER=I, seed=args.seed)

        # Inputs for the kernel path.
        h_tt        = to_ttnn(fx.h.reshape(1, H), device)
        W1_tt       = to_ttnn(fx.W1, device)
        W2_tt       = to_ttnn(fx.W2, device)
        rw_bcast    = broadcast_rw_to_tiles(fx.routing_weight)
        rw_bcast_tt = to_ttnn(rw_bcast, device)

        # Inputs for the 3-op chain: h replicated to [E,1,H], rw as [E,1,1].
        h_E   = np.broadcast_to(fx.h.reshape(1, 1, H), (E, 1, H)).copy()
        h_E_tt = to_ttnn(h_E, device)
        rw_E  = fx.routing_weight.astype(np.float32).reshape(E, 1, 1)
        rw_E_tt = to_ttnn(rw_E, device)

        expected = oracle.moe_ffn_oracle(fx, bf16=True)

        log("--- kernel path (G1b fused) ---")
        kernel_ts, kernel_out = run_kernel_path(
            h_tt, W1_tt, W2_tt, rw_bcast_tt, device, args.n_warmup, args.n_iters)
        log(f"kernel: mean {kernel_ts.mean():.3f} ms  median {np.median(kernel_ts):.3f} ms  "
            f"min {kernel_ts.min():.3f}  max {kernel_ts.max():.3f}  std {kernel_ts.std():.3f}")
        log(f"kernel pcc vs oracle = {oracle.pcc(kernel_out, expected):.6f}")

        log("--- 3-op chain (bmm + silu + bmm + rw·sum) ---")
        three_ts, three_out = run_threeop_path(
            h_E_tt, W1_tt, W2_tt, rw_E_tt, device, args.n_warmup, args.n_iters, E, H, I)
        log(f"3-op:   mean {three_ts.mean():.3f} ms  median {np.median(three_ts):.3f} ms  "
            f"min {three_ts.min():.3f}  max {three_ts.max():.3f}  std {three_ts.std():.3f}")
        log(f"3-op pcc vs oracle = {oracle.pcc(three_out, expected):.6f}")

        log("--- comparison ---")
        log(f"kernel vs 3-op pcc = {oracle.pcc(kernel_out, three_out):.6f}")
        ratio = three_ts.mean() / kernel_ts.mean() if kernel_ts.mean() > 0 else float("inf")
        log(f"speedup: 3-op / kernel = {ratio:.3f}x  "
            f"({'kernel wins' if ratio > 1 else '3-op wins'})")

        if args.traced:
            log("--- 3-op chain TRACED ---")
            traced_ts, traced_out = run_threeop_traced(
                h_E_tt, W1_tt, W2_tt, rw_E_tt, device, args.n_warmup, args.n_iters, E, H, I)
            log(f"3-op traced: mean {traced_ts.mean():.3f} ms  median {np.median(traced_ts):.3f}  "
                f"min {traced_ts.min():.3f}  max {traced_ts.max():.3f}  std {traced_ts.std():.3f}")
            log(f"3-op traced pcc vs oracle = {oracle.pcc(traced_out, expected):.6f}")
            log(f"eager / traced ratio = {three_ts.mean() / traced_ts.mean():.3f}x  "
                f"(dispatch savings from trace)")

        if args.breakdown:
            log("--- 3-op per-op breakdown ---")
            br = run_threeop_breakdown(
                h_E_tt, W1_tt, W2_tt, rw_E_tt, device, args.n_warmup, args.n_iters, E, H, I)
            sum_mean = sum(v.mean() for v in br.values())
            for name, ts in br.items():
                log(f"  {name:18s} mean {ts.mean():7.3f} ms  "
                    f"median {np.median(ts):7.3f}  "
                    f"({100*ts.mean()/sum_mean:5.1f}% of sum)")
            log(f"  {'sum-of-isolated':18s} mean {sum_mean:7.3f} ms "
                f"vs chained {three_ts.mean():.3f} ms  "
                f"(chained / sum = {three_ts.mean()/sum_mean:.3f})")

        for t in [h_tt, W1_tt, W2_tt, rw_bcast_tt, h_E_tt, rw_E_tt]:
            try: ttnn.deallocate(t)
            except Exception: pass
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
