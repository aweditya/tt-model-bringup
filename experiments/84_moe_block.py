#!/usr/bin/env python3
"""
Experiment 84 — MoE block isolated kernel (Phase A5).

Tests the MoE-routing pattern from Qwen3.6-35B-A3B:
  - Router: linear(hidden→num_experts) + softmax + topk(8) + renormalize
  - 8 routed experts, each: gate_proj + up_proj + silu(gate)*up + down_proj
  - 1 shared expert with its own sigmoid gate
  - Sum: top-8 routed + sigmoid_gated_shared

We test FIRST with 8 experts (debug-easy), then scale to 256.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/84_moe_block.py
    cd ~/tt-xla && .venv/bin/python experiments/84_moe_block.py --full256
"""
import os, sys, time, statistics, argparse
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np

# Qwen3.6 MoE shapes
HIDDEN = 2048
MOE_INT = 512                 # per-expert intermediate
SHARED_INT = 512              # shared expert intermediate
TOP_K = 8
B = 1
EPS = 1e-6


# ============================================================
# Numpy reference
# ============================================================

def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def moe_block_numpy(x, router_w,
                     gate_ws, up_ws, down_ws,
                     shared_gate_w, shared_up_w, shared_down_w, shared_seg_w,
                     top_k=TOP_K, num_experts=8):
    """
    Inputs (all fp32):
      x:            [B, 1, hidden]
      router_w:     [hidden, num_experts]
      gate_ws:      [num_experts, hidden, moe_int]
      up_ws:        [num_experts, hidden, moe_int]
      down_ws:      [num_experts, moe_int, hidden]
      shared_*_w:   single expert weights
      shared_seg_w: [hidden, 1] — for shared expert's sigmoid gate

    Returns: out [B, 1, hidden]
    """
    h = x.reshape(B, HIDDEN)                       # [B, hidden]

    # Router
    logits = h @ router_w                          # [B, num_experts]
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    top_idx = np.argsort(probs, axis=-1)[:, -top_k:][:, ::-1]  # [B, top_k]
    # Gather top-k values
    top_val = np.take_along_axis(probs, top_idx, axis=-1)       # [B, top_k]
    top_val = top_val / top_val.sum(axis=-1, keepdims=True)     # renormalize

    # Sum top-k experts
    acc = np.zeros((B, HIDDEN), dtype=np.float32)
    for rank in range(top_k):
        e_idx = top_idx[0, rank]
        w = top_val[0, rank]
        g = silu_np(h @ gate_ws[e_idx])             # [B, moe_int]
        u = h @ up_ws[e_idx]                        # [B, moe_int]
        d = (g * u) @ down_ws[e_idx]                # [B, hidden]
        acc = acc + w * d

    # Shared expert with sigmoid gate
    sg = silu_np(h @ shared_gate_w)                 # [B, moe_int]
    su = h @ shared_up_w
    sd = (sg * su) @ shared_down_w                  # [B, hidden]
    seg = sigmoid_np(h @ shared_seg_w)              # [B, 1]
    acc = acc + seg * sd

    return acc.reshape(B, 1, HIDDEN)


# ============================================================
# ttnn implementation
# ============================================================

def moe_block_ttnn(x_tt, router_w_tt, gate_ws_tt, up_ws_tt, down_ws_tt,
                    shared_gate_w_tt, shared_up_w_tt, shared_down_w_tt,
                    shared_seg_w_tt, num_experts, device, ttnn):
    """
    Returns out [B, 1, hidden] on device. Routing uses host readback for
    expert indices, matching the production pattern in demos/generate_moe.py.
    """
    h = ttnn.reshape(x_tt, [B, HIDDEN])

    # Router on device
    logits = ttnn.matmul(h, router_w_tt)
    probs = ttnn.softmax(logits, dim=-1)
    top_val, top_idx = ttnn.topk(probs, TOP_K)
    ttnn.synchronize_device(device)
    # Host readback for sparse dispatch
    top_idx_np = ttnn.to_torch(top_idx).int().numpy().flatten()
    top_val_np = ttnn.to_torch(top_val).float().numpy().flatten()
    # Renormalize topk values
    top_val_np = top_val_np / top_val_np.sum()

    # Accumulate top-k experts
    acc = None
    for rank in range(TOP_K):
        e = int(top_idx_np[rank])
        w = float(top_val_np[rank])
        g = ttnn.linear(h, gate_ws_tt[e], activation="silu")
        u = ttnn.matmul(h, up_ws_tt[e])
        d = ttnn.matmul(ttnn.mul(g, u), down_ws_tt[e])
        weighted = ttnn.multiply(d, w)
        acc = weighted if acc is None else ttnn.add(acc, weighted)

    # Shared expert + sigmoid gate
    sg = ttnn.linear(h, shared_gate_w_tt, activation="silu")
    su = ttnn.matmul(h, shared_up_w_tt)
    sd = ttnn.matmul(ttnn.mul(sg, su), shared_down_w_tt)
    seg_logit = ttnn.matmul(h, shared_seg_w_tt)
    seg = ttnn.sigmoid(seg_logit)
    shared_gated = ttnn.mul(sd, seg)
    acc = ttnn.add(acc, shared_gated)

    return ttnn.reshape(acc, [B, 1, HIDDEN]), top_idx_np


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full256", action="store_true",
                        help="Use full 256 experts (default: 8 for debug)")
    args = parser.parse_args()
    num_experts = 256 if args.full256 else 8

    print("=" * 64)
    print(f"Phase A5 — MoE block isolated kernel ({num_experts} experts, top-{TOP_K})")
    print("=" * 64)
    print(f"  hidden={HIDDEN}, moe_intermediate={MOE_INT}, top_k={TOP_K}")
    print()

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((B, 1, HIDDEN)).astype(np.float32) * 0.1
    router_w = rng.standard_normal((HIDDEN, num_experts)).astype(np.float32) * 0.02
    gate_ws = rng.standard_normal((num_experts, HIDDEN, MOE_INT)).astype(np.float32) * 0.02
    up_ws = rng.standard_normal((num_experts, HIDDEN, MOE_INT)).astype(np.float32) * 0.02
    down_ws = rng.standard_normal((num_experts, MOE_INT, HIDDEN)).astype(np.float32) * 0.02
    shared_gate_w = rng.standard_normal((HIDDEN, SHARED_INT)).astype(np.float32) * 0.02
    shared_up_w = rng.standard_normal((HIDDEN, SHARED_INT)).astype(np.float32) * 0.02
    shared_down_w = rng.standard_normal((SHARED_INT, HIDDEN)).astype(np.float32) * 0.02
    shared_seg_w = rng.standard_normal((HIDDEN, 1)).astype(np.float32) * 0.02

    print(f"[1/4] Numpy reference")
    out_np = moe_block_numpy(x_np, router_w, gate_ws, up_ws, down_ws,
                              shared_gate_w, shared_up_w, shared_down_w,
                              shared_seg_w, num_experts=num_experts)
    print(f"  out range [{out_np.min():.4f}, {out_np.max():.4f}], norm={np.linalg.norm(out_np):.4f}")

    try:
        import ttnn, torch
    except ImportError:
        print("\n[ttnn not available — numpy reference verified, skipping ttnn]")
        return

    print("\n[2/4] Opening device and uploading weights")
    device = ttnn.open_device(device_id=0)

    def upload(arr, dtype=ttnn.bfloat16):
        t = torch.from_numpy(arr.astype(np.float32))
        while t.dim() < 2:
            t = t.unsqueeze(0)
        return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    x_tt = upload(x_np)
    router_w_tt = upload(router_w)
    # Per-expert weights: list of device tensors. Use bf8 like production port.
    print(f"  Uploading {num_experts} experts × 3 weights each (bf8)...")
    gate_ws_tt = [upload(gate_ws[i], dtype=ttnn.bfloat8_b) for i in range(num_experts)]
    up_ws_tt = [upload(up_ws[i], dtype=ttnn.bfloat8_b) for i in range(num_experts)]
    down_ws_tt = [upload(down_ws[i], dtype=ttnn.bfloat8_b) for i in range(num_experts)]
    shared_gate_w_tt = upload(shared_gate_w, dtype=ttnn.bfloat8_b)
    shared_up_w_tt = upload(shared_up_w, dtype=ttnn.bfloat8_b)
    shared_down_w_tt = upload(shared_down_w, dtype=ttnn.bfloat8_b)
    shared_seg_w_tt = upload(shared_seg_w)
    ttnn.synchronize_device(device)
    print("  All weights on device.")

    print("\n[3/4] Cosine check")
    out_tt, top_idx_dev = moe_block_ttnn(
        x_tt, router_w_tt, gate_ws_tt, up_ws_tt, down_ws_tt,
        shared_gate_w_tt, shared_up_w_tt, shared_down_w_tt, shared_seg_w_tt,
        num_experts, device, ttnn)
    ttnn.synchronize_device(device)
    out_back = ttnn.to_torch(out_tt).float().numpy().reshape(B, 1, HIDDEN)

    # NOTE: routing on device picks experts via ttnn.topk on bf16 probs.
    # Numpy picks experts via fp32 argsort. Different precision → potentially
    # different expert ranking. We compare:
    #   (a) the device-selected experts against numpy with the SAME experts
    #   (b) the device output vs numpy output assuming our selection is right
    cos_v = _cosine(out_np, out_back)
    max_abs = float(np.max(np.abs(out_np - out_back)))
    print(f"  cosine(out) = {cos_v:.6f}   max-abs-diff = {max_abs:.6f}")
    print(f"  device-selected experts: {sorted(top_idx_dev.tolist())}")

    # If cosine is mediocre, redo numpy with device's expert selection
    # (isolates routing-precision drift from expert-math drift).
    if cos_v < 0.99:
        print("  cosine < 0.99 — re-running numpy with device's expert selection")
        # Reconstruct numpy with the EXACT experts the device picked
        h_np = x_np.reshape(B, HIDDEN)
        # Recompute device's router probs in fp32 just to get the correct weights
        logits = h_np @ router_w
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs = probs / probs.sum(-1, keepdims=True)
        # Use device's selection for indices, get the corresponding (renormalized) weights
        dev_probs = probs[0, top_idx_dev]
        dev_probs = dev_probs / dev_probs.sum()
        acc = np.zeros((B, HIDDEN), dtype=np.float32)
        for rank in range(TOP_K):
            e_idx = int(top_idx_dev[rank])
            w = float(dev_probs[rank])
            g = silu_np(h_np @ gate_ws[e_idx])
            u = h_np @ up_ws[e_idx]
            d = (g * u) @ down_ws[e_idx]
            acc = acc + w * d
        sg = silu_np(h_np @ shared_gate_w)
        su = h_np @ shared_up_w
        sd = (sg * su) @ shared_down_w
        seg = sigmoid_np(h_np @ shared_seg_w)
        acc = acc + seg * sd
        out_np_dev_sel = acc.reshape(B, 1, HIDDEN)
        cos_v2 = _cosine(out_np_dev_sel, out_back)
        print(f"  cosine(out, numpy_with_dev_selection) = {cos_v2:.6f}")
        PASS = cos_v2 >= 0.99
    else:
        PASS = cos_v >= 0.99

    print(f"  VERDICT: {'PASS ✓' if PASS else 'FAIL ✗'}")

    print("\n[4/4] Performance")
    WARMUP, ITERS = 5, 50  # fewer iters because each call has top-k matmuls

    for _ in range(WARMUP):
        moe_block_ttnn(x_tt, router_w_tt, gate_ws_tt, up_ws_tt, down_ws_tt,
                        shared_gate_w_tt, shared_up_w_tt, shared_down_w_tt,
                        shared_seg_w_tt, num_experts, device, ttnn)
    ttnn.synchronize_device(device)

    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        moe_block_ttnn(x_tt, router_w_tt, gate_ws_tt, up_ws_tt, down_ws_tt,
                        shared_gate_w_tt, shared_up_w_tt, shared_down_w_tt,
                        shared_seg_w_tt, num_experts, device, ttnn)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
    med = statistics.median(samples)
    p90 = statistics.quantiles(samples, n=10)[8] if len(samples) >= 10 else samples[-1]

    # Memory ceiling: read 9 active experts × 3 matmuls × 2048×512×1 byte (bf8)
    # per active expert: 3 × 2048 × 512 = 3.1M bytes
    active_bytes = 9 * 3 * HIDDEN * MOE_INT  # bf8 = 1 byte/param
    mem_floor_us = active_bytes / 450e9 * 1e6
    pct = mem_floor_us / med * 100
    print(f"  decode step (eager):   median = {med:7.1f} µs    p90 = {p90:7.1f} µs")
    print(f"  active expert bytes (bf8): {active_bytes/1e6:.2f} MB → memory floor {mem_floor_us:.1f} µs")
    print(f"  eager % of ceiling: {pct:.2f}%")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
