#!/usr/bin/env python3
"""Bench: 4 separate DN in_proj matmuls vs 1 fused matmul on (1,4) traced.

Mirrors the production DN forward at server_35b_ttnn.dn_forward_ttnn lines
389-392:
    mixed_qkv = h @ W_qkv    [1, HIDDEN] x [HIDDEN, 2048] -> [1, 2048] per chip
    z         = h @ W_z      [1, HIDDEN] x [HIDDEN, 1024] -> [1, 1024]
    a         = h @ W_a      [1, HIDDEN] x [HIDDEN, 8]    -> [1, 8]
    b         = h @ W_b      [1, HIDDEN] x [HIDDEN, 8]    -> [1, 8]

Fused version concatenates the four weight matrices along the OUT dim:
    W_fused = concat([W_qkv, W_z, W_a, W_b], axis=1)  -> [HIDDEN, 3088]
    fused = h @ W_fused                                -> [1, 3088]
    mixed_qkv, z, a, b = slice fused along last dim

If fusion is a perf win in trace, we gain by: (1) saving 3 dispatch calls
(but trace amortizes — likely small), (2) the combined matmul has better
shape/utilization than four mismatched ones (the a/b matmuls are tiny:
HIDDEN=2048 × OUT=8 = pure tile-overhead).

Run on qb1 with TT_METAL_HOME etc set (see test_qwen36_decay_gate header).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

HIDDEN = 2048
NUM_K_HEADS = 16          # total across mesh
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM      # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM    # 4096
CONV_DIM = 2 * KEY_DIM + VALUE_DIM      # 8192
# Per-chip dims (NCHIPS=4)
NCHIPS = 4
CONV_DIM_CHIP = CONV_DIM // NCHIPS      # 2048
V_DIM_CHIP = VALUE_DIM // NCHIPS        # 1024
NV_PER_CHIP = NUM_V_HEADS // NCHIPS     # 8
A_OUT_CHIP = NV_PER_CHIP                # 8
B_OUT_CHIP = NV_PER_CHIP                # 8
FUSED_OUT_CHIP = CONV_DIM_CHIP + V_DIM_CHIP + A_OUT_CHIP + B_OUT_CHIP  # 3088


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_torch32(arr):
    return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))


def make_weights_per_chip(seed=0):
    """Build random per-chip weights matching the production sharding."""
    rng = np.random.default_rng(seed)
    # Per-chip weights. Total mesh weight is the concat across chips.
    def w(in_d, out_d):
        return rng.standard_normal((in_d, out_d)).astype(np.float32) * 0.02
    w_qkv = w(HIDDEN, CONV_DIM_CHIP)
    w_z   = w(HIDDEN, V_DIM_CHIP)
    w_a   = w(HIDDEN, A_OUT_CHIP)
    w_b   = w(HIDDEN, B_OUT_CHIP)
    h     = rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.1
    return h, w_qkv, w_z, w_a, w_b


def time_traced(device, capture_fn, n_warmup, n_iters):
    """Capture `capture_fn` in a trace, then time execute_trace N times."""
    import ttnn
    # Warmup before capture (JIT + caches).
    out_warm = capture_fn()
    ttnn.synchronize_device(device)
    if isinstance(out_warm, tuple):
        for x in out_warm: ttnn.deallocate(x)
    else:
        ttnn.deallocate(out_warm)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = capture_fn()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    for _ in range(n_warmup):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    ts = []
    for _ in range(n_iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1000.0)

    ttnn.release_trace(device, trace_id)
    return np.array(ts), out_trace


def to_mesh_replicated(arr_np, mesh, dtype, layout):
    import ttnn
    return ttnn.from_torch(to_torch32(arr_np), dtype=dtype, layout=layout,
                            device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


def to_mesh_sharded(arr_per_chip_np, mesh, dtype, layout, shard_dim):
    """arr_per_chip_np is [NCHIPS, ...] — chip i gets arr[i]."""
    import ttnn
    # Stack the per-chip slices into one tensor with leading mesh dim, then use
    # ShardTensorToMesh to distribute.
    stacked = np.stack(arr_per_chip_np, axis=0)  # [NCHIPS, in, out_chip]
    return ttnn.from_torch(
        to_torch32(stacked), dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-iters", type=int, default=50)
    args = ap.parse_args()

    import ttnn
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    except Exception as e:
        log(f"fabric config warning: {e}")

    log(f"opening (1,4) mesh on qb1")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        log(f"shapes per chip: HIDDEN={HIDDEN}, "
            f"qkv={CONV_DIM_CHIP}, z={V_DIM_CHIP}, a={A_OUT_CHIP}, b={B_OUT_CHIP}, "
            f"fused={FUSED_OUT_CHIP}")

        h_np, w_qkv, w_z, w_a, w_b = make_weights_per_chip(seed=0)

        # All weights are SAME on every chip (this is an isolated bench — we only
        # care about per-chip compute time, not sharding semantics).
        h_tt    = to_mesh_replicated(h_np,    mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        wqkv_tt = to_mesh_replicated(w_qkv,   mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        wz_tt   = to_mesh_replicated(w_z,     mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # a/b: out-dim < TILE=32 → ttnn pads up to a full tile. To keep the
        # production behavior exact, allocate them as bf16 TILE_LAYOUT just
        # like the production server does.
        wa_tt   = to_mesh_replicated(w_a,     mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        wb_tt   = to_mesh_replicated(w_b,     mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # Fused weight is the concat along OUT dim.
        w_fused = np.concatenate([w_qkv, w_z, w_a, w_b], axis=1)  # [HIDDEN, 3088]
        wfused_tt = to_mesh_replicated(w_fused, mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)

        HIFI4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # ---- 4-call path ----
        def forward_4():
            qkv = ttnn.matmul(h_tt, wqkv_tt, compute_kernel_config=HIFI4)
            z   = ttnn.matmul(h_tt, wz_tt,   compute_kernel_config=HIFI4)
            a   = ttnn.matmul(h_tt, wa_tt,   compute_kernel_config=HIFI4)
            b   = ttnn.matmul(h_tt, wb_tt,   compute_kernel_config=HIFI4)
            return (qkv, z, a, b)

        # ---- fused path ----
        # Slice the fused output along the last dim to recover qkv/z/a/b.
        # Slice offsets are in OUT-dim units (TILE-padded internally).
        OFF_QKV = 0
        OFF_Z   = OFF_QKV + CONV_DIM_CHIP            # 2048
        OFF_A   = OFF_Z + V_DIM_CHIP                 # 3072
        OFF_B   = OFF_A + A_OUT_CHIP                 # 3080
        END     = OFF_B + B_OUT_CHIP                 # 3088
        def forward_fused():
            fused = ttnn.matmul(h_tt, wfused_tt, compute_kernel_config=HIFI4)
            qkv = ttnn.slice(fused, [0, OFF_QKV],     [1, OFF_Z])
            z   = ttnn.slice(fused, [0, OFF_Z],       [1, OFF_A])
            a   = ttnn.slice(fused, [0, OFF_A],       [1, OFF_B])
            b   = ttnn.slice(fused, [0, OFF_B],       [1, END])
            ttnn.deallocate(fused)
            return (qkv, z, a, b)

        log("=== 4-call path (today) ===")
        ts4, out4 = time_traced(mesh, forward_4, args.n_warmup, args.n_iters)
        log(f"4-call traced: mean {ts4.mean():.4f} ms  median {np.median(ts4):.4f}  "
            f"min {ts4.min():.4f}  max {ts4.max():.4f}  std {ts4.std():.4f}")
        for x in out4: ttnn.deallocate(x)

        log("=== fused path (candidate) ===")
        tsf, outf = time_traced(mesh, forward_fused, args.n_warmup, args.n_iters)
        log(f"fused traced:  mean {tsf.mean():.4f} ms  median {np.median(tsf):.4f}  "
            f"min {tsf.min():.4f}  max {tsf.max():.4f}  std {tsf.std():.4f}")
        for x in outf: ttnn.deallocate(x)

        log("=== correctness: fused vs 4-call ===")
        # Run each path once eagerly and compare per-chip outputs.
        qkv4, z4, a4, b4 = forward_4()
        ttnn.synchronize_device(mesh)
        qkvf, zf, af, bf = forward_fused()
        ttnn.synchronize_device(mesh)
        def get_chip0(t):
            # Pull per-chip slice 0 from a sharded/replicated mesh tensor.
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().numpy()[0]
        def pcc(a, b):
            af_ = a.flatten().astype(np.float64); bf_ = b.flatten().astype(np.float64)
            af_ -= af_.mean(); bf_ -= bf_.mean()
            denom = np.sqrt((af_ ** 2).sum() * (bf_ ** 2).sum())
            return float((af_ * bf_).sum() / denom) if denom > 0 else 1.0
        for name, (t4, tf) in [
            ("qkv", (qkv4, qkvf)), ("z", (z4, zf)),
            ("a",   (a4,   af)),   ("b", (b4, bf)),
        ]:
            n4 = get_chip0(t4); nf = get_chip0(tf)
            # bf16 matmul outputs may differ between paths because the fused path's
            # tile-internal accumulation order changes. PCC should still be ~1.
            log(f"  {name}: shape4={n4.shape} shape_f={nf.shape} "
                f"pcc={pcc(n4, nf):.6f}  max_abs_diff={np.max(np.abs(n4-nf)):.2e}")
        for x in (qkv4, z4, a4, b4, qkvf, zf, af, bf):
            ttnn.deallocate(x)

        log("=== summary ===")
        delta_ms = ts4.mean() - tsf.mean()
        speedup = ts4.mean() / tsf.mean() if tsf.mean() > 0 else float("inf")
        log(f"delta:   {delta_ms:+.4f} ms per call")
        log(f"speedup: {speedup:.3f}x")
        log(f"per-token impact (30 DN layers): {delta_ms * 30:+.2f} ms/tok")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
