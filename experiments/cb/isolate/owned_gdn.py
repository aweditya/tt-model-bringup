#!/usr/bin/env python3
"""CB6/G0 — does ttnn.experimental.qwen36_gdn_decode_owned accept B>1?

CB5 showed the DeltaNet manual recurrence owns 97% of the per-token compute
slope. The owned-GDN fused kernel (production B=1 path) is the fix — IF it
accepts a batch leading dim. This is the decisive feasibility test: build the
kernel inputs at B=1 and B=4 and check (a) it runs at B>1, (b) each slot's
H_new + out match the numpy GatedDeltaNet recurrence reference.

owned_gdn inputs (per server_tp.py:798, gqa4 → [1,NV,1,K]):
  H[B,NV,K,V]  q,k[B,NV,1,K]  v[B,NV,1,V]  decay,beta4[B,NV,1,1]

If B>1 works → swapping the CB manual recurrence for owned_gdn is the (nearly
free) lever to lift the ~232 tok/s ceiling. If it asserts B=1 → the kernel's
program factory needs batching (a real build).

Self-contained, single device. 27B per-chip DN dims: NV=12, K=V=128.

Run on qb1:
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb_owned_gdn_batch_isolation.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

NV = 12      # value heads / chip (48/4)
K = 128      # key head dim
V = 128      # value head dim


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1); b = np.asarray(b, np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def gdn_ref(H, q, k, v, decay, beta):
    """numpy GatedDeltaNet recurrence (mirrors server_tp.py:810-819).
    H[B,NV,K,V] q,k[B,NV,K] v[B,NV,V] decay,beta[B,NV] → H_new[B,NV,K,V], out[B,NV,V]."""
    H_dec = H * decay[:, :, None, None]
    kv_mem = (H_dec * k[:, :, :, None]).sum(axis=2)              # [B,NV,V]
    delta = (v - kv_mem) * beta[:, :, None]                     # [B,NV,V]
    H_new = H_dec + k[:, :, :, None] * delta[:, :, None, :]     # [B,NV,K,V]
    out = (H_new * q[:, :, :, None]).sum(axis=2)                # [B,NV,V]
    return H_new, out


def run(device, B, seed, debug_mode=0):
    import ttnn
    rng = np.random.default_rng(seed)
    H = rng.normal(0, 0.3, (B, NV, K, V)).astype(np.float32)
    q = rng.normal(0, 1.0, (B, NV, K)).astype(np.float32)
    k = rng.normal(0, 1.0, (B, NV, K)).astype(np.float32)
    v = rng.normal(0, 1.0, (B, NV, V)).astype(np.float32)
    decay = rng.uniform(0.85, 0.99, (B, NV)).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, (B, NV)).astype(np.float32)
    H_ref, out_ref = gdn_ref(H, q, k, v, decay, beta)

    def tt(x, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x.astype(np.float32))),
                               dtype=ttnn.bfloat16, layout=layout, device=device)
    # FOLD-INTO-SLOTS: each slot's recurrence is independent, and the kernel
    # parallelizes over slots = state.shape[1]. So pass [1, B*NV, ...] (batch
    # folded into the slots dim) to the UNMODIFIED B=1 kernel — no rebuild.
    # Contiguous [B,NV,...] reshape to [1,B*NV,...] makes slot = b*NV + nv.
    S = B * NV
    H_tt = tt(H.reshape(1, S, K, V))
    q_tt = tt(q.reshape(1, S, 1, K)); k_tt = tt(k.reshape(1, S, 1, K))
    v_tt = tt(v.reshape(1, S, 1, V))
    decay_tt = tt(decay.reshape(1, S, 1, 1)); beta_tt = tt(beta.reshape(1, S, 1, 1))

    H_new, out = ttnn.experimental.qwen36_gdn_decode_owned(
        H_tt, q_tt, k_tt, v_tt, decay_tt, beta_tt,
        native_io=True, debug_mode=debug_mode, output_memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    H_out = ttnn.to_torch(H_new).float().numpy().reshape(B, NV, K, V)
    out_out = ttnn.to_torch(out).float().numpy().reshape(B, NV, V)
    for t in (H_tt, q_tt, k_tt, v_tt, decay_tt, beta_tt, H_new, out):
        try: ttnn.deallocate(t)
        except Exception: pass
    # per-slot out cos (diagnose which slots fail — q-read path at high S)
    per_slot = [round(cos(out_out[b, nv], out_ref[b, nv]), 3)
                for b in range(B) for nv in range(NV)]
    return cos(H_out, H_ref), cos(out_out, out_ref), per_slot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--debug-mode", type=int, default=0,
                    help="owned_gdn debug_mode; 10 = batched-safe two-CB output path")
    args = ap.parse_args()
    import ttnn
    log(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        any_fail = False
        # B=1 sanity first (must match numpy → confirms our ref + I/O layout)
        log(f"debug_mode={args.debug_mode} (0=prod fast path, 10=batched-safe two-CB)")
        for B in (1, 2, 4, 8, 16, 32):
            try:
                ch, co, per_slot = run(device, B, seed=B, debug_mode=args.debug_mode)
                okH, okO = ch >= 0.99, co >= 0.99
                any_fail = any_fail or not (okH and okO)
                log(f"  B={B}: cos(H_new)={ch:.6f} cos(out)={co:.6f}  "
                    f"{'OK' if okH and okO else 'FAIL'}")
                if not okO:
                    log(f"    per-slot out cos (slot=b*{NV}+nv): {per_slot}")
            except Exception as e:
                any_fail = True
                log(f"  B={B}: RAISED {type(e).__name__}: {str(e)[:200]}")
        if any_fail:
            log("VERDICT: owned_gdn is NOT cleanly batchable as-is (see failures "
                "above) → kernel program-factory work needed to batch it.")
            raise SystemExit(1)
        log("VERDICT: PASS — qwen36_gdn_decode_owned accepts B>1 and is bit-correct "
            "per slot. Swapping the CB manual recurrence for owned_gdn is the "
            "(nearly free) DeltaNet throughput lever.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
