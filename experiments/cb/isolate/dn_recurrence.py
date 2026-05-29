#!/usr/bin/env python3
"""CB1 isolation — does the DeltaNet decode recurrence batch correctly?

Continuous batching needs the per-chip GatedDeltaNet recurrence to run
with a leading batch dim B (one slot per concurrent sequence), each slot
carrying its own recurrent state H_t. Before touching the production
server_tp.py forward, validate that the ttnn ops the recurrence is built
from broadcast/reduce correctly per-slot with B>1.

The decode-step gated-delta-rule recurrence (mirrors server_tp.py /
server_35b_ttnn.py manual path), per chip, per step:
    state = prev_state * g            # decay   [B,NV,K,V] * [B,NV,1,1]
    kv    = sum(state * k_col, -2)    # read    [B,NV,K,V]*[B,NV,K,1]->[B,NV,1,V]
    delta = beta * (v - kv)           # [B,NV,1,V]
    state = state + k_col * delta     # rank-1 update (outer k⊗delta)
    out   = sum(state * q_col, -2)    # query   ->[B,NV,1,V]

Gate: per-slot ttnn output vs a per-slot numpy reference, AND the batched
B=8 run must match 8 independent B=1 runs (slot independence — no
cross-slot leakage). cos > 0.999 per slot.

Single device suffices — the recurrence is per-chip work; batching is the
leading dim, orthogonal to TP sharding.

Run on qb1:
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb/isolate/dn_recurrence.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Small per-chip-like shapes for a fast, clear test. NV=4 value heads,
# K=V=128 head dims (27B uses 128). Production NV_PER_CHIP=12; the math
# is identical, NV just scales.
NV = 4
KDIM = 128
VDIM = 128


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def numpy_recurrence(prev_state, q, k, v, g, beta):
    """fp64 reference. Shapes: prev_state [B,NV,K,V], q/k [B,NV,K], v [B,NV,V],
    g/beta [B,NV]. Returns (new_state [B,NV,K,V], out [B,NV,V])."""
    g_ = g[..., None, None]            # [B,NV,1,1]
    state = prev_state * g_            # decay
    k_col = k[..., :, None]            # [B,NV,K,1]
    kv = (state * k_col).sum(axis=-2)  # [B,NV,V]
    delta = beta[..., None] * (v - kv) # [B,NV,V]
    # rank-1 update: state += k ⊗ delta  -> [B,NV,K,V]
    state = state + k_col * delta[..., None, :]
    q_col = q[..., :, None]            # [B,NV,K,1]
    out = (state * q_col).sum(axis=-2) # [B,NV,V]
    return state, out


def ttnn_recurrence(device, prev_state_np, q_np, k_np, v_np, g_np, beta_np):
    """ttnn implementation mirroring server manual DN path, with batch dim B."""
    import ttnn
    B = prev_state_np.shape[0]

    def to_tt(x):
        return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x.astype(np.float32))),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    state = to_tt(prev_state_np)               # [B,NV,K,V]
    q = to_tt(q_np); k = to_tt(k_np); v = to_tt(v_np)
    g = to_tt(g_np.reshape(B, NV, 1, 1))
    beta = to_tt(beta_np.reshape(B, NV, 1))

    # decay
    state = ttnn.mul(state, g)                 # [B,NV,K,V] * [B,NV,1,1] broadcast
    # kv = sum(state * k_col, -2)
    k_col = ttnn.reshape(k, [B, NV, KDIM, 1])
    state_k = ttnn.mul(state, k_col)
    kv = ttnn.sum(state_k, dim=-2)             # [B,NV,V] (or [B,NV,1,V])
    ttnn.deallocate(state_k)
    kv_3d = ttnn.reshape(kv, [B, NV, VDIM])
    ttnn.deallocate(kv)
    # delta = beta * (v - kv)
    v_minus = ttnn.sub(v, kv_3d)
    ttnn.deallocate(kv_3d)
    delta = ttnn.mul(v_minus, beta)            # [B,NV,V] * [B,NV,1]
    ttnn.deallocate(v_minus)
    # state += k_col * delta_row  (outer product)
    delta_row = ttnn.reshape(delta, [B, NV, 1, VDIM])
    ttnn.deallocate(delta)
    update = ttnn.mul(k_col, delta_row)        # [B,NV,K,1]*[B,NV,1,V] -> [B,NV,K,V]
    ttnn.deallocate(k_col); ttnn.deallocate(delta_row)
    state_new = ttnn.add(state, update)
    ttnn.deallocate(state); ttnn.deallocate(update)
    # out = sum(state_new * q_col, -2)
    q_col = ttnn.reshape(q, [B, NV, KDIM, 1])
    state_q = ttnn.mul(state_new, q_col)
    ttnn.deallocate(q_col)
    out = ttnn.sum(state_q, dim=-2)            # [B,NV,V]
    ttnn.deallocate(state_q)
    out_3d = ttnn.reshape(out, [B, NV, VDIM])
    ttnn.deallocate(out)

    state_np = ttnn.to_torch(state_new).float().numpy().reshape(B, NV, KDIM, VDIM)
    out_np = ttnn.to_torch(out_3d).float().numpy().reshape(B, NV, VDIM)
    ttnn.deallocate(state_new); ttnn.deallocate(out_3d)
    for t in (q, k, v, g, beta):
        try: ttnn.deallocate(t)
        except Exception: pass
    return state_np, out_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pcc-threshold", type=float, default=0.999)
    args = ap.parse_args()

    import ttnn
    log(f"opening device {args.device_id}")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        rng = np.random.default_rng(args.seed)
        B = args.batch

        def make_inputs(n):
            return dict(
                prev_state=rng.normal(0, 0.3, (n, NV, KDIM, VDIM)).astype(np.float32),
                q=rng.normal(0, 0.5, (n, NV, KDIM)).astype(np.float32),
                k=rng.normal(0, 0.5, (n, NV, KDIM)).astype(np.float32),
                v=rng.normal(0, 0.5, (n, NV, VDIM)).astype(np.float32),
                g=rng.uniform(0.9, 1.0, (n, NV)).astype(np.float32),   # decay near 1
                beta=rng.uniform(0.0, 1.0, (n, NV)).astype(np.float32),
            )

        log(f"=== batched B={B} vs numpy reference ===")
        inp = make_inputs(B)
        ref_state, ref_out = numpy_recurrence(
            inp["prev_state"].astype(np.float64), inp["q"].astype(np.float64),
            inp["k"].astype(np.float64), inp["v"].astype(np.float64),
            inp["g"].astype(np.float64), inp["beta"].astype(np.float64))
        tt_state, tt_out = ttnn_recurrence(
            device, inp["prev_state"], inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"])

        any_fail = False
        for b in range(B):
            c_out = cos(tt_out[b], ref_out[b])
            c_state = cos(tt_state[b], ref_state[b])
            ok = c_out >= args.pcc_threshold and c_state >= args.pcc_threshold
            if not ok:
                any_fail = True
            log(f"  slot {b}: cos(out)={c_out:.6f}  cos(state)={c_state:.6f}  "
                f"{'OK' if ok else 'FAIL'}")

        # Slot-independence: run each slot as its OWN B=1 call; batched output
        # must match (no cross-slot leakage).
        log(f"=== slot-independence: B={B} batched vs {B}× B=1 ===")
        max_slot_diff = 0.0
        for b in range(B):
            single = {kk: vv[b:b+1] for kk, vv in inp.items()}
            s1_state, s1_out = ttnn_recurrence(
                device, single["prev_state"], single["q"], single["k"],
                single["v"], single["g"], single["beta"])
            d = float(np.max(np.abs(s1_out[0] - tt_out[b])))
            max_slot_diff = max(max_slot_diff, d)
        log(f"  max |batched - per-slot| out diff: {max_slot_diff:.4e}  "
            f"({'OK — slot-independent' if max_slot_diff < 1e-2 else 'FAIL — cross-slot leak'})")

        if any_fail or max_slot_diff >= 1e-2:
            log("FAIL: batched DN recurrence does not match reference / leaks across slots.")
            raise SystemExit(1)
        log(f"PASS: DN recurrence batches correctly at B={B}. CB1 DN path is "
            f"a shape change (add leading B), not an algorithm change.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
