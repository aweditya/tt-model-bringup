#!/usr/bin/env python3
"""
P1 — mesh-sharded SDPA + update_cache_for_token_ correctness probe (qb2).

Goal: validate the two riskiest ops in server_tp.py Stage C's gated_attn_step_tp
before any 17-min server bootstrap. If either op fails on a sharded KV cache,
we learn the constraint NOW and rearchitect.

Setup mirrors per-chip view in the production TP path:
  - Global KV cache: [batch=1, N_KV=4, MAX_POS=32, HEAD_DIM=256]
  - Shard dim=1 (N_KV) across 4 chips → per-chip [1, 1, MAX_POS, HEAD_DIM]
  - GQA: N_Q=24 globally; 6 Q heads per chip attending to its 1 KV head
  - Q per chip: [1, 1, 6, HEAD_DIM]

Operations probed:
  1. ttnn.kv_cache.update_cache_for_token_(cache, src, cur_pos)
     - cache: mesh-sharded [1, 1, MAX_POS, HEAD_DIM] per chip
     - src:   mesh-sharded [1, 1, 1, HEAD_DIM] per chip
     - cur_pos: int
     - Question: does this run per-chip, mutating each chip's slab in place?

  2. ttnn.transformer.scaled_dot_product_attention_decode(q, kc, vc, cur_pos=...)
     - q: mesh-sharded [1, 1, 6, HEAD_DIM] per chip
     - kc, vc: mesh-sharded [1, 1, MAX_POS, HEAD_DIM] per chip
     - cur_pos: list or replicated int32 tensor
     - Question: per-chip SDPA on sharded inputs?

Math validation:
  - Construct K_global, V_global on host, then a Q_global
  - Pre-fill all but the last position; insert a fresh K/V at L via the op
  - Compute numpy attention(Q_global, K_global, V_global) per-head
  - Compare per-chip output vs corresponding per-head numpy gold

Pass criteria:
  - update_cache_for_token_ returns without error
  - SDPA returns without error
  - per-chip output cos vs numpy gold ≥ 0.999

Wall: ~4 min (mesh open + JIT + 2 op invocations).
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# Qwen3.6-27B gated-attention shapes
HIDDEN = 5120
N_Q = 24
N_KV = 4
HEAD_DIM = 256
NCHIPS = 4
NQ_PER_CHIP = N_Q // NCHIPS   # 6
NKV_PER_CHIP = N_KV // NCHIPS  # 1
N_REP = N_Q // N_KV            # 6 — GQA broadcast factor
MAX_POS = 32                   # small probe shape


def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def numpy_gold(q_global, k_global, v_global, cur_pos):
    """Reference per-head attention (no scale baked into q).

    q_global: [N_Q, HEAD_DIM]
    k_global, v_global: [N_KV, MAX_POS, HEAD_DIM] — positions 0..cur_pos active
    Returns: [N_Q, HEAD_DIM]
    """
    out = np.zeros((N_Q, HEAD_DIM), dtype=np.float32)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    for qi in range(N_Q):
        kv_idx = qi // N_REP
        k_head = k_global[kv_idx, : cur_pos + 1, :]  # [L, HEAD_DIM]
        v_head = v_global[kv_idx, : cur_pos + 1, :]
        scores = (q_global[qi] @ k_head.T) * scale   # [L]
        attn = softmax_np(scores)
        out[qi] = attn @ v_head
    return out


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 78)
    print("P1: mesh-sharded update_cache_for_token_ + SDPA decode probe (qb2)")
    print("=" * 78)
    print(f"Per chip: KV cache [1, {NKV_PER_CHIP}, {MAX_POS}, {HEAD_DIM}]")
    print(f"          Q       [1, 1, {NQ_PER_CHIP}, {HEAD_DIM}]")
    print(f"MAX_POS={MAX_POS}, N_REP={N_REP} (each KV head broadcast over {N_REP} Q heads)")

    print("\n[1] Init fabric + open mesh...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ mesh {mesh.get_num_devices()} chips")

    try:
        rng = np.random.default_rng(42)

        # Pre-fill cache at positions 0..L-1; we'll insert fresh K/V at L.
        L = 5  # cur_pos where we insert
        # Global K/V at full shape
        K_global = rng.standard_normal((1, N_KV, MAX_POS, HEAD_DIM)).astype(np.float32) * 0.05
        V_global = rng.standard_normal((1, N_KV, MAX_POS, HEAD_DIM)).astype(np.float32) * 0.05
        # Generate the fresh K/V that will be written into slot L
        K_fresh = rng.standard_normal((1, N_KV, 1, HEAD_DIM)).astype(np.float32) * 0.05
        V_fresh = rng.standard_normal((1, N_KV, 1, HEAD_DIM)).astype(np.float32) * 0.05
        # Pre-populated cache initially has whatever was at position L from the
        # original random fill; the op should OVERWRITE it with K_fresh.
        # Numpy gold uses the cache AFTER the write.
        K_after = K_global.copy()
        V_after = V_global.copy()
        K_after[:, :, L, :] = K_fresh[:, :, 0, :]
        V_after[:, :, L, :] = V_fresh[:, :, 0, :]

        # Q at this step
        Q = rng.standard_normal((N_Q, HEAD_DIM)).astype(np.float32) * 0.1

        # --- Upload sharded to mesh ---
        print("\n[2] Upload sharded KV cache + Q to mesh...")
        # Cache: shard along dim=1 (N_KV axis) → per chip [1, 1, MAX_POS, HEAD_DIM]
        kc_tt = ttnn.from_torch(torch.from_numpy(K_global), dtype=ttnn.bfloat16,
                                  device=mesh, layout=ttnn.TILE_LAYOUT,
                                  mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
        vc_tt = ttnn.from_torch(torch.from_numpy(V_global), dtype=ttnn.bfloat16,
                                  device=mesh, layout=ttnn.TILE_LAYOUT,
                                  mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
        # Fresh K/V (same shard pattern)
        k_fresh_tt = ttnn.from_torch(torch.from_numpy(K_fresh), dtype=ttnn.bfloat16,
                                       device=mesh, layout=ttnn.TILE_LAYOUT,
                                       mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
        v_fresh_tt = ttnn.from_torch(torch.from_numpy(V_fresh), dtype=ttnn.bfloat16,
                                       device=mesh, layout=ttnn.TILE_LAYOUT,
                                       mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
        # Q: shard along the Q-head dim. Build a global Q tensor with the
        # per-chip layout [1, 1, N_Q_PER_CHIP, HEAD_DIM]. We pack as
        # [1, N_CHIPS, N_Q_PER_CHIP, HEAD_DIM] then shard dim=1.
        Q_packed = Q.reshape(1, NCHIPS, NQ_PER_CHIP, HEAD_DIM)
        q_tt = ttnn.from_torch(torch.from_numpy(Q_packed), dtype=ttnn.bfloat16,
                                 device=mesh, layout=ttnn.TILE_LAYOUT,
                                 mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
        print("  ✓ uploaded")

        # --- TEST OP 1: update_cache_for_token_ on sharded cache ---
        print(f"\n[3] OP 1: update_cache_for_token_(cache, fresh_kv, cur_pos={L})")
        try:
            ttnn.kv_cache.update_cache_for_token_(kc_tt, k_fresh_tt, L)
            ttnn.kv_cache.update_cache_for_token_(vc_tt, v_fresh_tt, L)
            print(f"  ✓ both calls returned without error")
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:400]}")
            print("\n  → CONSTRAINT IDENTIFIED: update_cache_for_token_ doesn't accept"
                  " mesh-sharded cache. Need alternative (e.g. manual ttnn.copy +"
                  " in-place slice update).")
            return

        # --- TEST OP 2: scaled_dot_product_attention_decode on sharded inputs ---
        print(f"\n[4] OP 2: scaled_dot_product_attention_decode(q, kc, vc, cur_pos={L})")
        try:
            try:
                out_tt = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_tt, kc_tt, vc_tt,
                    cur_pos=[L],
                    scale=1.0 / (HEAD_DIM ** 0.5),
                )
                print(f"  ✓ ran with cur_pos=[{L}]")
            except Exception as e1:
                print(f"  ✗ cur_pos=list failed: {type(e1).__name__}: {str(e1)[:200]}")
                # Try with replicated cur_pos tensor
                cur_pos_tt = ttnn.from_torch(
                    torch.tensor([L], dtype=torch.int32),
                    device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                out_tt = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_tt, kc_tt, vc_tt,
                    cur_pos_tensor=cur_pos_tt,
                    scale=1.0 / (HEAD_DIM ** 0.5),
                )
                print(f"  ✓ ran with replicated cur_pos_tensor")
        except Exception as e:
            print(f"  ✗ FAILED both invocations: {type(e).__name__}: {str(e)[:400]}")
            print("\n  → CONSTRAINT IDENTIFIED: scaled_dot_product_attention_decode"
                  " doesn't accept mesh-sharded KV. Need manual Q@K^T softmax V"
                  " (like C'7.3 probe).")
            return

        # --- Math validation ---
        print("\n[5] Math validation vs numpy fp32 gold...")
        gold = numpy_gold(Q, K_after[0], V_after[0], L)  # [N_Q, HEAD_DIM]
        # Reshape per-chip output back to global view
        # out_tt is per-chip [1, 1, NQ_PER_CHIP, HEAD_DIM]; concat along dim=1
        # over the mesh gives [1, NCHIPS, NQ_PER_CHIP, HEAD_DIM] = [1, N_Q, HEAD_DIM]
        out_torch = ttnn.to_torch(out_tt,
                                    mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
        out_np = out_torch.float().cpu().numpy()
        # Reshape to [N_Q, HEAD_DIM]
        if out_np.ndim == 4:
            out_np = out_np.reshape(out_np.shape[1] * out_np.shape[2], HEAD_DIM)[:N_Q]
        elif out_np.ndim == 3:
            out_np = out_np.reshape(-1, HEAD_DIM)[:N_Q]
        cos = _cosine(out_np, gold)
        max_diff = float(np.abs(out_np - gold).max())
        print(f"  out shape: {out_np.shape}, gold: {gold.shape}")
        print(f"  cos(mesh SDPA, numpy gold) = {cos:.6f}")
        print(f"  max|Δ| = {max_diff:.4e}")

        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        ok_math = cos >= 0.999
        print(f"  Both ops ran ✓")
        print(f"  cos vs gold: {cos:.6f} {'✓' if ok_math else '✗'}")
        if ok_math:
            print("  P1 PASSES — server_tp's gated_attn_step_tp KV+SDPA path is viable.")
        else:
            print("  P1 PARTIAL — ops execute but math is wrong; investigate before"
                  " building on this path.")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
            print("\n  ✓ mesh closed")
        except Exception as e:
            print(f"  ✗ close error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception as e:
            print(f"  fabric reset: {e}")


if __name__ == "__main__":
    main()
