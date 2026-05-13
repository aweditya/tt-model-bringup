#!/usr/bin/env python3
"""
Probe: numerical correctness of paged_scaled_dot_product_attention_decode at
Qwen3.6-27B shapes (N_Q=24, N_KV=4, HEAD_DIM=256, GQA ratio = 6).

Prior probe (`feedback_paged_sdpa_decode_works_at_32k.md`) verified the op
runs without crashing from MAX_POS=256 through 32k. The OPEN QUESTION: does
it compute correct GQA math, or does it silently MQA-flatten?

This probe is the unblocker for the paged SDPA migration (which gates
long-context daily-driver use). If correctness holds, we migrate.

Method:
  1. Build random K, V caches of shape [1, N_KV, MAX_POS, HEAD_DIM] bf16 and
     a Q of shape [1, B=1, N_Q, HEAD_DIM] bf16.
  2. Compute a numpy reference attention output (proper GQA: each of N_Q
     queries attends to the matching kv head via group-mapping).
  3. Run NON-PAGED ttnn SDPA at MAX_POS=256 — sanity check vs numpy.
  4. Build paged cache of shape [max_num_blocks, N_KV, block_size, HD]
     plus page_table [1, max_num_blocks] that re-presents the SAME data.
  5. Run PAGED ttnn SDPA — compare to numpy ref.
  6. Repeat at MAX_POS=1024 (where non-paged fails) — paged-only.
  7. Repeat at MAX_POS=8192 — long-context viability.

If paged cosine vs numpy ≥ 0.999 across all MAX_POS values, the migration
is fully unblocked. If it's much lower than non-paged-vs-numpy, GQA is
broken.

Run on qb2 (device 0; the persistent server is on device 0 already — kill
the server temporarily OR set TT_DEVICE_ID=1):
    cd ~/tt-xla && TT_DEVICE_ID=1 .venv/bin/python experiments/utils/paged_sdpa_numerical_probe.py
"""
import os, sys, time
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

# Qwen3.6-27B shape
N_Q = 24
N_KV = 4
HEAD_DIM = 256
N_REP = N_Q // N_KV    # GQA group size: each KV head serves N_REP query heads
BLOCK_SIZE = 64        # paged cache block size (per `feedback_paged_sdpa_decode_works_at_32k`)


def numpy_gqa_attention(q, k, v, cur_pos):
    """Reference attention for one decode step (cur_pos is the LAST written position).

    q: [1, 1, N_Q, HEAD_DIM] fp32
    k, v: [1, N_KV, MAX_POS, HEAD_DIM] fp32 (positions > cur_pos are garbage / masked)
    cur_pos: int — only positions [0, cur_pos] are attended

    Returns: [1, 1, N_Q, HEAD_DIM] fp32
    """
    q_f = q.astype(np.float64)
    k_f = k.astype(np.float64)
    v_f = v.astype(np.float64)

    # Reshape Q to [N_KV, N_REP, HEAD_DIM] so we can broadcast over the KV group
    q_reshaped = q_f[0, 0].reshape(N_KV, N_REP, HEAD_DIM)  # [N_KV, N_REP, HD]

    out = np.zeros((N_KV, N_REP, HEAD_DIM), dtype=np.float64)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    for kv in range(N_KV):
        # K, V for this kv head, only valid positions
        k_kv = k_f[0, kv, :cur_pos + 1, :]   # [cur_pos+1, HD]
        v_kv = v_f[0, kv, :cur_pos + 1, :]
        for rep in range(N_REP):
            q_h = q_reshaped[kv, rep, :]                # [HD]
            scores = (k_kv @ q_h) * scale               # [cur_pos+1]
            # causal mask (everything <= cur_pos is allowed)
            scores_max = scores.max()
            p = np.exp(scores - scores_max)
            p /= p.sum()
            out[kv, rep, :] = p @ v_kv

    return out.reshape(1, 1, N_Q, HEAD_DIM).astype(np.float32)


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _max_abs(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())


def alloc_tt(arr, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
              memory_config=None):
    kw = dict(dtype=dtype, device=device, layout=layout)
    if memory_config is not None:
        kw["memory_config"] = memory_config
    return ttnn.from_torch(torch.from_numpy(arr), **kw)


def make_paged_cache(non_paged_cache, block_size):
    """Reshape [1, N_KV, MAX_POS, HD] -> [max_num_blocks, N_KV, block_size, HD].

    For single-user, page_table is an identity mapping [0, 1, ..., n_blocks-1].
    """
    _, n_kv, max_pos, hd = non_paged_cache.shape
    assert max_pos % block_size == 0, f"max_pos {max_pos} must be multiple of block_size {block_size}"
    n_blocks = max_pos // block_size
    # [1, N_KV, n_blocks, block_size, HD] -> [n_blocks, N_KV, block_size, HD]
    paged = non_paged_cache.reshape(1, n_kv, n_blocks, block_size, hd).transpose(2, 0, 1, 3, 4)
    return paged.reshape(n_blocks, n_kv, block_size, hd)


def run_test(device, max_pos, cur_pos, run_unpaged=True, run_paged=True):
    print(f"\n=== MAX_POS={max_pos}, cur_pos={cur_pos} ===")
    assert max_pos % BLOCK_SIZE == 0, f"MAX_POS must be multiple of {BLOCK_SIZE}"

    np.random.seed(42)
    # Use small initial values (matching real attention regime)
    k_np = (np.random.randn(1, N_KV, max_pos, HEAD_DIM) * 0.05).astype(np.float32)
    v_np = (np.random.randn(1, N_KV, max_pos, HEAD_DIM) * 0.05).astype(np.float32)
    q_np = (np.random.randn(1, 1, N_Q, HEAD_DIM) * 0.05).astype(np.float32)

    # Mask out positions > cur_pos (so they don't contribute, sanity)
    k_np[0, :, cur_pos + 1:, :] = 0.0
    v_np[0, :, cur_pos + 1:, :] = 0.0

    # Numpy reference
    ref = numpy_gqa_attention(q_np, k_np, v_np, cur_pos)
    print(f"  numpy ref output norm: {float(np.linalg.norm(ref)):.4f}")

    # Upload to device
    q_tt = alloc_tt(q_np, device)
    k_tt = alloc_tt(k_np, device)
    v_tt = alloc_tt(v_np, device)
    cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32), device=device)

    # Test 1: NON-PAGED SDPA (should fit at MAX_POS=256)
    if run_unpaged:
        try:
            out_unpaged = ttnn.transformer.scaled_dot_product_attention_decode(
                q_tt, k_tt, v_tt, cur_pos_tensor=cur_pos_tt)
            ttnn.synchronize_device(device)
            unp = ttnn.to_torch(out_unpaged).float().cpu().numpy()
            cos_u = _cosine(ref, unp)
            mae_u = _max_abs(ref, unp)
            print(f"  [non-paged]  cos vs numpy = {cos_u:.6f}  max|Δ| = {mae_u:.4e}")
            unpaged_ok = cos_u > 0.999
        except Exception as e:
            print(f"  [non-paged]  FAILED: {type(e).__name__}: {str(e)[:120]}")
            unpaged_ok = None

    # Test 2: PAGED SDPA
    if run_paged:
        n_blocks = max_pos // BLOCK_SIZE
        # Build paged cache (re-presents same data in paged layout)
        k_paged_np = make_paged_cache(k_np, BLOCK_SIZE)
        v_paged_np = make_paged_cache(v_np, BLOCK_SIZE)
        # Page table: identity mapping for single user
        page_table_np = np.arange(n_blocks, dtype=np.int32).reshape(1, n_blocks)

        k_paged_tt = alloc_tt(k_paged_np, device)
        v_paged_tt = alloc_tt(v_paged_np, device)
        page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                          dtype=ttnn.int32, device=device,
                                          layout=ttnn.ROW_MAJOR_LAYOUT)
        cur_pos_tt2 = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                        device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        try:
            out_paged = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_tt, k_paged_tt, v_paged_tt, page_table_tt,
                cur_pos_tensor=cur_pos_tt2)
            ttnn.synchronize_device(device)
            pag = ttnn.to_torch(out_paged).float().cpu().numpy()
            cos_p = _cosine(ref, pag)
            mae_p = _max_abs(ref, pag)
            print(f"  [paged]      cos vs numpy = {cos_p:.6f}  max|Δ| = {mae_p:.4e}")
            paged_ok = cos_p > 0.999
            return paged_ok
        except Exception as e:
            print(f"  [paged]      FAILED: {type(e).__name__}: {str(e)[:200]}")
            return False


def main():
    device_id = int(os.environ.get("TT_DEVICE_ID", "0"))
    print(f"Probe: paged SDPA numerical correctness  device_id={device_id}")
    print(f"  shape: N_Q={N_Q} N_KV={N_KV} HEAD_DIM={HEAD_DIM} block_size={BLOCK_SIZE}")
    print(f"  GQA ratio: {N_REP}x (each KV head serves {N_REP} query heads)")

    device = ttnn.open_device(device_id=device_id)
    try:
        # Test 1: MAX_POS=256 — non-paged and paged BOTH run, compare both vs numpy
        run_test(device, max_pos=256, cur_pos=200, run_unpaged=True, run_paged=True)
        # Test 2: MAX_POS=1024 — non-paged fails (L1 overflow), paged should work
        run_test(device, max_pos=1024, cur_pos=900, run_unpaged=False, run_paged=True)
        # Test 3: MAX_POS=8192 — long context viability
        run_test(device, max_pos=8192, cur_pos=8000, run_unpaged=False, run_paged=True)

        print("\n=== VERDICT ===")
        print("  If all [paged] cosines >= 0.999, GQA correctness is validated and")
        print("  the migration is unblocked. Update memory with the result.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
