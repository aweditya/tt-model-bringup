#!/usr/bin/env python3
"""
Probe #4: numerical correctness of paged_scaled_dot_product_attention_decode
at GQA shape (N_Q=32, N_KV=4 — ratio 8).

The smoke tests confirmed the op runs without error at our shape and scales
to MAX_POS=32k. This is the gate that decides whether the op also computes
CORRECT GQA math, or whether it silently treats the cache as MQA (collapses
KV heads) and produces wrong output.

Methodology:
  - Build a paged cache with KNOWN random values (fixed seed)
  - Build a Q tensor with known values
  - Compute paged SDPA decode output
  - Compute a numpy reference that explicitly handles GQA via np.repeat:
      kv_head(q_head) = q_head // (N_Q // N_KV) = q_head // 8
  - Compare per-q-head cosine and max|Δ| against the numpy reference

Three diagnostic outcomes:
  - All 32 Q heads cos > 0.99 → correct GQA, integration cleared
  - Only the first head in each group (0, 8, 16, 24) high cos → silent MQA
    (op collapses KV heads); workaround needed
  - All low → something else wrong; deeper investigation

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/utils/paged_sdpa_gqa_correctness_probe.py
"""
import sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

B = 1
N_Q = 32
N_KV = 4
N_REP = N_Q // N_KV    # GQA ratio = 8
P = 64                  # block size
HD = 256
MAX_BLOCKS = 8
CUR_POS = 100           # mid-block 1, plenty of valid prefix


def numpy_sdpa_gqa_decode(q, k_per_head, v_per_head, cur_pos, n_q, n_kv, head_dim):
    """Reference: each q_head h attends to kv_head h // (n_q/n_kv)."""
    rep = n_q // n_kv
    out = np.zeros((n_q, head_dim), dtype=np.float32)
    for h in range(n_q):
        kv_h = h // rep
        k = k_per_head[kv_h, :cur_pos + 1, :]    # [L, HD]
        v = v_per_head[kv_h, :cur_pos + 1, :]
        scores = (q[h] @ k.T) / np.sqrt(head_dim)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        out[h] = weights @ v
    return out


def paged_to_logical(paged_cache):
    """Convert [max_blocks, n_kv, P, HD] → [n_kv, max_blocks*P, HD]."""
    return paged_cache.transpose(1, 0, 2, 3).reshape(N_KV, MAX_BLOCKS * P, HD)


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 64)
    print("Probe #4: paged SDPA decode — GQA numerical correctness")
    print(f"  N_Q={N_Q}, N_KV={N_KV}, N_REP={N_REP}, HD={HD}, P={P}")
    print(f"  cur_pos={CUR_POS}, max_blocks={MAX_BLOCKS}")
    print("=" * 64)

    rng = np.random.default_rng(42)

    # Cache values with KNOWN seed
    cache_shape = (MAX_BLOCKS, N_KV, P, HD)
    k_paged_np = rng.standard_normal(cache_shape).astype(np.float32) * 0.1
    v_paged_np = rng.standard_normal(cache_shape).astype(np.float32) * 0.1

    # Q tensor: each head with distinctive values
    q_np = rng.standard_normal((N_Q, HD)).astype(np.float32) * 0.1

    # Compute numpy reference
    k_logical_np = paged_to_logical(k_paged_np)   # [N_KV, max_pos, HD]
    v_logical_np = paged_to_logical(v_paged_np)
    ref_out = numpy_sdpa_gqa_decode(q_np, k_logical_np, v_logical_np,
                                     CUR_POS, N_Q, N_KV, HD)
    print(f"  numpy reference computed, shape {ref_out.shape}")

    device = ttnn.open_device(device_id=0)
    try:
        keys_tt = ttnn.from_torch(torch.from_numpy(k_paged_np), dtype=ttnn.bfloat16,
                                   device=device, layout=ttnn.TILE_LAYOUT)
        vals_tt = ttnn.from_torch(torch.from_numpy(v_paged_np), dtype=ttnn.bfloat16,
                                   device=device, layout=ttnn.TILE_LAYOUT)
        q_tt = ttnn.from_torch(torch.from_numpy(q_np.reshape(1, B, N_Q, HD)),
                                dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        page_table_np = np.arange(MAX_BLOCKS, dtype=np.int32).reshape(B, MAX_BLOCKS)
        page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                         dtype=ttnn.int32, device=device,
                                         layout=ttnn.ROW_MAJOR_LAYOUT)
        cur_pos_tt = ttnn.from_torch(torch.from_numpy(np.array([CUR_POS], dtype=np.int32)),
                                      dtype=ttnn.int32, device=device,
                                      layout=ttnn.ROW_MAJOR_LAYOUT)

        hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, math_approx_mode=False,
        )

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt, keys_tt, vals_tt, page_table_tt,
            cur_pos_tensor=cur_pos_tt,
            compute_kernel_config=hifi4,
        )
        ttnn.synchronize_device(device)
        out_np = ttnn.to_torch(attn).float().cpu().numpy()
        out_np = out_np.reshape(N_Q, HD)
        print(f"  ttnn output shape: {out_np.shape}")

        # Compare per Q-head
        print(f"\n{'q_head':>7} {'kv_head':>8} {'cosine':>10} {'max|Δ|':>10}  {'verdict':>10}")
        print("-" * 60)
        per_group_first_cos = {}
        per_group_others_cos = {}
        all_pass = True
        for h in range(N_Q):
            kv_h = h // N_REP
            cos = _cosine(out_np[h], ref_out[h])
            md = float(np.abs(out_np[h] - ref_out[h]).max())
            is_first = (h % N_REP == 0)
            tag = "✓" if cos > 0.99 else ("⚠" if cos > 0.9 else "✗")
            if cos < 0.99:
                all_pass = False
            print(f"{h:>7} {kv_h:>8} {cos:>10.6f} {md:>10.4f}  {tag:>10}")
            if is_first:
                per_group_first_cos.setdefault(kv_h, cos)
            else:
                per_group_others_cos.setdefault(kv_h, []).append(cos)

        # Diagnostic
        print()
        if all_pass:
            print("✓ ALL 32 Q-heads agree with the GQA numpy reference at cosine > 0.99.")
            print("  Paged SDPA decode handles GQA correctly. No custom kernel needed.")
        else:
            # Check the silent-MQA signature
            firsts_high = all(c > 0.99 for c in per_group_first_cos.values())
            others_low = all(c < 0.5 for cs in per_group_others_cos.values() for c in cs)
            if firsts_high and others_low:
                print("✗ SILENT MQA: only the first q-head in each group matches; others are noise.")
                print("  The op collapses N_KV heads. Custom kernel or Python workaround required.")
            else:
                print("✗ MIXED: some heads pass, some fail. Investigate per-head pattern.")
                print(f"  First-of-group cosines: {per_group_first_cos}")
                for k, v in per_group_others_cos.items():
                    print(f"  Group {k} others: mean={np.mean(v):.4f}, min={np.min(v):.4f}, max={np.max(v):.4f}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
