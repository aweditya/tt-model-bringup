#!/usr/bin/env python3
"""
Probe: ttnn.transformer.scaled_dot_product_attention_decode at large
cur_pos (the only remaining open risk in the C'0.5 long-context plan).

Background: A4 validated SDPA-decode at KV_LEN=128. For daily-driver use
we need 32k+. Does SDPA-decode handle cur_pos in the thousands? The op
might have undocumented internal limits (tile shapes, page sizes, etc.)
that show up only at large positions.

What this probe answers:
1. Does SDPA-decode complete without error at cur_pos ∈ {0, 127, 1000,
   8192, 16384, 30000, 32767}?
2. Is the numerical output correct vs a numpy reference that does the
   same attention over the first cur_pos+1 cache slots?
3. Wall-clock cost growth with cur_pos — is it linear, sub-linear, or
   does something nasty happen at certain boundaries?

Run on qb1 (qb2 busy with C'0.6 gates):
    cd ~/tt-xla && .venv/bin/python experiments/utils/sdpa_long_pos_probe.py
"""
import sys
import time
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# Qwen3.6-27B shapes
N_KV = 4
N_Q = 32           # n_q_heads
HEAD_DIM = 256
MAX_POS = 32768    # 32k context (tile-aligned: 32768 / 32 = 1024)


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def numpy_sdpa_decode(q, k_cache, v_cache, cur_pos, n_kv, n_q, head_dim):
    """SDPA-decode reference. q: [n_q, head_dim]. k_cache, v_cache: [n_kv, max_pos, head_dim].
    Computes attention from q against k_cache[:, :cur_pos+1, :] / v_cache[:, :cur_pos+1, :].
    Returns: [n_q, head_dim]."""
    n_rep = n_q // n_kv
    # Repeat KV to match Q heads (GQA)
    k_repeated = np.repeat(k_cache, n_rep, axis=0)   # [n_q, max_pos, head_dim]
    v_repeated = np.repeat(v_cache, n_rep, axis=0)
    # Slice up to cur_pos (inclusive)
    k = k_repeated[:, :cur_pos+1, :]                  # [n_q, cur_pos+1, head_dim]
    v = v_repeated[:, :cur_pos+1, :]
    # Attention scores: q [n_q, 1, head_dim] @ k.T [n_q, head_dim, L]
    scores = np.einsum('hd,hld->hl', q, k) / np.sqrt(head_dim)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    out = np.einsum('hl,hld->hd', weights, v)
    return out.astype(np.float32)


def main():
    print("=" * 64)
    print(f"Probe: SDPA-decode at large cur_pos")
    print(f"  MAX_POS={MAX_POS}, N_KV={N_KV}, N_Q={N_Q}, HEAD_DIM={HEAD_DIM}")
    print("=" * 64)

    test_positions = [0, 127, 1000, 4096, 8192, 16384, 30000, 32767]

    rng = np.random.default_rng(42)
    # Build the FULL cache with KNOWN random values, ~unit norm
    k_cache_np = rng.standard_normal((1, N_KV, MAX_POS, HEAD_DIM)).astype(np.float32) * 0.1
    v_cache_np = rng.standard_normal((1, N_KV, MAX_POS, HEAD_DIM)).astype(np.float32) * 0.1

    device = ttnn.open_device(device_id=0)
    try:
        # Upload the full caches once
        print("\nUploading caches (one-time, ~32 MB each)...")
        kv_k_tt = ttnn.from_torch(
            torch.from_numpy(k_cache_np), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT)
        kv_v_tt = ttnn.from_torch(
            torch.from_numpy(v_cache_np), dtype=ttnn.bfloat16,
            device=device, layout=ttnn.TILE_LAYOUT)

        hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, math_approx_mode=False)

        print(f"\n{'cur_pos':>8} {'wall_ms':>10} {'cos vs np':>11} {'max|Δ|':>10}  result")
        print("-" * 60)

        for cur_pos in test_positions:
            # Build Q for this position
            q_np = rng.standard_normal((N_Q, HEAD_DIM)).astype(np.float32) * 0.1

            # Numpy reference
            ref = numpy_sdpa_decode(
                q_np,
                k_cache_np[0], v_cache_np[0],
                cur_pos, N_KV, N_Q, HEAD_DIM)

            # ttnn call
            q_tt = ttnn.from_torch(
                torch.from_numpy(q_np.reshape(1, 1, N_Q, HEAD_DIM)),
                dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            cur_pos_tt = ttnn.from_torch(
                torch.tensor([cur_pos], dtype=torch.int32), device=device)

            try:
                ttnn.synchronize_device(device)
                t0 = time.time()
                attn = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_tt, kv_k_tt, kv_v_tt,
                    cur_pos_tensor=cur_pos_tt,
                    compute_kernel_config=hifi4)
                ttnn.synchronize_device(device)
                t1 = time.time()
                attn_np = ttnn.to_torch(attn).float().cpu().numpy().reshape(N_Q, HEAD_DIM)
                cos = _cosine(attn_np, ref)
                max_diff = float(np.abs(attn_np - ref).max())
                wall_ms = (t1 - t0) * 1000
                ok = (cos > 0.99 and max_diff < 0.2)  # bf16 tolerance
                tag = "✓ OK" if ok else "⚠ DRIFT"
                print(f"{cur_pos:>8} {wall_ms:>10.2f} {cos:>11.6f} {max_diff:>10.4f}  {tag}")
            except Exception as e:
                msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                print(f"{cur_pos:>8}        ✗ EXCEPTION: {msg[:80]}")

        print()
        print("Interpretation:")
        print("  - If ALL positions pass → SDPA-decode handles 32k context cleanly.")
        print("    C'0.5 (MAX_POS scaleup) can use stock SDPA-decode unchanged.")
        print("  - If only some positions fail → there's a size/alignment threshold.")
        print("    Need to investigate and pad MAX_POS to the next safe value.")
        print("  - If all positions fail → SDPA-decode has a hard limit; we'd need")
        print("    a chunked SDPA-prefill or chunked SDPA-decode workaround.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
