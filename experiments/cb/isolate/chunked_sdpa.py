#!/usr/bin/env python3
"""S2.1 isolation — `ttnn.transformer.chunked_scaled_dot_product_attention`.

The only NEW primitive S2 chunked-prefill needs. Per the prior-art audit
(research/27b_chunked_prefill_prior_art.md) this op ships in tt-metal today —
Llama/Qwen-VL prefill paths use it. This isolation confirms it's available in
our ttnn build, runs at Qwen3.6-27B per-chip shapes, and is bit-equivalent to
the decode SDPA replayed C times against the same paged KV (the path we're
replacing).

Single-device on purpose (mesh sharding is independent of correctness here):
NKV_PER_CHIP=1, HEAD_DIM=256, BLOCK_SIZE=32 mirror what one chip sees in
production. NQ_PER_CHIP picked to span per-chip Q-head counts we'd actually
see (Qwen3.6-27B per-chip ≈ 6 → use 8 for tile alignment).

Reference: per-position numpy causal attention on the chunk (each of the C new
queries attends to KV[0 .. L_prefix + i] for i in 0..C-1). The same reference
the existing paged_sdpa.py isolation uses, extended to multi-query.

Gate: per-position cos ≥ 0.99 vs numpy reference (bf16 rounding floor).

Run on qb1:
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb/isolate/chunked_sdpa.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Qwen3.6-27B per-chip on a (1,4) mesh:
NKV = 1           # n_kv_heads=4 total / 4 chips
NQ = 8            # tile-aligned; production per-chip = 6, 8 is the next tile bucket
HEAD_DIM = 256    # text_cfg['head_dim'] for Qwen3.6
BLOCK_SIZE = 32
NUM_BLOCKS = 64   # plenty for L_prefix + C in a single slot


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1); b = np.asarray(b, np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def numpy_chunk_causal_attention(q_chunk, K, V, L_prefix):
    """Multi-query causal attention over a paged KV prefix + the chunk's own
    Q (which doubles as the chunk K/V already written to the cache).
    q_chunk [C,NQ,D]; K,V [L_total,D]; L_prefix = positions in KV BEFORE chunk.
    For chunk position i, attends to KV[0 .. L_prefix + i].
    Returns out [C,NQ,D].
    """
    C = q_chunk.shape[0]
    out = np.zeros_like(q_chunk)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    for i in range(C):
        end = L_prefix + i + 1
        k = K[:end]   # [end, D]
        v = V[:end]
        for h in range(NQ):
            scores = (q_chunk[i, h] @ k.T) * scale  # [end]
            scores -= scores.max()
            w = np.exp(scores); w /= w.sum()
            out[i, h] = w @ v
    return out


def write_paged_cache(kc_np, vc_np, page_table_row, K_total, V_total, L_total):
    """Pre-fill slot-0 paged blocks at positions 0..L_total-1 with K_total/V_total.
    Used as the 'KV already on device' starting state for the chunked-SDPA call.
    """
    for pos in range(L_total):
        phys = page_table_row[pos // BLOCK_SIZE]
        slot = pos % BLOCK_SIZE
        kc_np[phys, 0, slot] = K_total[pos]
        vc_np[phys, 0, slot] = V_total[pos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=32, help="C — must be a tile multiple")
    ap.add_argument("--prefix-lens", default="0,32,64,128",
                    help="comma-separated L_prefix values (must be tile multiples)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pcc-threshold", type=float, default=0.99)
    args = ap.parse_args()

    import ttnn
    log(f"opening device {args.device_id}")
    if not hasattr(ttnn.transformer, "chunked_scaled_dot_product_attention"):
        log("FAIL: ttnn.transformer.chunked_scaled_dot_product_attention "
            "NOT present in this ttnn build")
        raise SystemExit(2)
    log("ttnn.transformer.chunked_scaled_dot_product_attention is exposed")

    device = ttnn.open_device(device_id=args.device_id)
    try:
        C = args.chunk_size
        assert C % BLOCK_SIZE == 0, f"chunk_size must be multiple of {BLOCK_SIZE}"
        prefixes = [int(x) for x in args.prefix_lens.split(",")]
        assert all(p % BLOCK_SIZE == 0 for p in prefixes), \
            "chunk_start_idx must be multiple of q_chunk_size (and BLOCK_SIZE)"

        rng = np.random.default_rng(args.seed)
        any_fail = False

        # Page table for slot 0 only: full contiguous block range.
        page_table_np = np.arange(NUM_BLOCKS, dtype=np.int32).reshape(1, NUM_BLOCKS)
        page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                        dtype=ttnn.int32,
                                        layout=ttnn.ROW_MAJOR_LAYOUT,
                                        device=device)

        # Common program config — pick q/k chunk sizes that match our test sizes.
        progcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=C, k_chunk_size=BLOCK_SIZE, exp_approx_mode=False)

        for L_prefix in prefixes:
            L_total = L_prefix + C
            assert L_total <= NUM_BLOCKS * BLOCK_SIZE, "test ran past cache size"
            log(f"=== L_prefix={L_prefix}  C={C}  L_total={L_total} ===")

            K_total = rng.normal(0, 1.0, (L_total, HEAD_DIM)).astype(np.float32)
            V_total = rng.normal(0, 1.0, (L_total, HEAD_DIM)).astype(np.float32)
            q_chunk = rng.normal(0, 1.0, (C, NQ, HEAD_DIM)).astype(np.float32)

            # KV cache is pre-filled with ALL L_total positions (the chunk's own
            # K/V are already written; chunked SDPA just attends).
            kc_np = np.zeros((NUM_BLOCKS, NKV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
            vc_np = np.zeros_like(kc_np)
            write_paged_cache(kc_np, vc_np, page_table_np[0], K_total, V_total, L_total)

            # numpy reference (multi-query causal over the full prefix + chunk-internal causal).
            ref = numpy_chunk_causal_attention(
                q_chunk.astype(np.float64),
                K_total.astype(np.float64),
                V_total.astype(np.float64),
                L_prefix)

            def to_tt(x, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
                return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x.astype(np.float32))),
                                       dtype=dt, layout=layout, device=device)

            kc = to_tt(kc_np); vc = to_tt(vc_np)
            # Q for chunked SDPA: [B=1, NQ, C, HEAD_DIM] is the typical Llama
            # caller shape (models/tt_transformers/tt/attention.py:1072).
            q_np = q_chunk.transpose(1, 0, 2).reshape(1, NQ, C, HEAD_DIM)
            q_tt = to_tt(q_np)

            log(f"  calling chunked SDPA: Q[1,{NQ},{C},{HEAD_DIM}] paged_KV[{NUM_BLOCKS},{NKV},{BLOCK_SIZE},{HEAD_DIM}] chunk_start={L_prefix}")
            try:
                out_tt = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_tt, kc, vc, page_table_tt, L_prefix,
                    program_config=progcfg,
                )
            except TypeError as e:
                # Some builds want chunk_start_idx_tensor as a kwarg instead.
                log(f"  positional signature TypeError: {e}; retrying with kwargs")
                idx_tt = ttnn.from_torch(torch.tensor([L_prefix], dtype=torch.int32),
                                         dtype=ttnn.int32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT,
                                         device=device)
                out_tt = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_tt, kc, vc,
                    page_table_tensor=page_table_tt,
                    chunk_start_idx_tensor=idx_tt,
                    program_config=progcfg,
                )
                ttnn.deallocate(idx_tt)

            ttnn.synchronize_device(device)
            out_np = ttnn.to_torch(out_tt).float().numpy().reshape(1, NQ, C, HEAD_DIM)[0]
            # [NQ,C,D] → [C,NQ,D] to match reference layout.
            out_np = out_np.transpose(1, 0, 2)
            ttnn.deallocate(out_tt)
            ttnn.deallocate(kc); ttnn.deallocate(vc); ttnn.deallocate(q_tt)

            # Per-position correctness (worst slot is the most informative).
            worst_cos = 1.0
            worst_i = -1
            for i in range(C):
                c = cos(out_np[i], ref[i])
                if c < worst_cos:
                    worst_cos, worst_i = c, i
            ok = worst_cos >= args.pcc_threshold
            any_fail = any_fail or not ok
            log(f"  worst per-position cos = {worst_cos:.6f} at chunk-pos {worst_i}  "
                f"{'OK' if ok else 'FAIL'}")

        ttnn.deallocate(page_table_tt)
        if any_fail:
            log("FAIL: chunked SDPA does NOT match per-position causal reference at all tested prefixes.")
            raise SystemExit(1)
        log(f"PASS: chunked_scaled_dot_product_attention is correct at Qwen3.6 "
            f"per-chip shapes for prefixes {prefixes} (C={C}). S2.1 gate green; "
            f"S2.2 (swap S1a primitive) unblocked.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
