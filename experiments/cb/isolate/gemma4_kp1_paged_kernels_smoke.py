#!/usr/bin/env python3
"""Phase 2.B kernel isolation gate — does paged_update_cache +
paged_scaled_dot_product_attention_decode accept B=K+1 inputs with
the alias-page-table pattern WITHOUT TT_FATAL?

This is the HARD-STOP gate per Phase 2.B agent's research:
  paged_update_cache_device_operation.cpp:49 asserts padded_shape[0]==1
  (batch on dim 1, not 0); cur_pos_tensor shape must match Q.
The existing paged_update_cache.py probe (B=4 with DISTINCT page-table
rows) confirms kernel works at B>1; this probe specifically tests the
ALIAS pattern (all K+1 rows → row 0's blocks) which is what verify needs.

If this PASSES, the 2.B.1 server refactor is unblocked.
If this FATALs, we escalate to research/gemma4_verify_kp1_blocker.md
and rethink Phase 2.B before any server work.

Forks experiments/cb/isolate/paged_update_cache.py (B>1 paged write,
single device) with:
  - cur_pos all-at-L (verify position)
  - page_table aliased per spec_dec_scheduler.build_verify_alias_page_table_host
  - adds paged_sdpa_decode call at B=K+1 (the existing probe only writes,
    doesn't read)

Run on qb1:
  ssh qb1 'cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb/isolate/gemma4_kp1_paged_kernels_smoke.py'
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Gemma 4 12B sliding-attention shapes (per server_gemma4_unified_ttnn.py:68-85)
NKV = 1                  # NKV_PER_CHIP for our probe (single-cache-per-call simplification;
                         # the production server has NKV_PER_CHIP_SLIDING=2 via two_call pattern)
HEAD_DIM = 256
BLOCK_SIZE = 32
TILE_HEIGHT = 32
NUM_BLOCKS_TOTAL = 64
NUM_Q_HEADS = 16         # Q heads per chip in Gemma 4 12B sliding
K_LOOKAHEAD = 5
B = K_LOOKAHEAD + 1      # verify batch = 6
VERIFY_POS = 12          # arbitrary position to read/write at


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None) -> int:  # state= accepted for dev-harness compat
    import ttnn
    log(f"opening device 0")
    device = ttnn.open_device(device_id=0)
    rc = 0
    try:
        # Build aliased page-table:
        #   row 0 = active prompt's KV blocks (real, contiguous)
        #   rows [1..K+1) = aliased to row 0 (all point at same physical blocks)
        #   rows [K+1..) = spare (unused; could be any)
        from spec_dec_scheduler import build_verify_alias_page_table_host
        rows = max(B + 1, 8)  # at least B+1 rows so verify_offset=1 + K+1 fits
        blocks_per_seq = NUM_BLOCKS_TOTAL // rows
        # base page-table: each row gets distinct contiguous blocks
        base_pt = torch.stack([
            torch.arange(r * blocks_per_seq, (r + 1) * blocks_per_seq, dtype=torch.int32)
            for r in range(rows)
        ], dim=0)
        log(f"base page-table shape={tuple(base_pt.shape)} (rows={rows}, "
            f"blocks_per_seq={blocks_per_seq})")

        alias_pt = build_verify_alias_page_table_host(base_pt, K=K_LOOKAHEAD,
                                                     verify_offset=1)
        # Truncate to B rows (we only need B=K+1 verify rows; rows 0..K)
        # Actually for the kernel call, we pass exactly B rows of page-table
        # matching the B input rows. Row 0 of THE CALL'S page-table = first
        # logical batch index — but we want all B rows of the call to alias
        # row 0 of the base. So pass alias_pt[0:B] which has:
        #   row 0 = base row 0 (the original "active" prompt)
        #   rows 1..K+1 = aliased copies of base row 0 (set by helper)
        call_pt = alias_pt[:B].contiguous()
        log(f"call page-table shape={tuple(call_pt.shape)}; row 0 == row 1: "
            f"{torch.equal(call_pt[0], call_pt[1])}")

        # Sanity: all B rows of call_pt should equal call_pt[0]
        for r in range(1, B):
            assert torch.equal(call_pt[0], call_pt[r]), \
                f"alias row {r} != row 0"
        log(f"  all {B} alias rows == row 0 ✓")

        # K=K+1 candidate K vectors to write (each candidate's K projection
        # for token at VERIFY_POS). Distinct per candidate so we can verify
        # which one "won" the write race.
        rng = np.random.default_rng(seed=0)
        K_candidates_np = rng.normal(0, 1.0, (B, HEAD_DIM)).astype(np.float32)
        # Marker so we can identify each: K_candidates_np[b, 0] = b (sentinel)
        K_candidates_np[:, 0] = np.arange(B, dtype=np.float32)

        # Cache zeroed
        cache_np = np.zeros((NUM_BLOCKS_TOTAL, NKV, BLOCK_SIZE, HEAD_DIM),
                            dtype=np.float32)
        cache = ttnn.from_torch(torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)

        # Input [1, B, TILE_HEIGHT, head_dim], real row 0 per slot = K_candidates,
        # rest TILE padding. Height-sharded on B cores.
        inp_np = np.zeros((1, B, TILE_HEIGHT, HEAD_DIM), dtype=np.float32)
        inp_np[0, :, 0, :] = K_candidates_np

        compute_grid = device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(B, compute_grid, row_wise=True)
        shard_spec = ttnn.ShardSpec(shard_grid, [TILE_HEIGHT, HEAD_DIM],
                                    ttnn.ShardOrientation.ROW_MAJOR)
        write_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        inp = ttnn.from_torch(torch.from_numpy(inp_np), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=device,
                              memory_config=write_mem_cfg)

        # cur_pos shape [B]: all candidates write at VERIFY_POS
        cur_pos_np = np.full((B,), VERIFY_POS, dtype=np.int32)
        update_idxs = ttnn.from_torch(torch.from_numpy(cur_pos_np),
                                      dtype=ttnn.int32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT,
                                      device=device)
        page_table_tt = ttnn.from_torch(torch.from_numpy(call_pt.numpy()),
                                        dtype=ttnn.int32,
                                        layout=ttnn.ROW_MAJOR_LAYOUT,
                                        device=device)

        # ── GATE 1: paged_update_cache accepts B=K+1 + aliased page-table ──
        log("")
        log("=" * 64)
        log(f"GATE 1: paged_update_cache(B={B}, aliased)")
        log("=" * 64)
        try:
            ttnn.experimental.paged_update_cache(
                cache, inp, update_idxs_tensor=update_idxs,
                page_table=page_table_tt,
            )
            ttnn.synchronize_device(device)
            log("  ✓ paged_update_cache returned cleanly (no TT_FATAL)")
        except Exception as e:
            log(f"  ✗ paged_update_cache FATAL: {type(e).__name__}: {e}")
            traceback.print_exc()
            rc = 1
            return rc

        # ── Inspect what landed at row 0's physical slot ──
        cache_out = ttnn.to_torch(cache).float().numpy().reshape(
            NUM_BLOCKS_TOTAL, NKV, BLOCK_SIZE, HEAD_DIM)
        phys_block = int(base_pt[0, VERIFY_POS // BLOCK_SIZE])
        slot_in_block = VERIFY_POS % BLOCK_SIZE
        written = cache_out[phys_block, 0, slot_in_block]
        marker = float(written[0])  # bf16-rounded sentinel
        log(f"  slot (block={phys_block}, slot={slot_in_block}, head=0)[0] = "
            f"{marker:.2f}")
        log(f"  (one of {list(range(B))} candidates won the write race; "
            f"bf16-rounded so may not be exact)")

        # Check: written K should be close to ONE of the K_candidates (whichever won)
        cos_per_cand = []
        for b in range(B):
            c = float(np.dot(written, K_candidates_np[b]) /
                      (np.linalg.norm(written) *
                       np.linalg.norm(K_candidates_np[b]) + 1e-9))
            cos_per_cand.append(c)
        winner = int(np.argmax(cos_per_cand))
        log(f"  argmax cos vs candidates = {winner} (cos={cos_per_cand[winner]:.4f})")
        log(f"  per-candidate cos: " +
            " ".join(f"{i}:{c:.3f}" for i, c in enumerate(cos_per_cand)))
        if cos_per_cand[winner] < 0.9:
            log(f"  ✗ no candidate matches the written K — kernel may have written garbage")
            rc = 1

        # ── GATE 2: paged_sdpa_decode accepts B=K+1 + aliased page-table ──
        log("")
        log("=" * 64)
        log(f"GATE 2: paged_scaled_dot_product_attention_decode(B={B}, aliased)")
        log("=" * 64)
        # SDPA decode: Q shape [1, B, NUM_Q_HEADS, HEAD_DIM]
        # Build a small Q with B=K+1 rows. Each row gets distinct Q so we
        # see distinct outputs.
        Q_np = rng.normal(0, 1.0, (1, B, NUM_Q_HEADS, HEAD_DIM)).astype(np.float32)
        Q = ttnn.from_torch(torch.from_numpy(Q_np), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
        # We need K and V caches; reuse the same cache buffer for both
        # (since the contents don't matter for shape-contract validation;
        # we just need the read to not FATAL).
        # Build a V cache with similar shape.
        v_cache_np = rng.normal(0, 1.0,
                                (NUM_BLOCKS_TOTAL, NKV, BLOCK_SIZE, HEAD_DIM)
                                ).astype(np.float32)
        # Stamp the verify slot with marker too
        for b in range(B):
            v_cache_np[phys_block, 0, slot_in_block, 0] = 100.0 + b
        v_cache = ttnn.from_torch(torch.from_numpy(v_cache_np),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=device)

        # cur_pos for SDPA: all B at VERIFY_POS (reads history up through pos)
        cur_pos_sdpa = ttnn.from_torch(torch.from_numpy(
            np.full((B,), VERIFY_POS + 1, dtype=np.int32)),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device)

        try:
            sdpa_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                Q, cache, v_cache,
                cur_pos_tensor=cur_pos_sdpa,
                page_table_tensor=page_table_tt,
                scale=1.0 / (HEAD_DIM ** 0.5),
            )
            ttnn.synchronize_device(device)
            log(f"  ✓ paged_sdpa_decode returned cleanly; output shape "
                f"{list(sdpa_out.shape)}")
            sdpa_out_np = ttnn.to_torch(sdpa_out).float().numpy()
            log(f"  numpy shape={sdpa_out_np.shape}, "
                f"first-row mean={sdpa_out_np.flatten()[0]:.4f}")
            ttnn.deallocate(sdpa_out)
        except Exception as e:
            log(f"  ✗ paged_sdpa_decode FATAL: {type(e).__name__}: {e}")
            traceback.print_exc()
            rc = 1

        # cleanup
        for t in (cache, inp, update_idxs, page_table_tt, Q, v_cache,
                  cur_pos_sdpa):
            try: ttnn.deallocate(t)
            except Exception: pass

    finally:
        ttnn.close_device(device)

    log("")
    log("=" * 64)
    if rc == 0:
        log("PHASE 2.B KERNEL ISOLATION GATE: PASS")
        log("Both paged_update_cache + paged_sdpa_decode accept B=K+1 "
            "with the alias-page-table pattern. Server refactor unblocked.")
    else:
        log("PHASE 2.B KERNEL ISOLATION GATE: FAIL")
        log("See research/gemma4_verify_kp1_blocker.md for architectural escalation.")
    log("=" * 64)
    return rc


if __name__ == "__main__":
    sys.exit(main())
