#!/usr/bin/env python3
"""Gemma 4 v0.3.1 sliding-attention WRITE+READ isolation probe.

Question: at pos > 0, does the paged write at pos N + read at pos N
correctly reflect K/V values from positions [0..N]? The full-forward
probe (gm4_v031_multistep_cos.py) shows 3/6 PASS with pos 1 echoing
the input — symptom of attention contributing ~0 at some positions.

This probe mirrors the EXACT sliding shape:
  NKV=1 per SDPA call (the v0.3.0.1 two-call layout)
  HEAD_DIM=256 (sliding)
  NQ_PER_CALL=2 (Q_HALF for GQA group=2)
  sliding_window_size=1024

What we do per pos N in 0..5:
  1. Reallocate cur_pos_buf = [N]
  2. paged_update_cache(K_N, V_N) at slot N
  3. paged_scaled_dot_product_attention_decode at pos N
  4. Compare to numpy attention over [K_0..K_N], [V_0..V_N]

If any pos > 0 fails: the write/read primitive is the bug.
If all pass: the bug is upstream (RoPE, projection, GQA mapping).

~5s/cycle vs full-forward 80s — matches [[use-existing-isolation-probes]].

Run on qb1:
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb/isolate/gm4_sliding_write_read.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

NKV = 1            # per SDPA call (v0.3.0.1: two calls per sliding layer)
NQ = 2             # Q_HALF — Q heads attending to this KV head (GQA group=2)
HEAD_DIM = 256     # Gemma 4 sliding head_dim
BLOCK_SIZE = 32
NUM_BLOCKS = 8     # 256-token capacity (plenty for 6-step probe)
SLIDING_WINDOW = 1024
N_STEPS = 6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1); b = np.asarray(b, np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def numpy_attn(q, K, V, pos):
    """q [NQ, D]; K, V [pos+1, D] (NKV=1, broadcast). Returns [NQ, D]."""
    scale = 1.0 / np.sqrt(HEAD_DIM)
    out = np.zeros_like(q)
    k = K[:pos + 1]
    v = V[:pos + 1]
    for h in range(NQ):
        s = (q[h] @ k.T) * scale
        s -= s.max()
        w = np.exp(s); w /= w.sum()
        out[h] = w @ v
    return out


def main():
    import ttnn
    log("opening device 0")
    device = ttnn.open_device(device_id=0)
    try:
        rng = np.random.default_rng(0)

        # Cache: [NUM_BLOCKS, NKV=1, BLOCK_SIZE, HEAD_DIM], bf16, TILE.
        cs = (NUM_BLOCKS, NKV, BLOCK_SIZE, HEAD_DIM)
        kc_np = np.zeros(cs, dtype=np.float32)
        vc_np = np.zeros(cs, dtype=np.float32)
        kc = ttnn.from_torch(torch.from_numpy(kc_np), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=device)
        vc = ttnn.from_torch(torch.from_numpy(vc_np), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=device)

        # Page table [1, NUM_BLOCKS] identity (batch=1 decode).
        pt_np = np.arange(NUM_BLOCKS, dtype=np.int32).reshape(1, NUM_BLOCKS)
        page_table = ttnn.from_torch(torch.from_numpy(pt_np), dtype=ttnn.int32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        # Sharded write memory config (1 core, NKV=1, BLOCK_SIZE × HEAD_DIM).
        grid = device.compute_with_storage_grid_size()
        write_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(1, grid, row_wise=True),
                [BLOCK_SIZE, HEAD_DIM],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # SDPA program config (Gemma 4 sliding, head_dim=256).
        sdpa_progcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            q_chunk_size=32, k_chunk_size=128, exp_approx_mode=False,
        )
        sdpa_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Per-step K/V vectors and Q.
        K_steps = rng.normal(0, 1.0, (N_STEPS, HEAD_DIM)).astype(np.float32)
        V_steps = rng.normal(0, 1.0, (N_STEPS, HEAD_DIM)).astype(np.float32)
        Q_steps = rng.normal(0, 1.0, (N_STEPS, NQ, HEAD_DIM)).astype(np.float32)

        # Numpy ref: build history; at each pos compute attention.
        ref_outs = []
        for pos in range(N_STEPS):
            K_hist = K_steps[:pos + 1]  # [pos+1, D]
            V_hist = V_steps[:pos + 1]
            ref_outs.append(numpy_attn(Q_steps[pos], K_hist, V_hist, pos))

        any_fail = False
        log(f"running {N_STEPS}-step write+SDPA loop…")
        for pos in range(N_STEPS):
            # Build cur_pos_buf = [pos] (int32).
            cur_pos_buf = ttnn.from_torch(
                torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
            )

            # paged_update_cache input: [1, NKV=1, BLOCK_SIZE, HEAD_DIM]
            # but real row = K_steps[pos] at slot 0.
            k_in_np = np.zeros((1, NKV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
            v_in_np = np.zeros((1, NKV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
            k_in_np[0, 0, 0] = K_steps[pos]
            v_in_np[0, 0, 0] = V_steps[pos]
            k_in = ttnn.from_torch(torch.from_numpy(k_in_np), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=device,
                                   memory_config=write_mem_cfg)
            v_in = ttnn.from_torch(torch.from_numpy(v_in_np), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=device,
                                   memory_config=write_mem_cfg)
            ttnn.experimental.paged_update_cache(
                kc, k_in, update_idxs_tensor=cur_pos_buf, page_table=page_table,
            )
            ttnn.experimental.paged_update_cache(
                vc, v_in, update_idxs_tensor=cur_pos_buf, page_table=page_table,
            )
            ttnn.deallocate(k_in); ttnn.deallocate(v_in)

            # SDPA decode at this pos. Q shape [1, 1, NQ, HEAD_DIM].
            q_np = Q_steps[pos].reshape(1, 1, NQ, HEAD_DIM).astype(np.float32)
            q_tt = ttnn.from_torch(torch.from_numpy(q_np), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=device)
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_tt, kc, vc,
                cur_pos_tensor=cur_pos_buf,
                page_table_tensor=page_table,
                scale=1.0 / (HEAD_DIM ** 0.5),
                program_config=sdpa_progcfg,
                compute_kernel_config=sdpa_compute,
                sliding_window_size=SLIDING_WINDOW,
            )
            ttnn.synchronize_device(device)
            out = ttnn.to_torch(attn).float().numpy().reshape(NQ, HEAD_DIM)
            ttnn.deallocate(attn); ttnn.deallocate(q_tt); ttnn.deallocate(cur_pos_buf)

            c = cos(out, ref_outs[pos])
            ok = c >= 0.99
            any_fail = any_fail or not ok
            mad = float(np.max(np.abs(out - ref_outs[pos])))
            log(f"  pos={pos} cos={c:.6f} max_abs_diff={mad:.4e} {'PASS' if ok else 'FAIL'}")

        for t in (kc, vc, page_table):
            try: ttnn.deallocate(t)
            except Exception: pass

        if any_fail:
            log("VERDICT: FAIL — primitive write+read is broken at some pos. "
                "Bug is in paged_update_cache or paged SDPA decode for this shape.")
            raise SystemExit(1)
        log("VERDICT: PASS — primitive is correct. Bug is upstream "
            "(RoPE, projections, GQA mapping in _layer_pos0_sliding_paged).")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
