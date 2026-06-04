#!/usr/bin/env python3
"""Gemma 4 global-attention WRITE+READ isolation probe.

Sliding shape primitive PASSES (gm4_sliding_write_read.py). Global has a
different shape (NKV=1, head_dim=512, p-RoPE) and uses the canonical
Tenstorrent config (CoreCoord(8,4), q_chunk=32, k_chunk=64). Test the
write+read primitive at multiple positions for the global shape.

Run on qb1:
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb/isolate/gm4_global_write_read.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

NKV = 1
NQ = 4              # NQ_PER_CHIP for global
HEAD_DIM = 512      # Gemma 4 global head_dim
BLOCK_SIZE = 32
NUM_BLOCKS = 8
N_STEPS = 6


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1); b = np.asarray(b, np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def numpy_attn(q, K, V, pos):
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

        cs = (NUM_BLOCKS, NKV, BLOCK_SIZE, HEAD_DIM)
        kc = ttnn.from_torch(torch.from_numpy(np.zeros(cs, dtype=np.float32)),
                             dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        vc = ttnn.from_torch(torch.from_numpy(np.zeros(cs, dtype=np.float32)),
                             dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        pt_np = np.arange(NUM_BLOCKS, dtype=np.int32).reshape(1, NUM_BLOCKS)
        page_table = ttnn.from_torch(torch.from_numpy(pt_np), dtype=ttnn.int32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        grid = device.compute_with_storage_grid_size()
        write_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(1, grid, row_wise=True),
                [BLOCK_SIZE, HEAD_DIM],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        # Canonical Gemma 4 global SDPA config from in-tree demo.
        sdpa_progcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            q_chunk_size=32, k_chunk_size=64, exp_approx_mode=False,
        )
        sdpa_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        K_steps = rng.normal(0, 1.0, (N_STEPS, HEAD_DIM)).astype(np.float32)
        V_steps = rng.normal(0, 1.0, (N_STEPS, HEAD_DIM)).astype(np.float32)
        Q_steps = rng.normal(0, 1.0, (N_STEPS, NQ, HEAD_DIM)).astype(np.float32)

        ref_outs = [numpy_attn(Q_steps[pos], K_steps[:pos+1], V_steps[:pos+1], pos)
                    for pos in range(N_STEPS)]

        any_fail = False
        log(f"running {N_STEPS}-step global write+SDPA loop…")
        for pos in range(N_STEPS):
            cur_pos_buf = ttnn.from_torch(
                torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
            )
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
                # No sliding_window_size for global.
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
            log("VERDICT: FAIL — global primitive broken at some pos.")
            raise SystemExit(1)
        log("VERDICT: PASS — global primitive correct; bug is upstream.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
