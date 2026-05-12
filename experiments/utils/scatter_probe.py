#!/usr/bin/env python3
"""
Probe ttnn.scatter for KV cache slot writes at our specific layout.

We want to write k_new (shape [N_KV, HEAD_DIM]) into kv_cache
(shape [1, N_KV, MAX_POS, HEAD_DIM]) at position cur_pos.

This script verifies whether ttnn.scatter handles that pattern and what
exactly it produces. If scatter works, C'1 implementation is ~10 LOC.
If it doesn't, we fall back to on-device mask+mul+add.

The semantic test: scatter K at cur_pos into a zero cache, then read it
back. The cache slot at cur_pos should equal K; all other slots zero.

Run on qb2:
    cd ~/tt-xla && .venv/bin/python experiments/utils/scatter_probe.py
"""
import sys
import numpy as np
import torch
import ttnn

N_KV = 4
HEAD_DIM = 256
MAX_POS = 256
CUR_POS = 7   # arbitrary position to write into


def cosine(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 64)
    print("Probe: ttnn.scatter for KV cache slot write")
    print("=" * 64)

    device = ttnn.open_device(device_id=0)

    # Build a zero cache on device
    cache_np = np.zeros((1, N_KV, MAX_POS, HEAD_DIM), dtype=np.float32)
    cache_tt = ttnn.from_torch(torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
    print(f"cache: shape={tuple(cache_tt.shape)}  dtype={cache_tt.dtype}")

    # Build k_new with distinctive per-head, per-dim values
    # value at (h, d) = (h+1) * 1000 + d  — lets us verify exact placement
    k_np = np.zeros((1, N_KV, 1, HEAD_DIM), dtype=np.float32)
    for h in range(N_KV):
        for d in range(HEAD_DIM):
            k_np[0, h, 0, d] = (h + 1) * 1000 + d
    k_tt = ttnn.from_torch(torch.from_numpy(k_np), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)
    print(f"k_new: shape={tuple(k_tt.shape)}  values=[h={'each head' } d, (h+1)*1000+d]")
    print(f"  sample k_new[0, 0, 0, 0:5] = {k_np[0, 0, 0, 0:5]}")
    print(f"  sample k_new[0, 3, 0, 0:5] = {k_np[0, 3, 0, 0:5]}")

    # Build index tensor — same shape as k_tt, with value = CUR_POS at every position
    index_np = np.full((1, N_KV, 1, HEAD_DIM), CUR_POS, dtype=np.int32)
    index_tt = ttnn.from_torch(torch.from_numpy(index_np), dtype=ttnn.int32,
                                device=device, layout=ttnn.TILE_LAYOUT)
    print(f"index: shape={tuple(index_tt.shape)}  value=CUR_POS={CUR_POS} everywhere")

    # Try ttnn.scatter
    print(f"\nCalling ttnn.scatter(cache, dim=2, index=index, src=k_new)…")
    try:
        result_tt = ttnn.scatter(cache_tt, dim=2, index=index_tt, src=k_tt)
        print(f"  result shape: {tuple(result_tt.shape)}  dtype={result_tt.dtype}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        ttnn.close_device(device)
        return

    # Read back and verify
    result_np = ttnn.to_torch(result_tt).float().cpu().numpy()
    print(f"  read back: shape={result_np.shape}")

    # Check: result[0, :, CUR_POS, :] should equal k_new (within bf16 precision)
    written = result_np[0, :, CUR_POS, :]
    expected = k_np[0, :, 0, :]
    cos = cosine(written, expected)
    max_diff = float(np.abs(written - expected).max())
    print(f"\n  written slot vs expected k_new:")
    print(f"    cosine = {cos:.6f}")
    print(f"    max|Δ| = {max_diff:.4f} (bf16 precision ~1)")
    print(f"    sample written[0, 0:5] = {written[0, 0:5]}")
    print(f"    sample written[3, 0:5] = {written[3, 0:5]}")

    # Check: result[0, :, p, :] should be ~zero for p != CUR_POS
    other_positions = np.delete(np.arange(MAX_POS), CUR_POS)
    others = result_np[0, :, other_positions, :]
    others_max = float(np.abs(others).max())
    print(f"\n  other positions (should be ~zero):")
    print(f"    max|·| across all = {others_max:.4f}")

    # Verdict
    print()
    if cos > 0.99 and others_max < 1.0:
        print("✓ ttnn.scatter WORKS as expected — write at cur_pos lands correctly,")
        print("  other positions remain zero. We can use this for C'1.")
    elif cos > 0.99 and others_max >= 1.0:
        print("⚠ scatter wrote the slot correctly but also touched other positions.")
        print("  Indexes might not behave as expected. Investigate.")
    else:
        print("✗ scatter didn't write what we expected. Need alternative.")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
