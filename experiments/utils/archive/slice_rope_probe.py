#!/usr/bin/env python3
"""
Probe ttnn.slice for non-tile-aligned row slicing of a [MAX_POS, rotary_dim]
table — the critical risk for C'0.6 (RoPE precompute).

C'0.6 swaps per-token cos/sin upload for a precomputed table + per-step
on-device slice. The open question: does `ttnn.slice(table, [pos, 0],
[pos+1, rotary_dim])` actually work when `pos` is not a multiple of 32?

A tile is 32×32. The TILE_LAYOUT storage groups every 32 rows into one tile.
A 1-row slice from inside a tile means partial-tile read — historically a
weak path in ttnn.

This probe tests every alignment case so we know BEFORE the 30-min C'0.6
gate cycle whether the approach is sound.

Run on qb1 (qb2 is busy):
    cd ~/tt-xla && .venv/bin/python experiments/utils/slice_rope_probe.py
"""
import sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    MAX_POS = 128       # 4 tiles of 32 rows each — covers all alignment cases
    ROTARY_DIM = 64     # Qwen3.6: 0.25 * 256 = 64

    print("=" * 64)
    print(f"Probe: ttnn.slice row-extraction from [{MAX_POS}, {ROTARY_DIM}] table")
    print("=" * 64)

    # Build a table with KNOWN values per row — value at (row, col) = row * 1000 + col
    table_np = np.zeros((MAX_POS, ROTARY_DIM), dtype=np.float32)
    for r in range(MAX_POS):
        for c in range(ROTARY_DIM):
            table_np[r, c] = r * 1000.0 + c

    device = ttnn.open_device(device_id=0)
    try:
        # Upload as fp32 in TILE_LAYOUT — matches C'0.6 production path
        table_tt = ttnn.from_torch(
            torch.from_numpy(table_np), dtype=ttnn.float32,
            device=device, layout=ttnn.TILE_LAYOUT)
        print(f"\ntable shape: {tuple(table_tt.shape)}  dtype={table_tt.dtype}  layout=TILE")

        # Test positions: covers tile-aligned, off-by-1, mid-tile, edge-of-tile
        test_positions = [0, 1, 5, 17, 31, 32, 33, 47, 63, 64, 65, 95, 96, 97, 127]

        print(f"\nTesting {len(test_positions)} positions across all tile boundaries:")
        print(f"{'pos':>5} {'tile-aligned?':>14} {'result':>10}  {'sample slice[0:5]':>40}")
        print("-" * 75)

        all_pass = True
        for pos in test_positions:
            tile_aligned = (pos % 32 == 0)
            try:
                sliced = ttnn.slice(table_tt, [pos, 0], [pos + 1, ROTARY_DIM])
                back = ttnn.to_torch(sliced).float().cpu().numpy().flatten()
                expected = table_np[pos, :]
                cos = _cosine(back, expected)
                max_diff = float(np.abs(back - expected).max())
                result = "✓ OK" if (cos > 0.9999 and max_diff < 1e-3) else "⚠ DRIFT"
                if cos < 0.9999 or max_diff > 1e-3:
                    all_pass = False
                print(f"{pos:>5} {'YES' if tile_aligned else 'no':>14} "
                      f"{result:>10}  back={back[:5]}")
                if cos < 0.9999:
                    print(f"      cos={cos:.6f} max|Δ|={max_diff:.4e}  expected={expected[:5]}")
            except Exception as e:
                msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                print(f"{pos:>5} {'YES' if tile_aligned else 'no':>14}  ✗ EXCEPTION: {msg[:80]}")
                all_pass = False

        print()
        if all_pass:
            print("✓ ALL positions return the correct row. ttnn.slice handles the")
            print("  non-tile-aligned case correctly. C'0.6 RoPE precompute is safe.")
        else:
            print("✗ Some positions fail. C'0.6 needs the ttnn.embedding fallback.")

        # Bonus: same test in ROW_MAJOR layout for comparison
        print("\n" + "=" * 64)
        print("Bonus: same test in ROW_MAJOR layout (no tile machinery)")
        print("=" * 64)
        table_rm = ttnn.from_torch(
            torch.from_numpy(table_np), dtype=ttnn.float32,
            device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        print(f"table shape: {tuple(table_rm.shape)}  layout=ROW_MAJOR")
        for pos in [0, 17, 33, 65, 97]:
            try:
                sliced = ttnn.slice(table_rm, [pos, 0], [pos + 1, ROTARY_DIM])
                back = ttnn.to_torch(sliced).float().cpu().numpy().flatten()
                cos = _cosine(back, table_np[pos, :])
                max_diff = float(np.abs(back - table_np[pos, :]).max())
                print(f"  pos={pos:>3}: cos={cos:.6f} max|Δ|={max_diff:.4e}")
            except Exception as e:
                print(f"  pos={pos:>3}: EXCEPTION: {str(e).splitlines()[0][:80]}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
