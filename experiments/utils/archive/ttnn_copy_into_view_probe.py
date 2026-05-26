#!/usr/bin/env python3
"""
Probe: does ttnn.copy(src, ttnn.slice(cache, ...)) propagate the write
back into the original cache tensor?

This is the make-or-break test for unblocking:
  - C'0.5 long-context KV writes (paged_update_cache hangs #16674 means
    we need an alternative in-place writer)
  - C'4 multi-step trace (C'1's functional scatter doesn't thread the
    cache through trace replay; need in-place mutation)

If ttnn.slice returns a TRUE view (metadata-only, points into the cache's
storage), then writing through the view should propagate to the cache.
If ttnn.slice materializes a new tensor, the write goes to that new tensor
and the original cache is unchanged.

Three patterns tested:
  A: ttnn.copy(src, ttnn.slice(cache, ...))     # anon view
  B: view = ttnn.slice(...); ttnn.copy(src, view)  # named view
  C: cache = ttnn.copy(src, ttnn.slice(...))    # functional, returns new cache

Run on qb1 (qb2 holds the warm server):
    cd ~/tt-xla && .venv/bin/python experiments/utils/ttnn_copy_into_view_probe.py
"""
import sys
import inspect
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

# Small shape: 4 blocks, each [N_KV=4, P=16, HD=32]
MAX_BLOCKS = 4
N_KV = 4
P = 16
HD = 32

# Where we'll try to write
TARGET_BLOCK = 1
TARGET_SLOT = 5     # not tile-aligned within the block (slot dim is P=16)


def _show_signature():
    fn = getattr(ttnn, "copy", None)
    if fn is None:
        print("  ttnn.copy NOT FOUND")
        return
    print("\nttnn.copy signature:")
    try:
        print(f"  {inspect.signature(fn)}")
    except Exception as e:
        print(f"  (signature unavailable: {e})")
    doc = inspect.getdoc(fn)
    if doc:
        print("  doc (first 15 lines):")
        for line in doc.split("\n")[:15]:
            print(f"    | {line}")


def fresh_cache(device, fill_value=0.0):
    np_arr = np.full((MAX_BLOCKS, N_KV, P, HD), fill_value, dtype=np.float32)
    return ttnn.from_torch(torch.from_numpy(np_arr), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)


def build_new_row(device, seed=42):
    """[1, N_KV, 1, HD] with distinctive values."""
    rng = np.random.default_rng(seed)
    new = rng.standard_normal((1, N_KV, 1, HD)).astype(np.float32) * 0.1
    return new, ttnn.from_torch(torch.from_numpy(new), dtype=ttnn.bfloat16,
                                  device=device, layout=ttnn.TILE_LAYOUT)


def verify(cache_tt, new_np, label):
    """Read back cache, check whether the write landed at (TARGET_BLOCK, TARGET_SLOT)."""
    back = ttnn.to_torch(cache_tt).float().cpu().numpy()
    written = back[TARGET_BLOCK, :, TARGET_SLOT, :]    # [N_KV, HD]
    expected = new_np[0, :, 0, :]
    cos = float(np.dot(written.flatten().astype(np.float64),
                       expected.flatten().astype(np.float64)) /
                (np.linalg.norm(written) * np.linalg.norm(expected) + 1e-12))
    max_diff = float(np.abs(written - expected).max())

    # Sample other positions to ensure they're untouched (still zero)
    other_max = float(np.abs(np.concatenate([
        back[TARGET_BLOCK, :, :TARGET_SLOT, :].flatten(),
        back[TARGET_BLOCK, :, TARGET_SLOT+1:, :].flatten(),
        back[:TARGET_BLOCK].flatten(),
        back[TARGET_BLOCK+1:].flatten(),
    ])).max())

    print(f"  [{label}] target-slot cos={cos:.6f}, max|Δ|={max_diff:.4e}, "
          f"others_max={other_max:.4e}")
    if cos > 0.99 and other_max < 0.01:
        return "PASS"
    elif cos > 0.99 and other_max >= 0.01:
        return "WRITE-OK-BUT-LEAKED"
    elif other_max < 0.01:
        return "NO-WRITE (slice is materialized, write went nowhere)"
    else:
        return "GARBAGE"


def main():
    print("=" * 64)
    print(f"Probe: ttnn.copy(src, ttnn.slice(cache, ...)) — does it write through?")
    print(f"  cache shape: [{MAX_BLOCKS}, {N_KV}, {P}, {HD}] bf16 TILE")
    print(f"  target: cache[{TARGET_BLOCK}, :, {TARGET_SLOT}, :] = new_row")
    print("=" * 64)

    _show_signature()

    device = ttnn.open_device(device_id=0)
    try:
        new_np, new_tt = build_new_row(device)

        # ---------------- Pattern A: anon view ----------------
        print("\n[A] ttnn.copy(new_tt, ttnn.slice(cache, ...))")
        cache_tt = fresh_cache(device)
        try:
            ret = ttnn.copy(new_tt, ttnn.slice(cache_tt,
                [TARGET_BLOCK, 0, TARGET_SLOT, 0],
                [TARGET_BLOCK + 1, N_KV, TARGET_SLOT + 1, HD]))
            verdict = verify(cache_tt, new_np, "A")
            print(f"  → {verdict}")
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  FAILED: {msg[:200]}")

        # ---------------- Pattern B: named view ----------------
        print("\n[B] view = ttnn.slice(...); ttnn.copy(new_tt, view)")
        cache_tt = fresh_cache(device)
        try:
            view = ttnn.slice(cache_tt,
                [TARGET_BLOCK, 0, TARGET_SLOT, 0],
                [TARGET_BLOCK + 1, N_KV, TARGET_SLOT + 1, HD])
            ttnn.copy(new_tt, view)
            verdict = verify(cache_tt, new_np, "B")
            print(f"  → {verdict}")
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  FAILED: {msg[:200]}")

        # ---------------- Pattern C: functional ----------------
        print("\n[C] cache = ttnn.copy(new_tt, ttnn.slice(...))  (treat as functional)")
        cache_tt = fresh_cache(device)
        try:
            cache_tt = ttnn.copy(new_tt, ttnn.slice(cache_tt,
                [TARGET_BLOCK, 0, TARGET_SLOT, 0],
                [TARGET_BLOCK + 1, N_KV, TARGET_SLOT + 1, HD]))
            # NB: This usually fails because copy returns the dst (the slice),
            # not the original cache. cache_tt is now the slice (1, 4, 1, 32).
            # If reading still works, expected vs actual will tell us.
            if tuple(cache_tt.shape) != (MAX_BLOCKS, N_KV, P, HD):
                print(f"  → cache_tt shape is now {tuple(cache_tt.shape)}, NOT the original. Pattern doesn't fit.")
            else:
                verdict = verify(cache_tt, new_np, "C")
                print(f"  → {verdict}")
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  FAILED: {msg[:200]}")

        print("\n" + "=" * 64)
        print("INTERPRETATION")
        print("=" * 64)
        print("  PASS                  → IN-PLACE WRITES WORK. Unblocks C'0.5 + C'4.")
        print("  NO-WRITE              → ttnn.slice materializes; need another approach.")
        print("  WRITE-OK-BUT-LEAKED   → write happened but corrupted other positions.")
        print("  FAILED                → op rejects the call; try other ops.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
