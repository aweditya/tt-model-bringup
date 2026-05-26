#!/usr/bin/env python3
"""
Probe: validate the "Level 1" partial-rotary trick mathematically.

Trick: extend cos/sin to the full head_dim with identity values in the
passthrough region:
  cos_extended[:rotary_dim] = real cos values
  cos_extended[rotary_dim:] = 1.0    (multiplicative identity)
  sin_extended[:rotary_dim] = real sin values
  sin_extended[rotary_dim:] = 0.0    (kills the rotate_half contribution)

Use a custom rotate_half_partial that:
  - splits the FIRST rotary_dim dims at rotary_dim/2 (rotate-half within rotary range)
  - leaves the rest unchanged

Then `q' = q * cos_ext + rotate_half_partial(q) * sin_ext` should give exactly
the same output as the current manual apply_partial_rope (which does the rotation
on the rotary slice and concats the passthrough).

This probe verifies the math is right BEFORE we touch the ttnn implementation.

Numpy only — no device. Runs in seconds locally.
"""
import sys
import numpy as np

sys.stdout.reconfigure(line_buffering=True)


def baseline_apply_partial_rope(q, cos, sin, rotary_dim):
    """Current manual implementation (matches 91f.apply_partial_rope)."""
    half = rotary_dim // 2
    rot = q[..., :rotary_dim]
    passthru = q[..., rotary_dim:]
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    rotated_half = np.concatenate([-x2, x1], axis=-1)
    rotated = rot * cos + rotated_half * sin
    return np.concatenate([rotated, passthru], axis=-1)


def level1_extended_cos_sin(q, cos, sin, rotary_dim, head_dim):
    """Level 1: extend cos/sin to full head_dim and use partial rotate_half."""
    half = rotary_dim // 2
    # Build extended cos/sin tables
    cos_ext = np.zeros(head_dim, dtype=q.dtype)
    sin_ext = np.zeros(head_dim, dtype=q.dtype)
    cos_ext[:rotary_dim] = cos
    cos_ext[rotary_dim:] = 1.0    # identity for passthrough
    sin_ext[:rotary_dim] = sin
    sin_ext[rotary_dim:] = 0.0    # zero kills rotate_half contribution

    # rotate_half_partial: split rotary range, leave rest unchanged
    x1 = q[..., :half]
    x2 = q[..., half:rotary_dim]
    passthru = q[..., rotary_dim:]
    rotated_half_partial = np.concatenate([-x2, x1, passthru], axis=-1)

    return q * cos_ext + rotated_half_partial * sin_ext


def main():
    # Qwen3.6 shapes
    head_dim = 256
    rotary_dim = 64    # partial_rotary_factor = 0.25 of head_dim
    n_heads = 32
    n_positions = 8

    rng = np.random.default_rng(42)

    # Use a sweep over multiple test positions to make sure the trick is robust
    print("=" * 64)
    print("Probe: Level 1 partial-rotary (extended cos/sin) math validation")
    print(f"  head_dim={head_dim}, rotary_dim={rotary_dim}, n_heads={n_heads}")
    print("=" * 64)
    print(f"\n{'position':>8} {'cosine':>10} {'max|Δ|':>10}  verdict")
    print("-" * 50)

    all_pass = True
    for pos in range(n_positions):
        # Build q tensor
        q = rng.standard_normal((n_heads, head_dim)).astype(np.float32) * 0.1

        # Build cos/sin for this position (half-format: each value duplicated)
        half = rotary_dim // 2
        freqs = 1.0 / (10_000_000.0 ** (np.arange(half).astype(np.float32) / half))
        angles = pos * freqs
        cos = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
        sin = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)

        # Run both implementations
        ref = baseline_apply_partial_rope(q, cos, sin, rotary_dim)
        new = level1_extended_cos_sin(q, cos, sin, rotary_dim, head_dim)

        # Compare
        ref_flat = ref.flatten().astype(np.float64)
        new_flat = new.flatten().astype(np.float64)
        cos_sim = float(np.dot(ref_flat, new_flat) / (np.linalg.norm(ref_flat) * np.linalg.norm(new_flat) + 1e-12))
        max_diff = float(np.abs(new - ref).max())
        tag = "✓" if (cos_sim > 0.99999 and max_diff < 1e-5) else "✗"
        if cos_sim < 0.99999 or max_diff > 1e-5:
            all_pass = False
        print(f"{pos:>8} {cos_sim:>10.8f} {max_diff:>10.2e}  {tag}")

    print()
    if all_pass:
        print("✓ MATH CONFIRMED: Level 1 trick produces identical output to baseline.")
        print()
        print("Op count comparison (per call):")
        print("  Current manual apply_partial_rope: 12 ttnn ops")
        print("    (2 slice + 2 slice + neg + concat + 2 reshape + 2 mul + add + concat)")
        print("  Level 1 (extended cos/sin):        ~7 ttnn ops")
        print("    (3 slice + neg + concat + 2 mul + add)")
        print("  Savings: ~5 ops × ~30 µs dispatch = ~150 µs / call")
        print("  × 16 attn layers × 1 token = ~2.4 ms/tok decode savings (est)")
    else:
        print("✗ math mismatch — re-examine the trick.")


if __name__ == "__main__":
    main()
