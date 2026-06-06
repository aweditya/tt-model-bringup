"""Round 5 isolation probe: replace `neg + concat` in `_apply_full_rope`
with `ttnn.roll` + pre-signed sin table.

Math currently in _apply_full_rope (server_gemma4_unified_ttnn.py:1050-1060):
    x1 = slice(x, [0, 0], [n, half])          # view (free)
    x2 = slice(x, [0, half], [n, head_dim])   # view (free)
    neg_x2 = neg(x2)                          # 1 op (UnaryNg)
    rotated = concat([neg_x2, x1], dim=-1)    # 1 op (Concat)
    x_cos = mul(x, cos)                       # 1 op (BinaryNg)
    x_rope = addcmul(x_cos, rotated, sin, value=1.0)  # 1 op (Ternary)
# = 4 device ops

Proposed: roll x by half along last dim → `[x2, x1]` (NOT `[-x2, x1]`).
Pre-bake the negation into the sin table:
    sin_signed = sin * [-1, ..., -1, +1, ..., +1]   (first half negated)
Then:
    swapped = roll(x, shifts=half, dim=-1)              # 1 op
    x_cos = mul(x, cos)                                  # 1 op
    x_rope = addcmul(x_cos, swapped, sin_signed, 1.0)    # 1 op
# = 3 device ops, saves 1 op per RoPE call × 96 calls = 96 ops/forward

Why bit-equivalent:
    rotated * sin = concat([-x2, x1]) * concat([sin1, sin2])
                  = concat([-x2 * sin1, x1 * sin2])
    swapped * sin_signed = concat([x2, x1]) * concat([-sin1, sin2])
                         = concat([x2 * -sin1, x1 * sin2])
                         = concat([-x2 * sin1, x1 * sin2])  ← same

But Gemma 4 RoPE has cos/sin where cos1==cos2 and sin1==sin2 (both halves
are duplicate copies of the same half-dim frequencies). So sin_signed is:
    [-sin_freq_0, -sin_freq_1, ..., -sin_freq_{half-1},
      +sin_freq_0, +sin_freq_1, ..., +sin_freq_{half-1}]

Bake at bootstrap: state.sin_sliding_signed_tt = sin_sliding_tt * sign_mask
(where sign_mask = [-1]*half + [+1]*half).

Gate: cos(baseline, fused) > 0.99999 + max|delta| < 0.05 (bf16 round-off).

Forks: experiments/cb/isolate/gm4_addcmul_rope_probe.py (round 4 probe
structure: single-device + cos check + sliding + global).
"""
from __future__ import annotations

import sys
import time

import torch

import ttnn

N_HEADS = 4
HEAD_DIM = 256


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rotate_half_torch(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope_baseline(x, cos, sin, n_heads, head_dim):
    """Current 4-op chain (after round 4 addcmul fusion)."""
    half = head_dim // 2
    x1 = ttnn.slice(x, [0, 0], [n_heads, half])
    x2 = ttnn.slice(x, [0, half], [n_heads, head_dim])
    neg_x2 = ttnn.neg(x2)
    rotated = ttnn.concat([neg_x2, x1], dim=-1)
    ttnn.deallocate(neg_x2)
    x_cos = ttnn.mul(x, cos)
    x_rope = ttnn.addcmul(x_cos, rotated, sin, value=1.0)
    ttnn.deallocate(x_cos)
    ttnn.deallocate(rotated)
    return x_rope


def _apply_rope_roll(x, cos, sin_signed, n_heads, head_dim):
    """3-op variant: ttnn.roll + addcmul with pre-signed sin."""
    half = head_dim // 2
    swapped = ttnn.roll(x, shifts=half, dim=-1)  # ttnn signature: (shifts: int, dim: int)
    x_cos = ttnn.mul(x, cos)
    x_rope = ttnn.addcmul(x_cos, swapped, sin_signed, value=1.0)
    ttnn.deallocate(x_cos)
    ttnn.deallocate(swapped)
    return x_rope


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main(state=None):
    if state is None:
        device = ttnn.open_device(device_id=0)
        owned = True
    else:
        device = state.mesh
        owned = False

    torch.manual_seed(0)
    pos = 42
    half = HEAD_DIM // 2

    # Sign mask: [-1]*half + [+1]*half. Broadcasts across n_heads dim.
    sign_mask_t = torch.cat([
        -torch.ones(half),
        torch.ones(half),
    ]).unsqueeze(0)  # [1, head_dim]

    x_t = torch.randn(N_HEADS, HEAD_DIM, dtype=torch.float32) * 1.5
    inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    angles = pos * inv_freq
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos_t = torch.cat([cos_half, cos_half]).unsqueeze(0)  # [1, head_dim]
    sin_t = torch.cat([sin_half, sin_half]).unsqueeze(0)
    sin_signed_t = sin_t * sign_mask_t

    # Torch reference (rotate_half-style).
    y_ref = x_t * cos_t + _rotate_half_torch(x_t) * sin_t

    # Torch sanity: roll-style should produce same result.
    x_rolled = torch.roll(x_t, shifts=half, dims=-1)  # [x2, x1]
    y_roll = x_t * cos_t + x_rolled * sin_signed_t
    cos_torch = _cos(y_ref, y_roll)
    log(f"[torch sanity] cos(rotate_half_ref, roll_signed_ref) = {cos_torch:.7f}")
    assert cos_torch > 0.999999, "torch-level identity broken — math wrong"

    # Mesh path (matches dev harness).
    mapper = ttnn.ReplicateTensorToMesh(device)
    composer = ttnn.ConcatMeshToTensor(device, dim=0)

    def to_tt(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=mapper,
        )

    def to_torch_(tt, expected_shape):
        arr = ttnn.to_torch(tt, mesh_composer=composer)
        return arr[:expected_shape[0]].reshape(*expected_shape)

    x_tt = to_tt(x_t)
    cos_tt = to_tt(cos_t)
    sin_tt = to_tt(sin_t)
    sin_signed_tt = to_tt(sin_signed_t)

    y_b_tt = _apply_rope_baseline(x_tt, cos_tt, sin_tt, N_HEADS, HEAD_DIM)
    y_b = to_torch_(y_b_tt, (N_HEADS, HEAD_DIM))
    ttnn.deallocate(y_b_tt); ttnn.deallocate(x_tt)

    x_tt = to_tt(x_t)
    y_r_tt = _apply_rope_roll(x_tt, cos_tt, sin_signed_tt, N_HEADS, HEAD_DIM)
    y_r = to_torch_(y_r_tt, (N_HEADS, HEAD_DIM))
    ttnn.deallocate(y_r_tt); ttnn.deallocate(x_tt)
    ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt); ttnn.deallocate(sin_signed_tt)

    cos_pair = _cos(y_b, y_r)
    max_d = float((y_b - y_r).abs().max())
    mad = float((y_b - y_r).abs().mean())
    cos_ref_b = _cos(y_b, y_ref)
    cos_ref_r = _cos(y_r, y_ref)

    log(f"[sliding n={N_HEADS} d={HEAD_DIM}] cos(baseline, fused) = {cos_pair:.7f}")
    log(f"  cos(baseline, torch_ref) = {cos_ref_b:.7f}")
    log(f"  cos(fused,    torch_ref) = {cos_ref_r:.7f}")
    log(f"  max|baseline - fused|    = {max_d:.6f}")
    log(f"  mean|baseline - fused|   = {mad:.6f}")

    ok = cos_pair > 0.99999 and max_d < 0.1

    # Global path (head_dim=512).
    log("")
    log("Global path: n_heads=8, head_dim=512 …")
    head_dim_g = 512
    half_g = head_dim_g // 2
    sign_mask_g = torch.cat([
        -torch.ones(half_g),
        torch.ones(half_g),
    ]).unsqueeze(0)
    x_t_g = torch.randn(8, head_dim_g, dtype=torch.float32) * 1.5
    inv_freq_g = 1.0 / (1000000 ** (torch.arange(0, head_dim_g, 2).float() / head_dim_g))
    angles_g = pos * inv_freq_g
    cos_g = torch.cat([torch.cos(angles_g), torch.cos(angles_g)]).unsqueeze(0)
    sin_g = torch.cat([torch.sin(angles_g), torch.sin(angles_g)]).unsqueeze(0)
    sin_signed_g = sin_g * sign_mask_g

    x_tt_g = to_tt(x_t_g)
    cos_tt_g = to_tt(cos_g); sin_tt_g = to_tt(sin_g); sin_signed_tt_g = to_tt(sin_signed_g)

    y_b_tt_g = _apply_rope_baseline(x_tt_g, cos_tt_g, sin_tt_g, 8, head_dim_g)
    y_b_g = to_torch_(y_b_tt_g, (8, head_dim_g))
    ttnn.deallocate(y_b_tt_g); ttnn.deallocate(x_tt_g)
    x_tt_g = to_tt(x_t_g)
    y_r_tt_g = _apply_rope_roll(x_tt_g, cos_tt_g, sin_signed_tt_g, 8, head_dim_g)
    y_r_g = to_torch_(y_r_tt_g, (8, head_dim_g))
    ttnn.deallocate(y_r_tt_g); ttnn.deallocate(x_tt_g)
    ttnn.deallocate(cos_tt_g); ttnn.deallocate(sin_tt_g); ttnn.deallocate(sin_signed_tt_g)

    cos_g_pair = _cos(y_b_g, y_r_g)
    max_d_g = float((y_b_g - y_r_g).abs().max())
    log(f"[global  n=8 d=512]   cos(baseline, fused) = {cos_g_pair:.7f}")
    log(f"  max|baseline - fused| = {max_d_g:.6f}")
    ok_g = cos_g_pair > 0.99999 and max_d_g < 0.1

    log("")
    log(f"PASS: sliding={'yes' if ok else 'NO'}  global={'yes' if ok_g else 'NO'}")
    if owned:
        ttnn.close_device(device)
    return 0 if (ok and ok_g) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
