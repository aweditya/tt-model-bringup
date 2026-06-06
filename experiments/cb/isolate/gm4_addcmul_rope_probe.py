"""Isolation probe: validate ttnn.addcmul as a fused replacement for the
final mul+add in _apply_full_rope.

Math currently in _apply_full_rope (server_gemma4_unified_ttnn.py:1011-1048):
    x_cos        = ttnn.mul(x, cos_tt)          # x * cos
    rotated_sin  = ttnn.mul(rotated, sin_tt)    # rotated * sin
    x_rope       = ttnn.add(x_cos, rotated_sin) # sum

Proposed fused replacement (saves 1 op per call):
    x_cos        = ttnn.mul(x, cos_tt)          # x * cos
    x_rope       = ttnn.addcmul(x_cos, rotated, sin_tt, value=1.0)
                                                  # x_cos + 1.0 * rotated * sin

ttnn.addcmul(a, b, c, value) = a + value * b * c — TernaryOpType::ADDCMUL
LLK kernel; not a composite fallback (verified at
tt-metal/ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.cpp:244-302 — the
TTT path with bf16 + COL_BCAST broadcast type lands on the device kernel,
not the _addcmul composite fallback).

Probe shapes:
- x:     [n_heads=4, head_dim=256]   (sliding Q, per-chip)
- cos:   [1, head_dim=256]
- sin:   [1, head_dim=256]
- rotated: [n_heads, head_dim] (after slice+neg+concat)

Gate: PCC > 0.99999 between baseline and fused output (single chip, bf16).

Forks: experiments/cb/isolate/gm4_shard_for_paged_write_v2.py (single-device
boilerplate + comparison pattern; round 2).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

import ttnn

N_HEADS = 4       # sliding NQ_PER_CHIP at qb2 (16 / 4)
HEAD_DIM = 256    # sliding HEAD_DIM


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rotate_half_torch(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_full_rope_baseline(x, cos, sin, n_heads, head_dim):
    """Current 7-op chain from server_gemma4_unified_ttnn.py:1037-1048."""
    half = head_dim // 2
    x1 = ttnn.slice(x, [0, 0], [n_heads, half])
    x2 = ttnn.slice(x, [0, half], [n_heads, head_dim])
    neg_x2 = ttnn.neg(x2)
    rotated = ttnn.concat([neg_x2, x1], dim=-1)
    ttnn.deallocate(neg_x2)
    x_cos = ttnn.mul(x, cos)
    rotated_sin = ttnn.mul(rotated, sin)
    ttnn.deallocate(rotated)
    x_rope = ttnn.add(x_cos, rotated_sin)
    ttnn.deallocate(x_cos)
    ttnn.deallocate(rotated_sin)
    return x_rope


def _apply_full_rope_addcmul(x, cos, sin, n_heads, head_dim):
    """6-op variant: replace final mul+add with addcmul.

    Note: rotated is still needed as an intermediate but addcmul folds the
    `* sin` and `+ x_cos` into a single device dispatch.
    """
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


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main(state=None):
    # Single-device probe (no mesh). State=None path opens its own device.
    if state is None:
        device = ttnn.open_device(device_id=0)
        owned = True
    else:
        device = state.mesh
        owned = False

    torch.manual_seed(0)
    x_t = torch.randn(N_HEADS, HEAD_DIM, dtype=torch.float32) * 1.5
    # cos/sin have known structure (cos in [0,1]; sin in [-1,1]).
    pos = 42
    inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    angles = pos * inv_freq
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos_t = torch.cat([cos_half, cos_half]).unsqueeze(0)  # [1, head_dim]
    sin_t = torch.cat([sin_half, sin_half]).unsqueeze(0)

    # Torch ground truth.
    x_cos_ref = x_t * cos_t
    rotated_ref = _rotate_half_torch(x_t)
    rotated_sin_ref = rotated_ref * sin_t
    y_ref = x_cos_ref + rotated_sin_ref

    # When running under the dev harness, `device` is a mesh; tensors must
    # be created with a mesh_mapper (ReplicateTensorToMesh) so all devices
    # see the same data, and read back with a mesh_composer. We always
    # assume harness/mesh path here (the round-4 probe runs under the gm4
    # dev harness; standalone single-device path is unused).
    mapper = ttnn.ReplicateTensorToMesh(device)
    composer = ttnn.ConcatMeshToTensor(device, dim=0)

    def to_tt(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=mapper,
        )

    def to_torch(tt, expected_shape):
        arr = ttnn.to_torch(tt, mesh_composer=composer)
        # ConcatMeshToTensor(dim=0) concatenates along leading dim, so for
        # a replicated tensor of shape (n_heads, head_dim) we get
        # (NCHIPS * n_heads, head_dim). Take chip 0's slice.
        shape0 = expected_shape[0]
        return arr[:shape0].reshape(*expected_shape)

    x_tt = to_tt(x_t)
    cos_tt = to_tt(cos_t)
    sin_tt = to_tt(sin_t)

    y_baseline_tt = _apply_full_rope_baseline(x_tt, cos_tt, sin_tt, N_HEADS, HEAD_DIM)
    y_baseline = to_torch(y_baseline_tt, (N_HEADS, HEAD_DIM))
    ttnn.deallocate(y_baseline_tt)

    # Re-create x_tt because the dealloc path is destructive.
    ttnn.deallocate(x_tt)
    x_tt = to_tt(x_t)

    y_fused_tt = _apply_full_rope_addcmul(x_tt, cos_tt, sin_tt, N_HEADS, HEAD_DIM)
    y_fused = to_torch(y_fused_tt, (N_HEADS, HEAD_DIM))
    ttnn.deallocate(y_fused_tt)

    ttnn.deallocate(x_tt)
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)

    cos_ref_base = _cos(y_baseline, y_ref)
    cos_ref_fused = _cos(y_fused, y_ref)
    cos_pair = _cos(y_baseline, y_fused)
    max_abs = float((y_baseline - y_fused).abs().max())
    mad = float((y_baseline - y_fused).abs().mean())

    log(f"baseline shape: {tuple(y_baseline.shape)}  fused shape: {tuple(y_fused.shape)}")
    log(f"cos(baseline, torch_ref) = {cos_ref_base:.7f}")
    log(f"cos(fused,    torch_ref) = {cos_ref_fused:.7f}")
    log(f"cos(baseline, fused)     = {cos_pair:.7f}")
    log(f"max|baseline - fused|    = {max_abs:.6f}")
    log(f"mean|baseline - fused|   = {mad:.6f}")

    # Sliding test PASS = cos_pair > 0.99999 (bf16-equivalence to baseline).
    ok = cos_pair > 0.99999
    # Also check global head_dim=512 to confirm the larger-dim path.
    log("")
    log("Global path: n_heads=8, head_dim=512 …")
    x_t_g = torch.randn(8, 512, dtype=torch.float32) * 1.5
    inv_freq_g = 1.0 / (1000000 ** (torch.arange(0, 512, 2).float() / 512))
    angles_g = pos * inv_freq_g
    cos_g = torch.cat([torch.cos(angles_g), torch.cos(angles_g)]).unsqueeze(0)
    sin_g = torch.cat([torch.sin(angles_g), torch.sin(angles_g)]).unsqueeze(0)

    y_ref_g = x_t_g * cos_g + _rotate_half_torch(x_t_g) * sin_g
    x_tt_g = to_tt(x_t_g); cos_tt_g = to_tt(cos_g); sin_tt_g = to_tt(sin_g)
    y_b_tt_g = _apply_full_rope_baseline(x_tt_g, cos_tt_g, sin_tt_g, 8, 512)
    y_b_g = to_torch(y_b_tt_g, (8, 512))
    ttnn.deallocate(y_b_tt_g); ttnn.deallocate(x_tt_g)
    x_tt_g = to_tt(x_t_g)
    y_f_tt_g = _apply_full_rope_addcmul(x_tt_g, cos_tt_g, sin_tt_g, 8, 512)
    y_f_g = to_torch(y_f_tt_g, (8, 512))
    ttnn.deallocate(y_f_tt_g); ttnn.deallocate(x_tt_g)
    ttnn.deallocate(cos_tt_g); ttnn.deallocate(sin_tt_g)

    cos_pair_g = _cos(y_b_g, y_f_g)
    log(f"cos(baseline, fused) [global] = {cos_pair_g:.7f}")
    log(f"max|baseline - fused| [global] = {(y_b_g - y_f_g).abs().max():.6f}")
    ok_g = cos_pair_g > 0.99999

    log("")
    log(f"PASS: sliding={'yes' if ok else 'NO'}  global={'yes' if ok_g else 'NO'}")
    if owned:
        ttnn.close_device(device)
    return 0 if (ok and ok_g) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
