"""Probe: does our RoPE table match HF's reference for Qwen3.6-27B?

Hypothesis under test: our 91f / serve/server.py hardcode
    freqs = 1 / 10_000_000 ** (arange(half) / half)
might be diverging from HF's exact RoPE math at long positions, leading to
text-quality drift past the training context — a separate failure mode from
bf16 noise drift.

Config evidence already gathered (no need to repeat on-device):
  config.json text_config.rope_parameters:
    {"mrope_interleaved": true, "mrope_section": [11,11,10],
     "partial_rotary_factor": 0.25, "rope_theta": 10000000,
     "rope_type": "default"}
  -> rope_type "default" means standard RoPE; YaRN only kicks in if the user
     edits the config for >262k context. We run at MAX_POS=256 so YaRN is
     not the model author's prescribed math.
  -> mrope_interleaved=true with sections [11,11,10] is multimodal-only.
     For text-only inputs all three position-id streams collapse to the same
     stream (HF qwen3_vl_moe code: text_position_ids = position_ids[0]).
     So the interleaved mRoPE is mathematically identical to vanilla RoPE
     for text-only generation.

This probe verifies the math match numerically at a set of positions, on a
real Tenstorrent device (qb1 dev 2), comparing our serve/server tables
against HF's reference math (DeepseekV3 RotaryEmbedding code path, which
tt-qwen-36 also uses).
"""

import os
import sys
import json
import numpy as np
import torch
import ttnn

# Match the serve/server math exactly
HEAD_DIM = 256
PARTIAL_ROTARY = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY)  # 64
HALF_ROT = ROTARY_DIM // 2                    # 32
ROPE_THETA = 10_000_000.0


def server_freqs():
    """Exact copy of _build_rope_tables freq formula in serve/server.py."""
    return 1.0 / (ROPE_THETA ** (np.arange(HALF_ROT).astype(np.float32) / HALF_ROT))


def hf_reference_freqs():
    """HF / tt-qwen-36 / DeepseekV3 reference: arange(0, rotary_dim, 2)/rotary_dim.

    Mathematically identical to server_freqs by construction, but compute
    via the HF expression so we'd notice a typo.
    """
    return 1.0 / (ROPE_THETA ** (np.arange(0, ROTARY_DIM, 2).astype(np.float32) / ROTARY_DIM))


def server_cos_sin(positions):
    """Reproduce server's cos/sin tables at given positions (numpy)."""
    freqs = server_freqs()
    ang = positions.astype(np.float32)[:, None] * freqs[None, :]
    cos_all = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1).astype(np.float32)
    sin_all = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1).astype(np.float32)
    return cos_all, sin_all


def hf_cos_sin(positions):
    """HF reference cos/sin at given positions (numpy)."""
    freqs = hf_reference_freqs()
    ang = positions.astype(np.float32)[:, None] * freqs[None, :]
    cos_all = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1).astype(np.float32)
    sin_all = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1).astype(np.float32)
    return cos_all, sin_all


def cosine_sim(a, b):
    """Cosine similarity, but treat a pair of all-zero vectors as a perfect
    match (sin(0) at pos=0 is all zeros; this is not a divergence)."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    # Reproduce server / 91f tables at a representative span of positions.
    # 0..1024 covers our MAX_POS=256 production path plus a stress region.
    # Also poke 32768 / 65536 / 131072 / 262143 to confirm math at the model's
    # native training-context boundary (262k).
    positions_probe = np.array([0, 1, 127, 128, 255, 256, 511, 512, 1023, 1024,
                                4096, 16384, 32768, 65536, 131072, 262143],
                               dtype=np.int64)

    print("=" * 78)
    print("ROPE_SCALING PROBE")
    print(f"HEAD_DIM={HEAD_DIM}  ROTARY_DIM={ROTARY_DIM}  HALF_ROT={HALF_ROT}  "
          f"theta={ROPE_THETA}")
    print(f"freqs[0..5]:  server   {server_freqs()[:5]}")
    print(f"freqs[0..5]:  HF ref   {hf_reference_freqs()[:5]}")
    print(f"freqs[-3:]:   server   {server_freqs()[-3:]}")
    print(f"freqs[-3:]:   HF ref   {hf_reference_freqs()[-3:]}")
    print(f"freqs equal exactly (fp32)? "
          f"{np.array_equal(server_freqs(), hf_reference_freqs())}")
    print(f"freqs max-abs diff (fp64 promotion): "
          f"{np.max(np.abs(server_freqs().astype(np.float64) - hf_reference_freqs().astype(np.float64)))}")
    print()

    cos_s, sin_s = server_cos_sin(positions_probe)  # [P, ROTARY_DIM]
    cos_h, sin_h = hf_cos_sin(positions_probe)
    print("Numpy-vs-numpy (server math vs HF math), per-position cos cosine:")
    for i, p in enumerate(positions_probe):
        c = cosine_sim(cos_s[i], cos_h[i])
        s = cosine_sim(sin_s[i], sin_h[i])
        d_max = float(np.max(np.abs(cos_s[i] - cos_h[i])))
        print(f"  pos={p:>7d}  cos_cos={c:.12f}  sin_cos={s:.12f}  "
              f"max|Δcos|={d_max:.3e}")
    print()

    # Now upload server's table to the device (ttnn.float32) and slice it,
    # exactly as serve/server.py does, then compare against the HF reference
    # to catch any quantization/layout corruption along the device round-trip.
    print("-" * 78)
    print("Opening qb1 device 2 …")
    device = ttnn.open_device(device_id=2)

    # Build a table sized for the largest probe position. The server's
    # _build_rope_tables uploads cos_ext (HEAD_DIM-wide with pad). We mirror
    # that to mimic the exact production tensor.
    table_size = int(positions_probe.max()) + 1
    print(f"  table_size = {table_size}")
    positions_all = np.arange(table_size).astype(np.int64)
    cos_all, sin_all = server_cos_sin(positions_all)
    pad = HEAD_DIM - ROTARY_DIM
    cos_ext_pad = np.ones((table_size, pad), dtype=np.float32)
    sin_ext_pad = np.zeros((table_size, pad), dtype=np.float32)
    cos_ext_all = np.concatenate([cos_all, cos_ext_pad], axis=-1).astype(np.float32)
    sin_ext_all = np.concatenate([sin_all, sin_ext_pad], axis=-1).astype(np.float32)
    print(f"  cos_ext_all shape={cos_ext_all.shape}  "
          f"nbytes={cos_ext_all.nbytes/1e6:.1f}MB")

    cos_ext_tt = ttnn.from_torch(torch.from_numpy(cos_ext_all),
                                  dtype=ttnn.float32,
                                  device=device,
                                  layout=ttnn.TILE_LAYOUT)
    sin_ext_tt = ttnn.from_torch(torch.from_numpy(sin_ext_all),
                                  dtype=ttnn.float32,
                                  device=device,
                                  layout=ttnn.TILE_LAYOUT)
    print("  uploaded cos_ext / sin_ext to device")

    print()
    print("Device round-trip (ttnn.slice → to_torch) vs HF reference math:")
    pass_all = True
    for p in positions_probe.tolist():
        if p >= table_size:
            continue
        cos_slice = ttnn.slice(cos_ext_tt, [p, 0], [p + 1, HEAD_DIM])
        sin_slice = ttnn.slice(sin_ext_tt, [p, 0], [p + 1, HEAD_DIM])
        cos_dev = ttnn.to_torch(cos_slice).float().numpy().reshape(-1)
        sin_dev = ttnn.to_torch(sin_slice).float().numpy().reshape(-1)
        # First ROTARY_DIM entries are the active rotation; remainder is pad.
        cos_dev_rot = cos_dev[:ROTARY_DIM]
        sin_dev_rot = sin_dev[:ROTARY_DIM]
        cos_hf, sin_hf = hf_cos_sin(np.array([p], dtype=np.int64))
        c = cosine_sim(cos_dev_rot, cos_hf[0])
        s = cosine_sim(sin_dev_rot, sin_hf[0])
        d_cos = float(np.max(np.abs(cos_dev_rot - cos_hf[0])))
        d_sin = float(np.max(np.abs(sin_dev_rot - sin_hf[0])))
        # Pad check (must be ones / zeros for partial-rotary passthrough)
        pad_cos_ok = bool(np.allclose(cos_dev[ROTARY_DIM:], 1.0, atol=1e-5))
        pad_sin_ok = bool(np.allclose(sin_dev[ROTARY_DIM:], 0.0, atol=1e-5))
        gate = "PASS" if (c >= 0.9999 and s >= 0.9999 and pad_cos_ok and pad_sin_ok) else "FAIL"
        if gate == "FAIL":
            pass_all = False
        print(f"  pos={p:>7d}  cos_cos={c:.10f}  sin_cos={s:.10f}  "
              f"max|Δcos|={d_cos:.2e}  max|Δsin|={d_sin:.2e}  "
              f"pad_cos={pad_cos_ok}  pad_sin={pad_sin_ok}  {gate}")
        ttnn.deallocate(cos_slice)
        ttnn.deallocate(sin_slice)

    ttnn.deallocate(cos_ext_tt)
    ttnn.deallocate(sin_ext_tt)
    ttnn.close_device(device)

    print()
    print("=" * 78)
    print(f"OVERALL: {'PASS — RoPE matches HF reference at all probed positions' if pass_all else 'FAIL — divergence detected'}")
    print("=" * 78)
    return 0 if pass_all else 1


if __name__ == "__main__":
    sys.exit(main())
