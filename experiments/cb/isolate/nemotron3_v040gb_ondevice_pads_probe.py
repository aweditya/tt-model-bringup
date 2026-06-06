#!/usr/bin/env python3
"""MM7 v0.4.0g.b — on-device pad helpers correctness probe.

Validates that the device-side ttnn-pure equivalents of
`_pad_per_head_vector`, `_pad_dt_per_batch_per_head`, and
`_replicate_per_group_to_per_head` produce the same padded layout the
Mamba2 SSD kernel expects, bit-equivalent (cos > 0.999) to the
numpy reference.

Gate: cos ≥ 0.999 vs numpy for all three pads.

If PASS: we can integrate them into `mamba2_decode_step_ttnn_pure_state`
and the wrapper becomes fully ttnn.Tensor in/out (no numpy readback for
the small inputs).

REUSE: pattern at server_35b_ttnn.py:847 (ttnn.pad on 4D tile),
server_35b_ttnn.py:572 (ttnn.repeat for GQA).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

B = 1
NH = 64
HD = 64
NG = 8
SS = 128


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_and_mad(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.mean(np.abs(a - b)))
    return cos, mad


def numpy_pad_per_head_vector(arr_3d):
    """[B, NH, HD] → [B, NH, 32, HD] (value at row 0)."""
    out = np.zeros((B, NH, 32, HD), dtype=np.float32)
    out[:, :, 0, :] = arr_3d
    return out


def numpy_pad_dt(arr_2d):
    """[B, NH] → [B, NH, 32, 32] (value at [0,0])."""
    out = np.zeros((B, NH, 32, 32), dtype=np.float32)
    out[..., 0, 0] = arr_2d
    return out


def numpy_replicate_group_to_head(arr_3d):
    """[B, NG, SS] → [B, NH, 32, SS] (group-broadcast, value at row 0)."""
    heads_per_group = NH // NG
    out = np.zeros((B, NH, 32, SS), dtype=np.float32)
    for h in range(NH):
        g = h // heads_per_group
        out[:, h, 0, :] = arr_3d[:, g, :]
    return out


def device_pad_per_head_vector(x_tt, ttnn, mesh):
    """ttnn-pure equivalent. Takes x_tt at shape [B, 1, NH*HD] TILE bf16.
    Returns [B, NH, 32, HD] TILE bf16 with value at row 0.

    Pipeline:
      1. Reshape  [B, 1, NH*HD]  →  [B, 1, NH, HD]
      2. Permute  (0, 2, 1, 3)   →  [B, NH, 1, HD]
      3. Pad      [(0,0),(0,0),(0,31),(0,0)] → [B, NH, 32, HD]
    """
    x_4d = ttnn.reshape(x_tt, [B, 1, NH, HD])
    x_perm = ttnn.permute(x_4d, (0, 2, 1, 3))  # [B, NH, 1, HD]
    x_padded = ttnn.pad(x_perm, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
    return x_padded


def device_pad_dt(dt_tt, ttnn, mesh):
    """[B, 1, NH] → [B, NH, 32, 32] with value at [0,0].

    Pipeline:
      1. Reshape  [B, 1, NH]   →  [B, NH, 1, 1]
      2. Pad      [(0,0),(0,0),(0,31),(0,31)] → [B, NH, 32, 32]
    """
    dt_4d = ttnn.reshape(dt_tt, [B, NH, 1, 1])
    dt_padded = ttnn.pad(dt_4d, [(0, 0), (0, 0), (0, 31), (0, 31)], value=0.0)
    return dt_padded


def device_replicate_group_to_head(bc_tt, ttnn, mesh):
    """[B, 1, NG*SS] → [B, NH, 32, SS] (group-broadcast, value at row 0).

    Pipeline:
      1. Reshape  [B, 1, NG*SS]   →  [B, 1, NG, SS]
      2. Permute  (0, 2, 1, 3)    →  [B, NG, 1, SS]
      3. Repeat over NG dim — but we need each group → 8 consecutive heads
         (repeat_interleave). Trick: reshape to add a broadcast dim, then
         repeat that new dim, then collapse.
         [B, NG, 1, SS] → reshape [B, NG, 1, 1, SS] (logically same data)
         → repeat [1, 1, 8, 1, 1]  → [B, NG, 8, 1, SS]
         → reshape [B, NH, 1, SS]
      4. Pad [(0,0),(0,0),(0,31),(0,0)] → [B, NH, 32, SS]
    """
    heads_per_group = NH // NG
    bc_4d = ttnn.reshape(bc_tt, [B, 1, NG, SS])
    bc_perm = ttnn.permute(bc_4d, (0, 2, 1, 3))  # [B, NG, 1, SS]
    # Add interleave dim
    bc_5d = ttnn.reshape(bc_perm, [B, NG, 1, 1, SS])
    bc_rep5 = ttnn.repeat(bc_5d, ttnn.Shape([1, 1, heads_per_group, 1, 1]))
    # Collapse NG × heads_per_group → NH
    bc_4d_nh = ttnn.reshape(bc_rep5, [B, NH, 1, SS])
    bc_padded = ttnn.pad(bc_4d_nh, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
    return bc_padded


def _to_tt(arr_np, ttnn, mesh, dtype):
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(arr_np.astype(np.float32))),
        dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _to_np(t, ttnn, mesh):
    arr = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    )
    return arr[:1].float().numpy()


def main(state=None) -> int:
    import ttnn

    own_mesh = state is None
    if state is not None and getattr(state, "mesh", None) is not None:
        mesh = state.mesh
        log("[harness] reusing live mesh ✓")
    else:
        log("opening (1,4) mesh…")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4),
            l1_small_size=65536,
            trace_region_size=50_000_000,
        )

    try:
        rng = np.random.default_rng(seed=42)
        results = []

        # ── Test 1: pad_per_head_vector ────────────────────────────
        log("Test 1: pad_per_head_vector")
        x_logical = rng.standard_normal((B, NH, HD), dtype=np.float32) * 0.5
        x_flat = x_logical.reshape(B, 1, NH * HD)
        x_tt = _to_tt(x_flat, ttnn, mesh, ttnn.bfloat16)
        x_padded_tt = device_pad_per_head_vector(x_tt, ttnn, mesh)
        x_padded_np = _to_np(x_padded_tt, ttnn, mesh)
        x_ref = numpy_pad_per_head_vector(x_logical)
        cos, mad = cos_and_mad(x_ref, x_padded_np.reshape(B, NH, 32, HD))
        log(f"  cos={cos:.6f}  mad={mad:.4e}  "
            f"out_shape={list(x_padded_np.shape)}  "
            f"ref_shape={list(x_ref.shape)}")
        results.append(("pad_per_head_vector", cos, mad))
        ttnn.deallocate(x_tt)
        ttnn.deallocate(x_padded_tt)

        # ── Test 2: pad_dt ─────────────────────────────────────────
        log("Test 2: pad_dt")
        dt_logical = rng.standard_normal((B, NH), dtype=np.float32) * 0.3
        dt_flat = dt_logical.reshape(B, 1, NH)
        dt_tt = _to_tt(dt_flat, ttnn, mesh, ttnn.bfloat16)
        dt_padded_tt = device_pad_dt(dt_tt, ttnn, mesh)
        dt_padded_np = _to_np(dt_padded_tt, ttnn, mesh)
        dt_ref = numpy_pad_dt(dt_logical)
        cos, mad = cos_and_mad(dt_ref, dt_padded_np.reshape(B, NH, 32, 32))
        log(f"  cos={cos:.6f}  mad={mad:.4e}  "
            f"out_shape={list(dt_padded_np.shape)}")
        results.append(("pad_dt", cos, mad))
        ttnn.deallocate(dt_tt)
        ttnn.deallocate(dt_padded_tt)

        # ── Test 3: replicate_group_to_head ────────────────────────
        log("Test 3: replicate_group_to_head")
        bc_logical = rng.standard_normal((B, NG, SS), dtype=np.float32) * 0.2
        bc_flat = bc_logical.reshape(B, 1, NG * SS)
        bc_tt = _to_tt(bc_flat, ttnn, mesh, ttnn.bfloat16)
        bc_padded_tt = device_replicate_group_to_head(bc_tt, ttnn, mesh)
        bc_padded_np = _to_np(bc_padded_tt, ttnn, mesh)
        bc_ref = numpy_replicate_group_to_head(bc_logical)
        cos, mad = cos_and_mad(bc_ref, bc_padded_np.reshape(B, NH, 32, SS))
        log(f"  cos={cos:.6f}  mad={mad:.4e}  "
            f"out_shape={list(bc_padded_np.shape)}")
        results.append(("replicate_group_to_head", cos, mad))
        ttnn.deallocate(bc_tt)
        ttnn.deallocate(bc_padded_tt)

        # ── Report ─────────────────────────────────────────────────
        log("")
        log("=" * 60)
        log("REPORT")
        log("=" * 60)
        all_pass = True
        for name, cos, mad in results:
            ok = cos >= 0.999
            log(f"  {name:<28s}  cos={cos:.6f}  "
                f"mad={mad:.4e}  {'PASS' if ok else 'FAIL'}")
            all_pass = all_pass and ok
        log("")
        log(f"v0.4.0g.b probe {'PASS ✓' if all_pass else 'FAIL ✗'}")
        return 0 if all_pass else 1
    finally:
        if own_mesh:
            log("closing mesh…")
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
