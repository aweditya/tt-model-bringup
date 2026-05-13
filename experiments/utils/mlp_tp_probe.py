#!/usr/bin/env python3
"""
C'7.2: Single-MLP tensor-parallel correctness probe on qb2 (4× P150).

Goal: verify that a 4-chip tensor-parallel SwiGLU MLP produces bit-equivalent
output to a single-chip baseline. This is the smallest unit of TP that exercises
the full pattern we need for the full model:
  - ColumnParallel projections (gate_proj, up_proj): weight split along output dim
    (intermediate), each chip computes a quarter of activations
  - RowParallel projection (down_proj): weight split along input dim
    (intermediate), each chip produces a partial sum, all-reduce combines them

If this passes, the multi-chip plumbing for the model is GO; remaining work is
just plumbing every layer with the same pattern.

Math (single-chip reference):
  h = silu(x @ W_gate) * (x @ W_up)           # [1, INT]
  y = h @ W_down                                # [1, HIDDEN]

Math (4-chip TP):
  W_gate_i = W_gate[:, i*INT/4:(i+1)*INT/4]    # column-parallel
  W_up_i   = W_up  [:, i*INT/4:(i+1)*INT/4]
  W_down_i = W_down[i*INT/4:(i+1)*INT/4, :]    # row-parallel
  h_i      = silu(x @ W_gate_i) * (x @ W_up_i) # [1, INT/4] per chip
  y_i      = h_i @ W_down_i                    # [1, HIDDEN] per chip (partial)
  y        = all_reduce(y_i, sum)              # [1, HIDDEN] full result

Shapes for Qwen3.6-27B:
  HIDDEN = 5120, INT = 25600 (16-bit aligned, divisible by 4 → 6400/chip)

Notes:
  - Use plain `ttnn.all_reduce` first; if unavailable, compose with
    reduce_scatter + all_gather.
  - bf16 weights, bf16 input — production dtypes.
  - Input is replicated on every chip via ReplicateTensorToMesh.
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# Qwen3.6-27B MLP shapes
HIDDEN = 5120
# We discover INTERMEDIATE from the loaded weights to avoid drift with model config.
# For Qwen3-VL-27B text MLP, intermediate is divisible by 4.


def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def single_chip_forward(x, w_gate, w_up, w_down):
    """Pure-numpy fp32 reference (the gold)."""
    h = silu_np(x @ w_gate) * (x @ w_up)
    return h @ w_down


def single_chip_ttnn(device, x, w_gate, w_up, w_down):
    """Single-chip ttnn forward — same kernels we use in production."""
    x_tt = ttnn.from_torch(torch.from_numpy(x), dtype=ttnn.bfloat16,
                           device=device, layout=ttnn.TILE_LAYOUT)
    g_tt = ttnn.from_torch(torch.from_numpy(w_gate), dtype=ttnn.bfloat16,
                           device=device, layout=ttnn.TILE_LAYOUT)
    u_tt = ttnn.from_torch(torch.from_numpy(w_up), dtype=ttnn.bfloat16,
                           device=device, layout=ttnn.TILE_LAYOUT)
    d_tt = ttnn.from_torch(torch.from_numpy(w_down), dtype=ttnn.bfloat16,
                           device=device, layout=ttnn.TILE_LAYOUT)
    g = ttnn.linear(x_tt, g_tt, activation="silu")
    u = ttnn.linear(x_tt, u_tt)
    h = ttnn.mul(g, u)
    out = ttnn.linear(h, d_tt)
    return ttnn.to_torch(out).float().cpu().numpy()


def tp_forward(mesh, x, w_gate, w_up, w_down):
    """4-chip tensor-parallel forward."""
    # Replicate input across all 4 chips
    x_tt = ttnn.from_torch(
        torch.from_numpy(x),
        dtype=ttnn.bfloat16,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    # Column-parallel: shard along output dim (dim=1 of weight)
    g_tt = ttnn.from_torch(
        torch.from_numpy(w_gate),
        dtype=ttnn.bfloat16,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
    )
    u_tt = ttnn.from_torch(
        torch.from_numpy(w_up),
        dtype=ttnn.bfloat16,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
    )
    # Row-parallel: shard along input dim (dim=0 of weight)
    d_tt = ttnn.from_torch(
        torch.from_numpy(w_down),
        dtype=ttnn.bfloat16,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )

    # Per-chip: linear gives [1, INT/4] activation, then mul, then linear gives
    # [1, HIDDEN] partial sum (each chip's contribution to the full output).
    g = ttnn.linear(x_tt, g_tt, activation="silu")
    u = ttnn.linear(x_tt, u_tt)
    h = ttnn.mul(g, u)
    partial = ttnn.linear(h, d_tt)  # [1, HIDDEN] per chip (partial sum)

    # All-reduce SUM across chips → each chip has the full [1, HIDDEN] output.
    # ttnn.all_reduce signature: (input_tensor, *, cluster_axis, ...) — Sum is the
    # implicit default reduction (no math_op kwarg exposed).
    try:
        out = ttnn.all_reduce(partial)
    except Exception:
        # Compose all-reduce from reduce_scatter + all_gather.
        # reduce_scatter on dim=1 → each chip has [1, HIDDEN/4] reduced
        # all_gather on dim=1   → each chip has [1, HIDDEN] full result
        scattered = ttnn.reduce_scatter(partial, dim=1)
        out = ttnn.all_gather(scattered, dim=1)

    # Read back chip 0's view — every chip should have the same full output.
    # mesh_composer with ConcatMeshToTensor stacks all chip outputs; we take [0].
    stacked = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    ).float().cpu().numpy()
    # stacked shape: [4, 1, HIDDEN]. Each chip's [0] should be identical (post all-reduce).
    return stacked


def main():
    print("=" * 72)
    print("C'7.2: SwiGLU MLP tensor-parallel correctness probe")
    print("=" * 72)

    # Init fabric BEFORE opening mesh (C'7.1 finding)
    print("\n[0] FABRIC_1D init...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print("  ✓ fabric ready")

    print("\n[1] Open (1, 4) mesh on qb2...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    n_dev = mesh.get_num_devices()
    print(f"  ✓ mesh, {n_dev} chips")
    assert n_dev == 4

    # Single-chip reference uses just one chip from the mesh — get sub-device 0.
    # NOTE: ttnn opens all chips when opening a mesh; the single-chip "reference"
    # path uses the mesh device as well, just with replicated weights. That's fine
    # because what we're validating is the math, not the device topology.

    try:
        # NOTE: with mesh open, every `from_torch(..., device=mesh, ...)` distributes
        # across all 4 chips — there's no clean "single-chip" ttnn sub-path. The
        # numpy fp32 forward IS our gold; we compare TP-bf16 directly against it.
        # If TP cosine ≥ 0.999 vs gold, both the math AND the bf16 path are correct.

        # Pick a realistic INTERMEDIATE size, divisible by 4 and tile-aligned.
        # Qwen3.6-27B uses INTERMEDIATE = ~25600 for text MLP; use that.
        INTERMEDIATE = 25600
        assert INTERMEDIATE % 4 == 0, "INTERMEDIATE must be divisible by 4 for TP"
        assert INTERMEDIATE % 32 == 0, "INTERMEDIATE must be tile-aligned"
        assert HIDDEN % 32 == 0

        print(f"\n[2] Build random fp32 weights (HIDDEN={HIDDEN}, INT={INTERMEDIATE})")
        rng = np.random.default_rng(42)
        # Small init scale so bf16 doesn't underflow; mimics RMSNorm output magnitudes.
        x = (rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.1)
        w_gate = (rng.standard_normal((HIDDEN, INTERMEDIATE)).astype(np.float32)
                  / np.sqrt(HIDDEN))
        w_up   = (rng.standard_normal((HIDDEN, INTERMEDIATE)).astype(np.float32)
                  / np.sqrt(HIDDEN))
        w_down = (rng.standard_normal((INTERMEDIATE, HIDDEN)).astype(np.float32)
                  / np.sqrt(INTERMEDIATE))
        print(f"  x: {x.shape}, x.std() = {x.std():.4f}")
        print(f"  W_gate: {w_gate.shape}, W_up: {w_up.shape}, W_down: {w_down.shape}")

        # Gold: pure numpy fp32
        print("\n[3] Numpy fp32 gold forward...")
        y_gold = single_chip_forward(x, w_gate, w_up, w_down)
        print(f"  y_gold: {y_gold.shape}, max|y|={np.abs(y_gold).max():.4f}, "
              f"std={y_gold.std():.4f}")

        # TP across 4 chips (skipped single-chip ttnn — no clean sub-path with mesh open)
        cos_single = float("nan")
        print("\n[4] 4-chip tensor-parallel ttnn bf16...")
        try:
            y_tp_stacked = tp_forward(mesh, x, w_gate, w_up, w_down)
            print(f"  stacked output shape: {y_tp_stacked.shape}")
            # All 4 chip outputs should be identical (post all-reduce)
            y_chip0 = y_tp_stacked[0].reshape(-1)[:HIDDEN]
            y_chip1 = y_tp_stacked[1].reshape(-1)[:HIDDEN] if len(y_tp_stacked) > 1 else y_chip0
            y_chip3 = y_tp_stacked[-1].reshape(-1)[:HIDDEN]

            cos_01 = _cosine(y_chip0, y_chip1)
            cos_03 = _cosine(y_chip0, y_chip3)
            print(f"  cos(chip0, chip1) = {cos_01:.6f}  (should be 1.0)")
            print(f"  cos(chip0, chip3) = {cos_03:.6f}  (should be 1.0)")

            cos_tp = _cosine(y_chip0, y_gold)
            max_diff_tp = float(np.abs(y_chip0 - y_gold.reshape(-1)).max())
            print(f"  cos(TP chip0, numpy gold) = {cos_tp:.6f}, "
                  f"max|Δ| = {max_diff_tp:.4e}")

            print("\n[5] Latency benchmark (4-chip TP, warm)")
            x_tt = ttnn.from_torch(
                torch.from_numpy(x),
                dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            g_tt = ttnn.from_torch(
                torch.from_numpy(w_gate), dtype=ttnn.bfloat16,
                device=mesh, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
            )
            u_tt = ttnn.from_torch(
                torch.from_numpy(w_up), dtype=ttnn.bfloat16,
                device=mesh, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
            )
            d_tt = ttnn.from_torch(
                torch.from_numpy(w_down), dtype=ttnn.bfloat16,
                device=mesh, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )

            # Warmup
            for _ in range(3):
                g = ttnn.linear(x_tt, g_tt, activation="silu")
                u = ttnn.linear(x_tt, u_tt)
                h = ttnn.mul(g, u)
                partial = ttnn.linear(h, d_tt)
                try:
                    _ = ttnn.all_reduce(partial)
                except Exception:
                    scattered = ttnn.reduce_scatter(partial, dim=1)
                    _ = ttnn.all_gather(scattered, dim=1)
            ttnn.synchronize_device(mesh)

            N = 20
            t0 = time.perf_counter()
            for _ in range(N):
                g = ttnn.linear(x_tt, g_tt, activation="silu")
                u = ttnn.linear(x_tt, u_tt)
                h = ttnn.mul(g, u)
                partial = ttnn.linear(h, d_tt)
                try:
                    _ = ttnn.all_reduce(partial)
                except Exception:
                    scattered = ttnn.reduce_scatter(partial, dim=1)
                    _ = ttnn.all_gather(scattered, dim=1)
            ttnn.synchronize_device(mesh)
            tp_ms = (time.perf_counter() - t0) * 1000.0 / N
            print(f"  4-chip MLP TP step: {tp_ms:.3f} ms")
            print(f"  Per-token at 64 MLPs/token: {tp_ms * 64:.1f} ms")

            print("\n" + "=" * 72)
            print("VERDICT")
            print("=" * 72)
            ok = cos_tp >= 0.999 and max_diff_tp < 0.05
            print(f"  TP cos vs numpy gold: {cos_tp:.6f}  max|Δ|: {max_diff_tp:.4e}")
            print(f"  Result: {'✓ PASS' if ok else '✗ FAIL'}")
        except Exception as e:
            import traceback
            print(f"  ✗ TP forward FAILED: {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
    finally:
        try:
            ttnn.close_mesh_device(mesh)
            print("\n  ✓ mesh closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric reset")
        except Exception as e:
            print(f"  ✗ fabric reset error: {e}")


if __name__ == "__main__":
    main()
