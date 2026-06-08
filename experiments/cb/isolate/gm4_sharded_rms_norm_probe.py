#!/usr/bin/env python3
"""Adoption NEXT-B (Round 11) — sharded decode rms_norm isolation probe.

Forks LayerNormShardedMultiCoreProgramConfig pattern from tt-metal
`arg/gemma4_optimizations` branch (`tt/rms_norm.py:42-99`). Probes:
1. CORRECTNESS: cos(plain rms_norm, sharded rms_norm) ≥ 0.999999
2. SHAPE: handles our [1, HIDDEN=3840] 2D decode shape (may need 4D wrap)
3. PERF: per-call wall-time vs plain rms_norm

If correctness PASS + perf wins, integrate into _layer_forward_pos0_paged
behind TT_GM4_SHARDED_RMSNORM=1 (env-gated round 11 deploy).

Run via gm4 harness (target already bootstrapped):
  touch ~/tt-xla/.cache/gm4_runtime/trig/sharded_rms_norm_probe
or cold standalone:
  bash scripts/run_remote.sh experiments/cb/isolate/gm4_sharded_rms_norm_probe.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as srv  # noqa: E402

HIDDEN = srv.HIDDEN  # 3840
TILE = 32
EPS = srv.EPS


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if (na and nb) else 0.0


def build_sharded_cfg(mesh, dim):
    """Fork of arg/gemma4_optimizations tt/rms_norm.py:_build_sharded_cfg.
    Pick largest core grid whose count divides dim/32 tiles."""
    if dim % TILE != 0:
        return None
    tiles = dim // TILE
    grid = mesh.compute_with_storage_grid_size()
    best = None
    for gy in range(1, grid.y + 1):
        for gx in range(1, grid.x + 1):
            n = gx * gy
            if tiles % n == 0 and (best is None or n > best[0]):
                best = (n, gx, gy)
    if best is None or best[0] == 1:
        return None
    num_cores, gx, gy = best
    block_w = tiles // num_cores
    subblock_w = 4
    while subblock_w > 1 and block_w % subblock_w != 0:
        subblock_w -= 1
    input_memcfg = ttnn.create_sharded_memory_config(
        shape=(TILE, dim // num_cores),
        core_grid=ttnn.CoreGrid(x=gx, y=gy),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[gx, gy],
        subblock_w=subblock_w,
        block_h=1,
        block_w=block_w,
        inplace=False,
    )
    log(f"  build_sharded_cfg: dim={dim} tiles={tiles} grid={gx}×{gy} "
        f"({num_cores} cores) block_w={block_w} subblock_w={subblock_w}")
    return (input_memcfg, program_config)


def sharded_rms_norm(x, weight, eps, cfg):
    """Fork of tt/rms_norm.py:_forward_sharded.
    x must be rank-4 with shape[-2] == TILE (auto-padded by reshape).
    Returns interleaved DRAM tensor of same shape.
    """
    x_sh = ttnn.to_memory_config(x, cfg[0])
    out = ttnn.rms_norm(x_sh, weight=weight, epsilon=eps, program_config=cfg[1])
    ttnn.deallocate(x_sh)
    return ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)


def main(state=None):
    cold_start = state is None
    if cold_start:
        log("cold-start: bootstrap target (~70s)")
        state = srv.State()
        t0 = time.time()
        srv.bootstrap(state, log=log)
        log(f"  bootstrap took {time.time()-t0:.1f}s")
    else:
        log("dev-harness: using pre-bootstrapped state")

    mesh = state.mesh
    log(f"mesh grid: {mesh.compute_with_storage_grid_size()}")

    # Build sharded config for HIDDEN=3840.
    cfg = build_sharded_cfg(mesh, HIDDEN)
    if cfg is None:
        log("✗ no usable sharded config at hidden=3840 — abort")
        return 1

    # Reference weight: layer 0's input_layernorm. Use a synthetic activation
    # so we don't need to run the model — just a random [1, HIDDEN] tensor.
    w = state.per_layer_tt[0]["input_layernorm"]
    log(f"weight shape: {list(w.shape)} dtype: {w.dtype}")

    torch.manual_seed(0)
    x_torch = torch.randn(1, HIDDEN, dtype=torch.float32) * 0.5
    # Plain path: rank-2 input.
    x_plain = ttnn.from_torch(
        x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    log(f"x_plain shape: {list(x_plain.shape)}")

    # ── PATH A: plain rms_norm ──
    log("─" * 64)
    log("PATH A: plain rms_norm (baseline)")
    log("─" * 64)
    # Warmup
    for _ in range(3):
        out_a = ttnn.rms_norm(x_plain, weight=w, epsilon=EPS)
        ttnn.deallocate(out_a)
    ttnn.synchronize_device(mesh)
    times_a = []
    for i in range(10):
        t = time.time()
        out_a = ttnn.rms_norm(x_plain, weight=w, epsilon=EPS)
        ttnn.synchronize_device(mesh)
        times_a.append((time.time() - t) * 1e6)  # μs
        if i < 9:
            ttnn.deallocate(out_a)
    log(f"  plain per-call (μs): "
        f"{[f'{t:.1f}' for t in times_a[:5]]}... mean={np.mean(times_a):.1f} "
        f"median={np.median(times_a):.1f}")
    out_a_np = ttnn.to_torch(
        out_a, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    ).float().numpy()[0]  # chip 0
    ttnn.deallocate(out_a)

    # ── PATH B: sharded rms_norm via 4D wrap ──
    log("─" * 64)
    log("PATH B: sharded rms_norm (LayerNormShardedMultiCoreProgramConfig)")
    log("─" * 64)
    # Reshape [1, HIDDEN] → [1, 1, 1, HIDDEN] for the rank-4 contract.
    # to_memory_config + sharded_rms_norm expect [B, ?, ?, HIDDEN] with
    # height dim ≤ TILE.
    x_4d = ttnn.reshape(x_plain, [1, 1, 1, HIDDEN])
    log(f"x_4d shape: {list(x_4d.shape)}")

    # Warmup
    for _ in range(3):
        out_b = sharded_rms_norm(x_4d, w, EPS, cfg)
        ttnn.deallocate(out_b)
    ttnn.synchronize_device(mesh)
    times_b = []
    for i in range(10):
        t = time.time()
        out_b = sharded_rms_norm(x_4d, w, EPS, cfg)
        ttnn.synchronize_device(mesh)
        times_b.append((time.time() - t) * 1e6)  # μs
        if i < 9:
            ttnn.deallocate(out_b)
    log(f"  sharded per-call (μs): "
        f"{[f'{t:.1f}' for t in times_b[:5]]}... mean={np.mean(times_b):.1f} "
        f"median={np.median(times_b):.1f}")
    out_b_np = ttnn.to_torch(
        out_b, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    ).float().numpy()
    # out_b is rank-4 with shape [NCHIPS, 1, 1, HIDDEN]; reshape to flat.
    if out_b_np.ndim == 4:
        out_b_np = out_b_np[0, 0, 0]  # chip 0
    elif out_b_np.ndim == 3:
        out_b_np = out_b_np[0, 0]
    else:
        out_b_np = out_b_np.flatten()[:HIDDEN]
    ttnn.deallocate(out_b)

    log(f"out_a shape={out_a_np.shape}, out_b shape={out_b_np.shape}")
    if out_a_np.size != HIDDEN:
        # Try to extract HIDDEN-sized slice
        out_a_np = out_a_np.reshape(-1)[:HIDDEN]
    if out_b_np.size != HIDDEN:
        out_b_np = out_b_np.reshape(-1)[:HIDDEN]

    # ── GATE 1: CORRECTNESS ──
    c = cos(out_a_np, out_b_np)
    mad = float(np.max(np.abs(out_a_np.flatten() - out_b_np.flatten())))
    log(f"GATE 1 CORRECTNESS: cos = {c:.7f}  mad = {mad:.6f}")
    rc = 0
    if c < 0.999999:
        log(f"  ✗ FAIL — cos {c:.7f} below 0.999999 threshold")
        rc = 1
    else:
        log(f"  ✓ PASS")

    # ── GATE 2: PERF ──
    speedup = np.mean(times_a) / np.mean(times_b) if np.mean(times_b) > 0 else 0
    log(f"GATE 2 PERF: plain {np.mean(times_a):.1f} μs → sharded {np.mean(times_b):.1f} μs "
        f"({speedup:.2f}× speedup)")
    if speedup < 1.5:
        log(f"  ⚠ speedup below 1.5× — sharded path may not be worth integration")
    else:
        log(f"  ✓ PASS — {speedup:.2f}× speedup")
        # Project total impact: 4 sites × 48 layers = 192 calls / forward
        # Saved per call: plain - sharded.
        saved_us = np.mean(times_a) - np.mean(times_b)
        total_saved_ms = saved_us * 192 / 1000
        log(f"  Projected: -{total_saved_ms:.2f} ms/forward "
            f"(192 calls × {saved_us:.1f} μs saved)")

    ttnn.deallocate(x_plain)
    if cold_start:
        ttnn.close_mesh_device(mesh)
    return rc


if __name__ == "__main__":
    sys.exit(main(state=None))
