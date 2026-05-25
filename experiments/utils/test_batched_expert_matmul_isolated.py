#!/usr/bin/env python3
"""Isolated batched-expert-matmul test suite.

Goal: find ONE ttnn matmul shape+layout combo that does a single batched
matmul over E_LOCAL=64 stacked expert weights, so we can drop it into
moe_forward_ttnn_pattern_a_batched. Iterating in the full model takes ~5 min
per attempt; this opens the mesh once and runs the full test matrix in ~30 s.

We use a synthetic small problem with the SAME logical shapes / dtypes /
layouts the real MoE uses:
  h_tt:    [1, HIDDEN=2048] bf16 TILE replicated
  weights: [E_LOCAL=64, HIDDEN=2048, 2*MOE_INTER=1024] bf16 TILE per chip,
           sharded along expert dim or chip dim depending on the variant

For each variant we:
  1. Upload weights in the variant's layout.
  2. Try to issue the batched matmul.
  3. Compare cos vs a per-expert loop reference computed on the same weights.
  4. Time the batched matmul (sync-bounded).

A variant passes if matmul runs without exception AND cos > 0.999 vs loop ref.

Run (qb1):
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
  export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \\
  export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \\
  export ARCH_NAME=blackhole && \\
  export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \\
  export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \\
  .venv/bin/python -u experiments/utils/test_batched_expert_matmul_isolated.py
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import ttnn  # noqa: E402

# Match production
HIDDEN = 2048
MOE_INTER = 512
E_LOCAL = 64
NCHIPS = 4


def log(msg, ok=None):
    tag = "" if ok is None else ("✓ " if ok else "✗ ")
    print(f"[{time.strftime('%H:%M:%S')}] {tag}{msg}", flush=True)


def cos_np(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


# ── Single setup: synthetic h + weights as numpy ──────────────────────
rng = np.random.default_rng(42)
h_np = rng.normal(0, 1.0, size=(1, HIDDEN)).astype(np.float32)
# Weights stacked over E_LOCAL: [E_LOCAL, HIDDEN, 2*MOE_INTER] per chip.
# Pretend NCHIPS chips each have their own 64-expert slab — same shape for
# all chips in this synthetic test (don't care about per-chip differences).
weights_per_chip_np = rng.normal(0, 0.1, size=(E_LOCAL, HIDDEN, 2 * MOE_INTER)).astype(np.float32)
# Stacked across chips, just by duplicating (chips will hold IDENTICAL slabs
# in this test — the goal here is shape plumbing, not cross-chip correctness).
weights_stacked_np = np.broadcast_to(
    weights_per_chip_np, (NCHIPS, E_LOCAL, HIDDEN, 2 * MOE_INTER)
).copy()


def reference_loop_np(h, W):
    """Per-expert loop reference: matmul h with each expert's weights."""
    # h: [1, HIDDEN], W: [E_LOCAL, HIDDEN, 2*MOE_INTER]
    return np.stack([h @ W[e] for e in range(E_LOCAL)], axis=0)  # [E_LOCAL, 1, 2*MOE_INTER]


def to_ttnn_replicated(arr_np, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch.from_numpy(arr_np.astype(np.float32)),
        dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def to_ttnn_sharded(arr_np, mesh, shard_dim, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch.from_numpy(arr_np.astype(np.float32)),
        dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=shard_dim),
    )


def chip0_to_np(t, mesh):
    arr = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    ).float().numpy()
    if arr.shape[0] >= NCHIPS:
        n_per_chip = arr.shape[0] // NCHIPS
        return arr[:n_per_chip]
    return arr  # already per-chip


# Reference output (numpy)
ref_loop = reference_loop_np(h_np, weights_per_chip_np)  # [E_LOCAL, 1, 2*MOE_INTER]
ref_first_expert_cos_target = ref_loop[0]                 # for sanity sub-check


# ── Bootstrap mesh once ────────────────────────────────────────────────
def bootstrap_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    log(f"mesh open: {mesh}")
    return mesh


def teardown_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ── Variant runners ────────────────────────────────────────────────────
def variant_A_rank4_with_chip_dim(mesh):
    """Match current production: weights logical [NCHIPS, E_LOCAL, H, 2I] sharded dim 0.
    h reshaped to [1,1,1,H] and matmul'd. Expected to fail per prior attempt."""
    log("VARIANT A: rank-4 weights sharded chip-dim, h=[1,1,1,H] (no repeat)")
    W_tt = to_ttnn_sharded(weights_stacked_np, mesh, shard_dim=0)
    h_tt = to_ttnn_replicated(h_np, mesh)
    h_4d = ttnn.reshape(h_tt, [1, 1, 1, HIDDEN])
    log(f"  W_tt shape={list(W_tt.shape)} dtype={W_tt.dtype}")
    log(f"  h_4d shape={list(h_4d.shape)} dtype={h_4d.dtype}")
    out_tt = ttnn.matmul(h_4d, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    out_np = chip0_to_np(out_tt, mesh)
    cos = cos_np(out_np.reshape(-1), ref_loop.reshape(-1))
    log(f"  cos vs ref_loop = {cos:.6f}")
    ttnn.deallocate(h_tt); ttnn.deallocate(h_4d); ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return cos > 0.999


def variant_B_rank3_shard_expert_dim(mesh):
    """Weights uploaded as [E_LOCAL, H, 2I] sharded on dim 0 (the expert dim itself).
    Each chip gets E_LOCAL/NCHIPS=16 experts. h is [1,1,H] rank 3.
    NOTE: this changes the production sharding semantics — each chip would only
    have 16 experts, not 64. Documenting for completeness, not as the target."""
    log("VARIANT B: rank-3 weights sharded on expert dim (16 experts per chip)")
    # Take only first 16 experts per chip's slab
    W_for_b = np.zeros((NCHIPS * 16, HIDDEN, 2 * MOE_INTER), dtype=np.float32)
    for c in range(NCHIPS):
        W_for_b[c * 16:(c + 1) * 16] = weights_per_chip_np[c * 16:(c + 1) * 16]
    W_tt = to_ttnn_sharded(W_for_b, mesh, shard_dim=0)
    h_tt = to_ttnn_replicated(h_np, mesh)
    h_3d = ttnn.reshape(h_tt, [1, 1, HIDDEN])
    log(f"  W_tt shape={list(W_tt.shape)} (expect rank 3 logical)")
    log(f"  h_3d shape={list(h_3d.shape)}")
    out_tt = ttnn.matmul(h_3d, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    ttnn.deallocate(h_tt); ttnn.deallocate(h_3d); ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return True


def variant_C_rank3_replicated(mesh):
    """Both operands REPLICATED rank 3 — no sharding at all.
    Tests whether ttnn.matmul supports dim-0 broadcast at rank 3 in the
    non-sharded case, to isolate the broadcast question from the shard question."""
    log("VARIANT C: rank-3 BOTH replicated, h=[1,1,H], W=[E_LOCAL,H,2I]")
    W_tt = to_ttnn_replicated(weights_per_chip_np, mesh)
    h_tt = to_ttnn_replicated(h_np, mesh)
    h_3d = ttnn.reshape(h_tt, [1, 1, HIDDEN])
    log(f"  W_tt shape={list(W_tt.shape)}")
    log(f"  h_3d shape={list(h_3d.shape)}")
    out_tt = ttnn.matmul(h_3d, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    out_np = chip0_to_np(out_tt, mesh)[:E_LOCAL].reshape(E_LOCAL, 1, 2 * MOE_INTER)
    cos = cos_np(out_np.reshape(-1), ref_loop.reshape(-1))
    log(f"  cos vs ref_loop = {cos:.6f}")
    ttnn.deallocate(h_tt); ttnn.deallocate(h_3d); ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return cos > 0.999


def variant_D_rank3_replicated_explicit_h_batch(mesh):
    """Rank 3, replicated weights, h pre-broadcast to [E_LOCAL, 1, H] so no
    broadcast needed in matmul itself. This is the "true bmm" path."""
    log("VARIANT D: rank-3 both replicated, h pre-broadcast to [E_LOCAL,1,H]")
    W_tt = to_ttnn_replicated(weights_per_chip_np, mesh)
    h_repeated_np = np.broadcast_to(
        h_np.reshape(1, 1, HIDDEN), (E_LOCAL, 1, HIDDEN)
    ).copy()
    h_tt = to_ttnn_replicated(h_repeated_np, mesh)
    log(f"  W_tt shape={list(W_tt.shape)}")
    log(f"  h_tt shape={list(h_tt.shape)}")
    out_tt = ttnn.matmul(h_tt, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    out_np = chip0_to_np(out_tt, mesh)[:E_LOCAL].reshape(E_LOCAL, 1, 2 * MOE_INTER)
    cos = cos_np(out_np.reshape(-1), ref_loop.reshape(-1))
    log(f"  cos vs ref_loop = {cos:.6f}")
    ttnn.deallocate(h_tt); ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return cos > 0.999


def variant_E_rank3_weights_sharded_chip_dim(mesh):
    """Weights logical [NCHIPS, E_LOCAL, H, 2I] but RESHAPED at upload to
    rank 3 [E_LOCAL, H, 2I] per chip. Still sharded along dim 0, but the
    logical shape after reshape no longer carries the chip dim.

    The trick: reshape ttnn tensor from [4, 64, H, 2I] to [4*64=256, H, 2I]?
    No — that would conflate experts across chips. We want each chip to
    see ONLY its 64 experts as [64, H, 2I].

    One approach: upload from numpy stacked as [NCHIPS*E_LOCAL, H, 2I] with
    shard_dim=0 → per chip [E_LOCAL, H, 2I]. The shard math gives each chip
    a non-overlapping E_LOCAL slice.
    """
    log("VARIANT E: weights as [NCHIPS*E_LOCAL=256, H, 2I] sharded dim 0 → per chip [E_LOCAL, H, 2I]")
    W_flat_np = weights_stacked_np.reshape(NCHIPS * E_LOCAL, HIDDEN, 2 * MOE_INTER)
    W_tt = to_ttnn_sharded(W_flat_np, mesh, shard_dim=0)  # per chip [E_LOCAL, H, 2I]
    h_tt = to_ttnn_replicated(h_np, mesh)
    h_3d = ttnn.reshape(h_tt, [1, 1, HIDDEN])
    log(f"  W_tt shape={list(W_tt.shape)} (expect [E_LOCAL, H, 2I] per chip rank 3 if shard collapses)")
    log(f"  h_3d shape={list(h_3d.shape)}")
    out_tt = ttnn.matmul(h_3d, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    out_np = chip0_to_np(out_tt, mesh)[:E_LOCAL].reshape(E_LOCAL, 1, 2 * MOE_INTER)
    cos = cos_np(out_np.reshape(-1), ref_loop.reshape(-1))
    log(f"  cos vs ref_loop = {cos:.6f}")
    ttnn.deallocate(h_tt); ttnn.deallocate(h_3d); ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return cos > 0.999


def variant_F_rank3_sharded_h_repeat(mesh):
    """Same as E but h pre-broadcast to [E_LOCAL, 1, H] so no matmul broadcast."""
    log("VARIANT F: rank-3 sharded weights + h pre-broadcast to [E_LOCAL, 1, H]")
    W_flat_np = weights_stacked_np.reshape(NCHIPS * E_LOCAL, HIDDEN, 2 * MOE_INTER)
    W_tt = to_ttnn_sharded(W_flat_np, mesh, shard_dim=0)
    h_repeated_np = np.broadcast_to(
        h_np.reshape(1, 1, HIDDEN), (E_LOCAL, 1, HIDDEN)
    ).copy()
    h_tt = to_ttnn_replicated(h_repeated_np, mesh)
    log(f"  W_tt shape={list(W_tt.shape)}")
    log(f"  h_tt shape={list(h_tt.shape)}")
    out_tt = ttnn.matmul(h_tt, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    out_np = chip0_to_np(out_tt, mesh)[:E_LOCAL].reshape(E_LOCAL, 1, 2 * MOE_INTER)
    cos = cos_np(out_np.reshape(-1), ref_loop.reshape(-1))
    log(f"  cos vs ref_loop = {cos:.6f}")
    ttnn.deallocate(h_tt); ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return cos > 0.999


def variant_G_rank3_replicated_ttnn_repeat(mesh):
    """Like D but the h broadcast is done ON DEVICE via ttnn.repeat."""
    log("VARIANT G: rank-3 replicated W + on-device ttnn.repeat for h")
    W_tt = to_ttnn_replicated(weights_per_chip_np, mesh)
    h_tt = to_ttnn_replicated(h_np, mesh)
    h_3d = ttnn.reshape(h_tt, [1, 1, HIDDEN])
    log(f"  h_3d before repeat: shape={list(h_3d.shape)}")
    h_repeated = ttnn.repeat(h_3d, [E_LOCAL, 1, 1])  # [E_LOCAL, 1, HIDDEN]
    log(f"  h_repeated: shape={list(h_repeated.shape)}")
    out_tt = ttnn.matmul(h_repeated, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    out_np = chip0_to_np(out_tt, mesh)[:E_LOCAL].reshape(E_LOCAL, 1, 2 * MOE_INTER)
    cos = cos_np(out_np.reshape(-1), ref_loop.reshape(-1))
    log(f"  cos vs ref_loop = {cos:.6f}")
    ttnn.deallocate(h_tt); ttnn.deallocate(h_3d); ttnn.deallocate(h_repeated)
    ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return cos > 0.999


def variant_H_rank3_sharded_ttnn_repeat(mesh):
    """Like F (the target prod layout) but with on-device ttnn.repeat for h."""
    log("VARIANT H: rank-3 [256,H,2I] sharded W + on-device ttnn.repeat for h")
    W_flat_np = weights_stacked_np.reshape(NCHIPS * E_LOCAL, HIDDEN, 2 * MOE_INTER)
    W_tt = to_ttnn_sharded(W_flat_np, mesh, shard_dim=0)
    h_tt = to_ttnn_replicated(h_np, mesh)
    h_3d = ttnn.reshape(h_tt, [1, 1, HIDDEN])
    log(f"  h_3d before repeat: shape={list(h_3d.shape)}")
    h_repeated = ttnn.repeat(h_3d, [E_LOCAL, 1, 1])  # [E_LOCAL, 1, HIDDEN]
    log(f"  h_repeated: shape={list(h_repeated.shape)}")
    out_tt = ttnn.matmul(h_repeated, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    out_np = chip0_to_np(out_tt, mesh)[:E_LOCAL].reshape(E_LOCAL, 1, 2 * MOE_INTER)
    cos = cos_np(out_np.reshape(-1), ref_loop.reshape(-1))
    log(f"  cos vs ref_loop = {cos:.6f}")
    ttnn.deallocate(h_tt); ttnn.deallocate(h_3d); ttnn.deallocate(h_repeated)
    ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return cos > 0.999


def variant_I_rank3_sharded_concat(mesh):
    """Like H but use ttnn.concat to broadcast h instead of ttnn.repeat."""
    log("VARIANT I: rank-3 [256,H,2I] sharded W + on-device ttnn.concat([h]*E_LOCAL)")
    W_flat_np = weights_stacked_np.reshape(NCHIPS * E_LOCAL, HIDDEN, 2 * MOE_INTER)
    W_tt = to_ttnn_sharded(W_flat_np, mesh, shard_dim=0)
    h_tt = to_ttnn_replicated(h_np, mesh)
    h_3d = ttnn.reshape(h_tt, [1, 1, HIDDEN])
    h_repeated = ttnn.concat([h_3d] * E_LOCAL, dim=0)  # [E_LOCAL, 1, HIDDEN]
    log(f"  h_repeated: shape={list(h_repeated.shape)}")
    out_tt = ttnn.matmul(h_repeated, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    out_np = chip0_to_np(out_tt, mesh)[:E_LOCAL].reshape(E_LOCAL, 1, 2 * MOE_INTER)
    cos = cos_np(out_np.reshape(-1), ref_loop.reshape(-1))
    log(f"  cos vs ref_loop = {cos:.6f}")
    ttnn.deallocate(h_tt); ttnn.deallocate(h_3d); ttnn.deallocate(h_repeated)
    ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return cos > 0.999


def _make_production_like_h(mesh):
    """Put h through a chain of ops similar to layer_forward_ttnn so its
    memory_config / layout / buffer state matches what moe_forward sees:
      embed-style → to_layout(TILE) → rms_norm → add → reshape

    The actual values are different (synthetic), but the OPS THAT TOUCHED
    THE BUFFER are the same — which is what determines the memory_config
    we end up with.
    """
    # ROW_MAJOR uint32 token buffer → embedding(table) → bf16 ROW_MAJOR
    tok_np = np.array([[0]], dtype=np.int32)
    tok_buf = ttnn.from_torch(
        torch.from_numpy(tok_np),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    embed_table_np = rng.normal(0, 0.5, size=(100, HIDDEN)).astype(np.float32)
    embed_table = ttnn.from_torch(
        torch.from_numpy(embed_table_np),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    embed_out = ttnn.embedding(tok_buf, embed_table)
    h_tt = ttnn.to_layout(embed_out, ttnn.TILE_LAYOUT)
    ttnn.deallocate(embed_out); ttnn.deallocate(tok_buf); ttnn.deallocate(embed_table)
    # rms_norm
    gamma_np = np.ones(HIDDEN, dtype=np.float32) * 1.117
    gamma = ttnn.from_torch(
        torch.from_numpy(gamma_np),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    h_normed = ttnn.rms_norm(h_tt, weight=gamma, epsilon=1e-6)
    ttnn.deallocate(h_tt); ttnn.deallocate(gamma)
    # add (residual style)
    h_other_np = rng.normal(0, 0.1, size=(1, HIDDEN)).astype(np.float32)
    h_other = ttnn.from_torch(
        torch.from_numpy(h_other_np),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    h_sum = ttnn.add(h_normed, h_other)
    ttnn.deallocate(h_normed); ttnn.deallocate(h_other)
    return h_sum


def variant_J_production_like_h_concat(mesh):
    """Same as I, but h is constructed via the production-like op chain to
    expose the actual failure we see in moe_forward_ttnn_pattern_a_batched."""
    log("VARIANT J: production-like h (embed→layout→rms_norm→add→reshape) + concat")
    W_flat_np = weights_stacked_np.reshape(NCHIPS * E_LOCAL, HIDDEN, 2 * MOE_INTER)
    W_tt = to_ttnn_sharded(W_flat_np, mesh, shard_dim=0)
    h_tt = _make_production_like_h(mesh)
    log(f"  h_tt post-prod-chain: shape={list(h_tt.shape)} dtype={h_tt.dtype}")
    h_3d = ttnn.reshape(h_tt, [1, 1, HIDDEN])
    log(f"  h_3d post-reshape:    shape={list(h_3d.shape)}")
    h_repeated = ttnn.concat([h_3d] * E_LOCAL, dim=0)
    log(f"  h_repeated:           shape={list(h_repeated.shape)}")
    out_tt = ttnn.matmul(h_repeated, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    ttnn.deallocate(h_tt); ttnn.deallocate(h_3d); ttnn.deallocate(h_repeated)
    ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return True  # if it ran at all, that's the goal — value comparison not meaningful here


def variant_K_production_like_h_repeat(mesh):
    """Variant J but with ttnn.repeat."""
    log("VARIANT K: production-like h + ttnn.repeat")
    W_flat_np = weights_stacked_np.reshape(NCHIPS * E_LOCAL, HIDDEN, 2 * MOE_INTER)
    W_tt = to_ttnn_sharded(W_flat_np, mesh, shard_dim=0)
    h_tt = _make_production_like_h(mesh)
    h_3d = ttnn.reshape(h_tt, [1, 1, HIDDEN])
    h_repeated = ttnn.repeat(h_3d, [E_LOCAL, 1, 1])
    log(f"  h_repeated: shape={list(h_repeated.shape)}")
    out_tt = ttnn.matmul(h_repeated, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    ttnn.deallocate(h_tt); ttnn.deallocate(h_3d); ttnn.deallocate(h_repeated)
    ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return True


def variant_L_production_like_h_reshape4d_then_concat(mesh):
    """If reshape from rank-2 [1,H] to rank-3 [1,1,H] is producing a 'fake'
    rank-3 tensor that's actually rank-2 with implicit batch, maybe explicit
    via to_layout or to_memory_config before the concat will materialize it."""
    log("VARIANT L: production-like h + to_memory_config(DRAM) before concat")
    W_flat_np = weights_stacked_np.reshape(NCHIPS * E_LOCAL, HIDDEN, 2 * MOE_INTER)
    W_tt = to_ttnn_sharded(W_flat_np, mesh, shard_dim=0)
    h_tt = _make_production_like_h(mesh)
    log(f"  h_tt mem: {h_tt.memory_config()}")
    h_3d = ttnn.reshape(h_tt, [1, 1, HIDDEN])
    # Try forcing a fresh DRAM memory_config materialization
    h_3d_dram = ttnn.to_memory_config(h_3d, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(h_3d)
    h_repeated = ttnn.concat([h_3d_dram] * E_LOCAL, dim=0)
    log(f"  h_repeated: shape={list(h_repeated.shape)}")
    out_tt = ttnn.matmul(h_repeated, W_tt)
    log(f"  out shape={list(out_tt.shape)}")
    ttnn.deallocate(h_tt); ttnn.deallocate(h_3d_dram); ttnn.deallocate(h_repeated)
    ttnn.deallocate(W_tt); ttnn.deallocate(out_tt)
    return True


VARIANTS = [
    ("A: rank-4 chip-sharded, broadcast", variant_A_rank4_with_chip_dim),
    ("B: rank-3 expert-sharded (16/chip)", variant_B_rank3_shard_expert_dim),
    ("C: rank-3 replicated, broadcast",   variant_C_rank3_replicated),
    ("D: rank-3 replicated, h prebroadcast", variant_D_rank3_replicated_explicit_h_batch),
    ("E: rank-3 [256,H,2I] sharded, broadcast", variant_E_rank3_weights_sharded_chip_dim),
    ("F: rank-3 [256,H,2I] sharded, h prebroadcast", variant_F_rank3_sharded_h_repeat),
    ("G: rank-3 replicated, ttnn.repeat h", variant_G_rank3_replicated_ttnn_repeat),
    ("H: rank-3 sharded, ttnn.repeat h",   variant_H_rank3_sharded_ttnn_repeat),
    ("I: rank-3 sharded, ttnn.concat h",   variant_I_rank3_sharded_concat),
    ("J: production-like h + concat",      variant_J_production_like_h_concat),
    ("K: production-like h + repeat",      variant_K_production_like_h_repeat),
    ("L: production-like h + to_memory_config(DRAM) + concat", variant_L_production_like_h_reshape4d_then_concat),
]


def main():
    log(f"reference loop output shape={ref_loop.shape}, |.|={np.linalg.norm(ref_loop):.4f}")
    mesh = bootstrap_mesh()
    results = []
    try:
        for name, fn in VARIANTS:
            log("")
            log(f"━━━ {name} ━━━")
            try:
                ok = fn(mesh)
                results.append((name, "PASS" if ok else "FAIL_cos"))
                log(f"RESULT: {'PASS ✓' if ok else 'FAIL_cos'}", ok=ok)
            except Exception as e:
                # Get the short reason from TT_FATAL/TT_THROW if present
                err = str(e)
                err_brief = err.split("info:")[-1].strip()[:200] if "info:" in err else err[:200]
                results.append((name, f"ERROR: {err_brief}"))
                log(f"RESULT: ERROR — {err_brief}", ok=False)
    finally:
        teardown_mesh(mesh)

    log("")
    log("═══ SUMMARY ═══")
    for name, result in results:
        log(f"  {name:55s}  {result}")


if __name__ == "__main__":
    main()
