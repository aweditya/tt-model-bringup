#!/usr/bin/env python3
"""
P3 — replicated lm_head matmul on (1,4) mesh (qb2).

server_tp.py:Stage B loads lm_head as REPLICATED across all 4 chips. This
probe validates:
  1. Memory: 152064 × 5120 × 2 bytes = 1.55 GB per chip — fits with room
  2. Correctness: per-chip output identical, matches numpy gold (cos ≥ 0.999)
  3. Latency: rough sanity check (single matmul, should be a few ms)

Why this matters: lm_head is the LAST op of every decode step on every chip.
If replicated matmul has any per-chip drift or memory issue, every TP decode
will be silently wrong. Cheap to validate now (~3 min wall).

Shapes:
  weight: [VOCAB=152064, HIDDEN=5120] bf16
  input : [1, HIDDEN=5120] bf16
  output: [1, VOCAB=152064] bf16  (each chip computes the same answer)
"""
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


VOCAB = 152064
HIDDEN = 5120
SEED = 7


def check_replication(per_chip_outs):
    """All 4 chips should produce bit-identical output (or very close in bf16)."""
    base = per_chip_outs[0]
    for i, c in enumerate(per_chip_outs[1:], 1):
        diff = np.abs(base.astype(np.float32) - c.astype(np.float32))
        if diff.max() > 1e-3:
            print(f"  ✗ chip {i} differs from chip 0: max|Δ|={diff.max():.6f}")
            return False
    print(f"  ✓ all 4 chips agree (max|Δ| ≤ 1e-3 across pairs)")
    return True


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("=" * 78)
    print("P3: replicated lm_head matmul on (1,4) mesh (qb2)")
    print("=" * 78)
    print(f"weight shape: [{VOCAB}, {HIDDEN}] bf16  (~{VOCAB*HIDDEN*2/1e9:.2f} GB per chip)")
    print(f"input shape:  [1, {HIDDEN}] bf16")

    print("\n[1] Build numpy gold...")
    rng = np.random.default_rng(SEED)
    W_np = rng.standard_normal((VOCAB, HIDDEN), dtype=np.float32) * 0.02
    x_np = rng.standard_normal((1, HIDDEN), dtype=np.float32) * 0.5
    y_gold = x_np @ W_np.T  # [1, VOCAB]
    print(f"  ✓ gold y shape={y_gold.shape}  range=[{y_gold.min():.3f}, {y_gold.max():.3f}]")
    print(f"  ✓ gold argmax = {int(y_gold.argmax())}  top val = {y_gold.max():.3f}")

    print("\n[2] Init fabric + open mesh...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ mesh {mesh.get_num_devices()} chips")

    overall_pass = True
    try:
        # ttnn.matmul does x @ W (not x @ W.T). We feed it W.T so semantics match.
        # i.e. compute is x[1,H] @ W_t[H,V] = [1, V]
        W_t_np = W_np.T  # [HIDDEN, VOCAB]

        print("\n[3] Upload replicated weight (bf16)...")
        t0 = time.time()
        W_tt = ttnn.from_torch(
            torch.from_numpy(W_t_np),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)
        upload_s = time.time() - t0
        print(f"  ✓ upload took {upload_s:.2f}s  ({VOCAB*HIDDEN*2/1e9/upload_s:.2f} GB/s effective)")

        print("\n[4] Upload replicated input (bf16)...")
        x_tt = ttnn.from_torch(
            torch.from_numpy(x_np),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)
        print(f"  ✓ x uploaded")

        print("\n[5] Warmup matmul (JIT)...")
        _ = ttnn.matmul(x_tt, W_tt)
        ttnn.synchronize_device(mesh)
        print(f"  ✓ warmup done")

        print("\n[6] Timed matmul (5 iters)...")
        times = []
        for _ in range(5):
            t0 = time.time()
            y_tt = ttnn.matmul(x_tt, W_tt)
            ttnn.synchronize_device(mesh)
            times.append((time.time() - t0) * 1000)
        print(f"  ✓ latency ms: {[f'{t:.2f}' for t in times]}  median={np.median(times):.2f}")

        print("\n[7] Read per-chip output via ConcatMeshToTensor (dim=0)...")
        # For a replicated [1, VOCAB] tensor, ConcatMeshToTensor(dim=0) gives
        # [NCHIPS, VOCAB] — each row is one chip's copy.
        y_np_concat = ttnn.to_torch(y_tt,
                                     mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().numpy()
        NCHIPS = mesh.get_num_devices()
        print(f"  ✓ concat shape: {y_np_concat.shape}  (expect [NCHIPS={NCHIPS}, VOCAB={VOCAB}])")
        per_chip = [y_np_concat[i] for i in range(NCHIPS)]

        print("\n[8] Test replication across chips...")
        if not check_replication(per_chip):
            overall_pass = False

        print("\n[9] Test cosine vs numpy gold...")
        y_dev = per_chip[0]
        cos = cosine(y_dev, y_gold.flatten())
        argmax_dev = int(y_dev.argmax())
        argmax_gold = int(y_gold.argmax())
        max_abs = float(np.abs(y_dev - y_gold.flatten()).max())
        print(f"  device argmax: {argmax_dev}  (gold: {argmax_gold})  match: {argmax_dev == argmax_gold}")
        print(f"  cosine vs gold: {cos:.6f}  max|Δ|: {max_abs:.4f}")
        if cos < 0.999:
            print(f"  ✗ cosine below 0.999 gate")
            overall_pass = False
        elif argmax_dev != argmax_gold:
            print(f"  ✗ argmax mismatch — would produce different next token")
            overall_pass = False
        else:
            print(f"  ✓ correctness gate passed")

        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        if overall_pass:
            print("  ✓ P3 PASSES — replicated lm_head matmul on mesh works correctly")
            print(f"    Median latency: {np.median(times):.2f} ms")
            print("    server_tp.py:Stage B's replicated lm_head pattern is unblocked.")
        else:
            print("  ✗ P3 FAIL — replicated lm_head needs alternative on mesh")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
            print("\n  ✓ mesh closed cleanly")
        except Exception as e:
            print(f"  ✗ close error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric reset to DISABLED")
        except Exception as e:
            print(f"  ✗ fabric reset error: {e}")


if __name__ == "__main__":
    main()
