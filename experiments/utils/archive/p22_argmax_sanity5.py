#!/usr/bin/env python3
"""argmax sanity #5 — exact probe sequence with linear + known peak.

Goal: reproduce the probe failure with a contrived weight that places a peak
at a known idx. If still broken, the bug is in `linear` interacting with
sharded weight + all_gather. If correct here but broken with real weights,
the bug is value-dependent (range/dtype/NaN).
"""
import sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78)
    print("argmax sanity #5 — linear with sharded weight + known peak target")
    print("=" * 78)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"mesh chips: {mesh.get_num_devices()}")

    HIDDEN = 5120
    VOCAB = 152064
    VOCAB_PADDED = 248320
    TARGET = 100000  # row in W (output index) where peak should land

    try:
        # Build W so that x = e_0 (one-hot) → W[0, :] is the logit row.
        # Set W[0, TARGET] = large value → argmax should be TARGET.
        W_np = np.zeros((HIDDEN, VOCAB_PADDED), dtype=np.float32)
        W_np[0, TARGET] = 99.0
        # Add a few smaller values elsewhere
        W_np[0, 12345] = 5.0
        W_np[0, 200000] = 50.0  # this is in PADDING — should be SLICED AWAY

        x_np = np.zeros((1, HIDDEN), dtype=np.float32)
        x_np[0, 0] = 1.0  # one-hot input on hidden dim 0

        # Verify: x @ W = [1, VOCAB_PADDED]; row 0 = W[0, :]; argmax = TARGET when sliced to VOCAB
        y_gold = x_np @ W_np
        y_gold_sliced = y_gold[:, :VOCAB]
        gold_argmax = int(y_gold_sliced.argmax())
        print(f"  gold argmax (sliced to VOCAB): {gold_argmax} (target was {TARGET})")
        print(f"  gold logit at TARGET = {y_gold[0, TARGET]:.2f}")
        print(f"  gold logit in padding (200000) = {y_gold[0, 200000]:.2f}")

        # Upload sharded W
        W_sh_tt = ttnn.from_torch(
            torch.from_numpy(W_np),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        x_tt = ttnn.from_torch(
            torch.from_numpy(x_np),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)

        # Linear
        y_sh = ttnn.linear(x_tt, W_sh_tt)
        ttnn.synchronize_device(mesh)
        print(f"\n  ttnn.linear output: shape={tuple(y_sh.shape)} layout={y_sh.layout}")
        # Read sharded matmul output for inspection
        y_sh_np = ttnn.to_torch(y_sh, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().cpu().numpy()
        print(f"  matmul output (concat dim=0): shape={y_sh_np.shape}")
        # Per-chip slab of 62080. TARGET=100000 → chip index = 100000//62080 = 1; local = 37920
        chip_idx = TARGET // (VOCAB_PADDED // 4)
        local_idx = TARGET % (VOCAB_PADDED // 4)
        print(f"  expected: chip {chip_idx}, local idx {local_idx}; value = {y_sh_np[chip_idx, local_idx]:.2f}")
        # Padding peak 200000 → chip 3, local 200000 - 186240 = 13760
        chip_pad = 200000 // (VOCAB_PADDED // 4)
        local_pad = 200000 % (VOCAB_PADDED // 4)
        print(f"  padding peak: chip {chip_pad}, local {local_pad}; value = {y_sh_np[chip_pad, local_pad]:.2f}")

        # all_gather
        y_g = ttnn.all_gather(y_sh, dim=-1)
        ttnn.synchronize_device(mesh)
        y_g_np = ttnn.to_torch(y_g, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().cpu().numpy()
        print(f"\n  gathered output: shape={tuple(y_g.shape)} concat={y_g_np.shape}")
        print(f"  gathered chip0[0, TARGET=100000] = {y_g_np[0, TARGET]:.2f} (expected ~99)")
        print(f"  gathered chip0[0, 200000] = {y_g_np[0, 200000]:.2f} (expected ~50)")
        print(f"  argmax of chip0 (full padded): {int(y_g_np[0].argmax())}")

        # slice
        y_s = ttnn.slice(y_g, [0, 0], [1, VOCAB])
        ttnn.synchronize_device(mesh)
        y_s_np = ttnn.to_torch(y_s, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().cpu().numpy()
        print(f"\n  sliced output: shape={tuple(y_s.shape)} concat={y_s_np.shape}")
        print(f"  sliced chip0[0, TARGET=100000] = {y_s_np[0, TARGET]:.2f}")
        print(f"  argmax of sliced chip0: {int(y_s_np[0].argmax())}")

        # untilize + argmax
        y_rm = ttnn.untilize(y_s, use_multicore=True)
        ttnn.synchronize_device(mesh)
        idx_tt = ttnn.argmax(y_rm, dim=-1, keepdim=True, use_multicore=True)
        ttnn.synchronize_device(mesh)
        idx_concat = ttnn.to_torch(idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        v = idx_concat.cpu().numpy().reshape(-1)
        print(f"\n  ttnn.argmax vals: {v[:4].tolist()}  expected={TARGET}")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except Exception:
            pass
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
