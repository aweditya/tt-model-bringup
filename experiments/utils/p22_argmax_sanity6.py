#!/usr/bin/env python3
"""argmax sanity #6 — REAL lm_head matmul + small random hidden.

Reproduce the broken case exactly. Then inspect what the gathered logits
actually look like at chip 0 and find the discrepancy.
"""
import sys
import os
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78)
    print("argmax sanity #6 — real lm_head + small random x")
    print("=" * 78)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"mesh chips: {mesh.get_num_devices()}")

    HIDDEN = 5120
    VOCAB = 152064
    VOCAB_PADDED = 248320

    try:
        sys.path.insert(0, "/home/aditya/tt-xla/experiments")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_91l", "/home/aditya/tt-xla/experiments/91l_fp32_residual_generate.py")
        _91l = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_91l)
        embed_weights = _91l.load_embed_lm_head_weights()
        W_np = embed_weights['lm_head']
        print(f"  W shape: {W_np.shape}  range=[{W_np.min():.3f}, {W_np.max():.3f}]")

        rng = np.random.default_rng(7)
        x_np = (rng.standard_normal((1, HIDDEN), dtype=np.float32) * 0.5).astype(np.float32)
        y_gold = x_np @ W_np
        y_gold_sliced = y_gold[:, :VOCAB]
        gold_argmax = int(y_gold_sliced.argmax())
        print(f"  gold (fp32) argmax (sliced): {gold_argmax}")
        print(f"  gold logit at argmax = {y_gold[0, gold_argmax]:.4f}")
        print(f"  gold logits range: [{y_gold.min():.3f}, {y_gold.max():.3f}]")

        # Upload weight sharded
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

        y_sh = ttnn.linear(x_tt, W_sh_tt)
        ttnn.synchronize_device(mesh)
        print(f"\n  linear output shape: {tuple(y_sh.shape)}  layout={y_sh.layout}  dtype={y_sh.dtype}")

        y_g = ttnn.all_gather(y_sh, dim=-1)
        ttnn.synchronize_device(mesh)
        y_g_np = ttnn.to_torch(y_g, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().cpu().numpy()
        print(f"  gathered shape: {tuple(y_g.shape)} concat={y_g_np.shape}")
        print(f"  gathered chip 0 range: [{y_g_np[0].min():.3f}, {y_g_np[0].max():.3f}]")
        chip0_argmax_padded = int(y_g_np[0].argmax())
        print(f"  chip 0 argmax over PADDED: {chip0_argmax_padded}")
        chip0_argmax_sliced = int(y_g_np[0][:VOCAB].argmax())
        print(f"  chip 0 argmax over SLICED [: VOCAB]: {chip0_argmax_sliced}")

        # Now go through the ttnn slice + untilize + argmax chain
        y_s = ttnn.slice(y_g, [0, 0], [1, VOCAB])
        ttnn.synchronize_device(mesh)
        y_s_np = ttnn.to_torch(y_s, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().cpu().numpy()
        print(f"\n  sliced concat shape: {y_s_np.shape}")
        print(f"  sliced chip 0 range: [{y_s_np[0].min():.3f}, {y_s_np[0].max():.3f}]")
        sliced_chip0_argmax = int(y_s_np[0].argmax())
        print(f"  sliced chip 0 argmax (numpy): {sliced_chip0_argmax}")

        y_rm = ttnn.untilize(y_s, use_multicore=True)
        ttnn.synchronize_device(mesh)
        # Read untilized back and check
        y_rm_np = ttnn.to_torch(y_rm, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().cpu().numpy()
        print(f"\n  untilized concat shape: {y_rm_np.shape}")
        print(f"  untilized chip 0 range: [{y_rm_np[0].min():.3f}, {y_rm_np[0].max():.3f}]")
        rm_argmax = int(y_rm_np[0].argmax())
        print(f"  untilized chip 0 argmax (numpy): {rm_argmax}")

        # Now ttnn.argmax
        idx_tt = ttnn.argmax(y_rm, dim=-1, keepdim=True, use_multicore=True)
        ttnn.synchronize_device(mesh)
        print(f"\n  argmax tensor shape: {tuple(idx_tt.shape)} dtype={idx_tt.dtype} layout={idx_tt.layout}")
        # Read back without composer (since result should be the same per-chip)
        idx_concat = ttnn.to_torch(idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        print(f"  to_torch concat shape: {tuple(idx_concat.shape)} dtype={idx_concat.dtype}")
        v = idx_concat.cpu().numpy().reshape(-1)
        print(f"  ttnn.argmax vals: {v.tolist()}")
        print(f"  EXPECTED: {gold_argmax}")

        # Compare what argmax should have been to what it actually returned
        for vi in v[:4]:
            vi = int(vi)
            if vi < y_rm_np[0].size:
                print(f"  val {vi} → y[{vi}] = {y_rm_np[0][vi]:.4f}  (max value in tensor = {y_rm_np[0].max():.4f})")
            else:
                # Probably bit-reinterp
                u32 = np.uint32(vi)
                fp32 = u32.view(np.float32)
                print(f"  val {vi} → as fp32 bits = {fp32:.6f}")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except Exception:
            pass
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
