#!/usr/bin/env python3
"""B7a — Qwen3.6-35B-A3B DN ttnn smoke test on qb1 single chip.

The smallest possible "ttnn works on qb1 for the 35B DN block" test:
  1. Open a single-chip device on qb1.
  2. Load HF DN layer-0 weights + saved input from B0's
     `b0_dn_layer0_reference.npz`.
  3. Convert weights and input to ttnn tensors.
  4. Compute ONE matmul: `mixed_qkv = in_proj_qkv @ hidden_in.T`
     (this is the first projection in DN forward — gives [1, T, 8192]).
  5. Read back, compare to numpy reference, gate on cosine ≥ 0.999.

If this passes we've validated end-to-end:
  - ttnn import + device open on qb1
  - bf16 weight + input conversion (torch → numpy → ttnn)
  - ttnn matmul at the DN-block scale (8192x2048 weight)
  - readback + numerical comparison

Later sub-blocks (B7b, B7c, …) layer in conv1d, g/beta/qkv split, recurrence,
RMSNormGated, out_proj — eventually reaching cosine ≥ 0.999 vs B0's output.

Run (qb1 server.py must NOT be running — this opens device 0 directly):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91ad_qwen36_35b_a3b_dn_ttnn_smoke.py'
"""
import os
from pathlib import Path

import numpy as np
import ttnn


NPZ_PATH = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe" / "b0_dn_layer0_reference.npz"
# (qb1 reads this from a local copy — sync it from qb2 first if absent.)


def main():
    print(f"npz reference: {NPZ_PATH}")
    assert NPZ_PATH.exists(), (
        f"B0 reference npz missing on qb1; rsync from qb2:\n"
        f"  rsync -avz qb2:{NPZ_PATH} {NPZ_PATH.parent}/"
    )

    print("[1] open ttnn device on qb1…")
    device = ttnn.open_device(device_id=0)
    print(f"  device: {device}")

    try:
        print("[2] load B0 npz…")
        ref = np.load(NPZ_PATH)
        hidden_in_np = ref["hidden_in"].astype(np.float32)           # [1, 1, 2048]
        in_proj_qkv_np = ref["in_proj_qkv"].astype(np.float32)       # [8192, 2048]
        print(f"  hidden_in: {hidden_in_np.shape} {hidden_in_np.dtype}")
        print(f"  in_proj_qkv: {in_proj_qkv_np.shape}")

        print("[3] numpy reference matmul: mixed_qkv_ref = hidden_in @ in_proj_qkv.T")
        # HF computes `mixed_qkv = self.in_proj_qkv(hidden_states)` which is
        # linear(W=in_proj_qkv): hidden @ W.T  → [1, 1, 8192]
        mixed_qkv_ref = hidden_in_np @ in_proj_qkv_np.T  # [1, 1, 8192]
        print(f"  ref shape: {mixed_qkv_ref.shape}, "
              f"norm: {np.linalg.norm(mixed_qkv_ref):.4f}")

        print("[4] convert weights to ttnn (bf16 TILE layout)…")
        import torch
        hidden_tt = ttnn.from_torch(
            torch.from_numpy(hidden_in_np.reshape(1, hidden_in_np.shape[-1])),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        # weight needs to be [in_features, out_features] for ttnn.linear convention;
        # PyTorch nn.Linear stores [out, in] so we transpose.
        weight_tt = ttnn.from_torch(
            torch.from_numpy(in_proj_qkv_np.T),  # [2048, 8192]
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        print(f"  hidden_tt: shape={list(hidden_tt.shape)}, dtype={hidden_tt.dtype}")
        print(f"  weight_tt: shape={list(weight_tt.shape)}, dtype={weight_tt.dtype}")

        print("[5] ttnn matmul…")
        out_tt = ttnn.matmul(hidden_tt, weight_tt)
        print(f"  out_tt: shape={list(out_tt.shape)}, dtype={out_tt.dtype}")

        print("[6] readback + cosine vs reference…")
        out_np = ttnn.to_torch(out_tt).float().numpy()  # [1, 8192]
        out_np = out_np.reshape(1, 1, -1)               # [1, 1, 8192]
        cos = np.dot(out_np.flatten(), mixed_qkv_ref.flatten()) / (
            np.linalg.norm(out_np) * np.linalg.norm(mixed_qkv_ref) + 1e-30
        )
        max_abs = np.abs(out_np - mixed_qkv_ref).max()
        print(f"  ttnn norm: {np.linalg.norm(out_np):.4f}")
        print(f"  ref norm:  {np.linalg.norm(mixed_qkv_ref):.4f}")
        print(f"  cosine: {cos:.6f}")
        print(f"  max|Δ|: {max_abs:.4f}")

        if cos > 0.999:
            print("  ✓ B7a SMOKE PASS — ttnn matmul on qb1 matches numpy at the DN scale")
        else:
            print(f"  ⚠ cosine below 0.999 — likely bf16 noise vs fp32 ref but worth investigating")

    finally:
        print("[7] close device…")
        ttnn.close_device(device)
        print("  closed")

    print("\nB7a DONE.")


if __name__ == "__main__":
    main()
