#!/usr/bin/env python3
"""B8 — Qwen3.6-35B-A3B MoE block FULL ttnn implementation on qb1 single chip.

End-to-end ttnn port of `Qwen3_5MoeSparseMoeBlock.forward()` for a single
hidden state. Reads B1's npz (input + all 256 expert weights + expected
output) and validates cosine ≥ 0.999 vs HF.

Stock ttnn ops only (consistent with B7 — owned_moe_expert_decode kernel
is a future B16 perf optimization, not a correctness gate).

Forward steps mirror HF (modeling_qwen3_5_moe.py:784-803):

  Router:
    logits = hidden @ router.weight.T                        # [1, 256]
    probs  = softmax(logits, dim=-1, dtype=fp32)             # HF uses fp32
    vals, idxs = topk(probs, k=8)
    weights = vals / sum(vals, dim=-1)                       # norm_topk_prob

  Routed experts (loop over K=8 selected):
    gate_up = hidden @ experts.gate_up_proj[idx].T           # [1, 1024]
    gate, up = chunk(gate_up, 2, dim=-1)                     # [1, 512] each
    mid     = silu(gate) * up
    out_k   = mid @ experts.down_proj[idx].T                 # [1, 2048]
    routed += weights[k] * out_k

  Shared expert (always-on dense SwiGLU):
    s_gate    = hidden @ shared_expert.gate_proj.T           # [1, 512]
    s_up      = hidden @ shared_expert.up_proj.T             # [1, 512]
    s_mid     = silu(s_gate) * s_up
    shared    = s_mid @ shared_expert.down_proj.T            # [1, 2048]
    g_scalar  = sigmoid(hidden @ shared_expert_gate.T)       # [1, 1]
    shared   *= g_scalar

  output = routed + shared

Hybrid pattern (like B7): big matmuls in ttnn, small ops (softmax/topk/
silu/sigmoid/accumulate) in numpy. Moves more to ttnn in later iterations.

Run (qb1 server must NOT be running):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91af_qwen36_35b_a3b_moe_ttnn_full.py'
"""
from pathlib import Path

import numpy as np
import torch
import ttnn


NPZ_PATH = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe" / "b1_moe_layer0_reference.npz"

HIDDEN = 2048
NUM_EXPERTS = 256
TOP_K = 8
MOE_INTER = 512
SHARED_INTER = 512


def to_ttnn(arr, device, dtype=ttnn.bfloat16):
    t = torch.from_numpy(arr.astype(np.float32))
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def from_ttnn(t):
    return ttnn.to_torch(t).float().numpy()


def matmul_via_ttnn(hidden_np, weight_np, device):
    """Single matmul hidden[1,D] @ weight[D, F] → out[1, F].

    weight_np must already be in [in, out] layout (transposed from PyTorch's
    [out, in] linear-weight convention).
    """
    h_tt = to_ttnn(hidden_np.reshape(1, hidden_np.shape[-1]), device)
    w_tt = to_ttnn(weight_np, device)
    out_tt = ttnn.matmul(h_tt, w_tt)
    out_np = from_ttnn(out_tt).reshape(1, -1)
    ttnn.deallocate(h_tt); ttnn.deallocate(w_tt); ttnn.deallocate(out_tt)
    return out_np


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    print(f"npz: {NPZ_PATH}")
    assert NPZ_PATH.exists(), f"sync B1 npz to {NPZ_PATH} first"

    print("[1] open device on qb1…")
    device = ttnn.open_device(device_id=0)
    print(f"  device: {device}")

    try:
        print("[2] load B1 npz (3 GB — may take a moment)…")
        ref = np.load(NPZ_PATH)
        hidden_in = ref["hidden_in"].astype(np.float32).reshape(1, HIDDEN)    # [1, 2048]
        expected_output = ref["output"].astype(np.float32).reshape(1, HIDDEN) # [1, 2048]
        expected_idxs = ref["selected_experts"][0]                            # [8]
        expected_weights = ref["routing_weights"][0]                          # [8]
        # weights
        router_weight = ref["router_weight"].astype(np.float32)                          # [256, 2048]
        experts_gate_up = ref["experts_gate_up_proj"].astype(np.float32)                 # [256, 1024, 2048]
        experts_down = ref["experts_down_proj"].astype(np.float32)                       # [256, 2048, 512]
        shared_gate = ref["shared_gate_proj"].astype(np.float32)                         # [512, 2048]
        shared_up = ref["shared_up_proj"].astype(np.float32)                             # [512, 2048]
        shared_down = ref["shared_down_proj"].astype(np.float32)                         # [2048, 512]
        shared_expert_gate = ref["shared_expert_gate"].astype(np.float32)                # [1, 2048]
        print(f"  hidden_in norm: {np.linalg.norm(hidden_in):.4f}")
        print(f"  expected_output norm: {np.linalg.norm(expected_output):.4f}")
        print(f"  HF top-8 experts: {expected_idxs.tolist()}")
        print(f"  HF top-8 weights sum: {expected_weights.sum():.6f}")

        # ────────────────────────────────────────────────────────────────────
        # Router: ttnn matmul + numpy softmax/topk/renorm
        # ────────────────────────────────────────────────────────────────────
        print("\n[3] router: logits = hidden @ router_weight.T (ttnn matmul)")
        router_logits = matmul_via_ttnn(hidden_in, router_weight.T, device)  # [1, 256]
        print(f"  logits stats: min={router_logits.min():.3f} max={router_logits.max():.3f}")

        print("[4] softmax (fp32) + topk + renormalize (numpy)")
        # HF uses softmax dtype=fp32
        logits_fp32 = router_logits.astype(np.float64)
        logits_fp32 -= logits_fp32.max()
        exps = np.exp(logits_fp32)
        probs = (exps / exps.sum(axis=-1, keepdims=True)).astype(np.float32)  # [1, 256]
        top_k_idxs = np.argsort(probs[0])[-TOP_K:][::-1].copy()                # [8]
        top_k_vals = probs[0, top_k_idxs].copy()                                # [8]
        weights = top_k_vals / top_k_vals.sum()                                 # renormalize
        print(f"  ttnn top-8 experts: {top_k_idxs.tolist()}")
        print(f"  HF   top-8 experts: {expected_idxs.tolist()}")
        idx_match = sorted(top_k_idxs.tolist()) == sorted(expected_idxs.tolist())
        print(f"  expert sets match (as sets): {idx_match}")

        # ────────────────────────────────────────────────────────────────────
        # Routed expert loop K=8
        # ────────────────────────────────────────────────────────────────────
        print(f"\n[5] routed experts loop (K={TOP_K})…")
        routed_out = np.zeros((1, HIDDEN), dtype=np.float32)
        for k_idx in range(TOP_K):
            e = int(top_k_idxs[k_idx])
            w = float(weights[k_idx])
            # gate_up_proj[e] is [1024, 2048]; gate_up = hidden @ gate_up_proj.T
            gate_up = matmul_via_ttnn(hidden_in, experts_gate_up[e].T, device)  # [1, 1024]
            gate = gate_up[:, :MOE_INTER]
            up = gate_up[:, MOE_INTER:]
            mid = silu(gate) * up                                                # [1, 512]
            # down_proj[e] is [2048, 512]; expert_out = mid @ down_proj.T
            expert_out = matmul_via_ttnn(mid, experts_down[e].T, device)         # [1, 2048]
            routed_out += w * expert_out
            print(f"  expert {e:3d} (rank {k_idx}, w={w:.4f}): mid_norm={np.linalg.norm(mid):.4f} "
                  f"out_norm={np.linalg.norm(expert_out):.4f}")
        print(f"  routed_out total norm: {np.linalg.norm(routed_out):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Shared expert
        # ────────────────────────────────────────────────────────────────────
        print("\n[6] shared expert (always-on dense SwiGLU + scalar sigmoid gate)")
        s_gate = matmul_via_ttnn(hidden_in, shared_gate.T, device)   # [1, 512]
        s_up = matmul_via_ttnn(hidden_in, shared_up.T, device)
        s_mid = silu(s_gate) * s_up                                  # [1, 512]
        shared_out = matmul_via_ttnn(s_mid, shared_down.T, device)   # [1, 2048]
        # Scalar gate: hidden @ [1, 2048].T → [1, 1] → sigmoid
        g_scalar_logit = matmul_via_ttnn(hidden_in, shared_expert_gate.T, device)  # [1, 1]
        g_scalar = sigmoid(g_scalar_logit)
        print(f"  shared_out norm pre-gate: {np.linalg.norm(shared_out):.4f}")
        print(f"  scalar gate value: {g_scalar[0, 0]:.6f}")
        shared_out *= g_scalar
        print(f"  shared_out norm post-gate: {np.linalg.norm(shared_out):.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Sum + compare
        # ────────────────────────────────────────────────────────────────────
        print("\n[7] output = routed + shared, compare vs B1")
        output = routed_out + shared_out
        cos = (output.flatten() @ expected_output.flatten()) / (
            np.linalg.norm(output) * np.linalg.norm(expected_output) + 1e-30
        )
        max_abs = np.abs(output - expected_output).max()
        print(f"  expected norm: {np.linalg.norm(expected_output):.6f}")
        print(f"  ttnn-mix norm: {np.linalg.norm(output):.6f}")
        print(f"  cosine: {cos:.6f}")
        print(f"  max|Δ|: {max_abs:.4f}")
        if cos > 0.999:
            print("  ✓ B8 PASS — single-chip MoE ttnn output matches HF reference")
        elif cos > 0.99:
            print(f"  ⚠ cos {cos:.4f} above 0.99 but below 0.999 — investigate per-expert")
        else:
            print(f"  ✗ FAIL — investigate")

    finally:
        ttnn.close_device(device)

    print("\nB8 DONE.")


if __name__ == "__main__":
    main()
