#!/usr/bin/env python3
"""
Experiment 46c: Is fp32_dest_acc_en broken after N calls on Blackhole?

Hypothesis: fp32_dest_acc_en causes state corruption in SDPA after ~3 calls.
Test: Call SDPA N times on the SAME inputs, check if output quality degrades.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn

def cosine(a, b):
    return np.dot(a.flatten(), b.flatten()) / (
        np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-8)

device = ttnn.open_device(device_id=0)

# Create random Q/K/V in the Qwen shape
np.random.seed(42)
B, n_q, n_kv, T, hd = 1, 14, 2, 5, 64
q = np.random.randn(B, n_q, T, hd).astype(np.float32) * 0.1
k = np.random.randn(B, n_kv, T, hd).astype(np.float32) * 0.1
v = np.random.randn(B, n_kv, T, hd).astype(np.float32) * 0.1

# Numpy reference
kv_r = n_q // n_kv
k_exp = np.repeat(k, kv_r, axis=1); v_exp = np.repeat(v, kv_r, axis=1)
sc = (q @ k_exp.transpose(0,1,3,2)) / np.sqrt(hd)
sc += np.triu(np.ones((T,T))*-1e9, k=1)[None,None]
e = np.exp(sc - np.max(sc, axis=-1, keepdims=True))
ref_out = (e / np.sum(e, axis=-1, keepdims=True)) @ v_exp

def to_dev(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(t, shape):
    return ttnn.to_torch(t).float().reshape(shape).numpy()

configs = {
    "default": None,
    "fp32_acc": ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=True),
    "HiFi4+fp32": ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False),
}

for name, cfg in configs.items():
    print(f"\n── {name}: 10 repeated SDPA calls on same data ──")
    for call_idx in range(10):
        q_tt = to_dev(q); k_tt = to_dev(k); v_tt = to_dev(v)
        kwargs = {"is_causal": True}
        if cfg is not None:
            kwargs["compute_kernel_config"] = cfg
        out_tt = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, **kwargs)
        out_np = from_dev(out_tt, ref_out.shape)
        cos = cosine(out_np, ref_out)
        flag = " <<< DEGRADED" if cos < 0.98 else ""
        print(f"  Call {call_idx}: cosine={cos:.6f}{flag}")

# Also test: does a matmul with fp32_acc break things?
print(f"\n── fp32_acc matmul test: 10 repeated calls ──")
a = np.random.randn(1, 5, 896).astype(np.float32) * 0.1
w = np.random.randn(896, 896).astype(np.float32) * 0.01
ref_mm = a @ w

cfg_mm = ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=True)
for call_idx in range(10):
    a_tt = to_dev(a); w_tt = to_dev(w)
    out_tt = ttnn.matmul(a_tt, w_tt, compute_kernel_config=cfg_mm)
    out_np = from_dev(out_tt, ref_mm.shape)
    cos = cosine(out_np, ref_mm)
    flag = " <<< DEGRADED" if cos < 0.98 else ""
    print(f"  Call {call_idx}: cosine={cos:.6f}{flag}")

ttnn.close_device(device)
print("\nDone!")
