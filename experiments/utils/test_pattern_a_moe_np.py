#!/usr/bin/env python3
"""Pattern A MoE numpy reference + correctness gate.

Goal: verify that the "run all 256 experts, mask by top-k" algorithm
(Mixtral/Grok style) produces the SAME output as the current "select
top-8, loop over selected" algorithm. They are mathematically
equivalent — this is a sanity check before refactoring the TT MoE.

The test:
  1. Load one MoE layer's weights from the Qwen3.6-35B-A3B safetensors.
  2. Generate a random input vector (single token).
  3. Run both algorithms.
  4. Assert cos(top8_out, pattern_a_out) > 0.99999.
  5. Assert ||top8 - pattern_a|| / ||top8|| < 1e-5.

Run on qb1 (CPU, no device needed):
  ssh qb1 'cd ~/tt-xla && .venv/bin/python -u experiments/utils/test_pattern_a_moe_np.py'
"""
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b as srv  # noqa: E402 — for HIDDEN, NUM_EXPERTS, TOP_K, MOE_INTER, silu, sigmoid

HIDDEN = srv.HIDDEN
NUM_EXPERTS = srv.NUM_EXPERTS
TOP_K = srv.TOP_K
MOE_INTER = srv.MOE_INTER


def load_layer_weights(layer_idx):
    """Pull one decoder layer's MoE-relevant weights from HF safetensors."""
    import json
    idx_path = hf_hub_download("Qwen/Qwen3.6-35B-A3B", "model.safetensors.index.json")
    idx = json.loads(Path(idx_path).read_text())

    keys = {
        "router":      f"model.language_model.layers.{layer_idx}.mlp.gate.weight",
        "experts_gu":  f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj",
        "experts_d":   f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj",
        "shared_g":    f"model.language_model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight",
        "shared_u":    f"model.language_model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight",
        "shared_d":    f"model.language_model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight",
        "shared_gate": f"model.language_model.layers.{layer_idx}.mlp.shared_expert_gate.weight",
    }
    sd = {}
    for name, k in keys.items():
        shard = hf_hub_download("Qwen/Qwen3.6-35B-A3B", idx["weight_map"][k])
        with safe_open(shard, framework="pt", device="cpu") as f:
            sd[name] = f.get_tensor(k).float().numpy()
    return sd


def moe_topk_np(h_np, w):
    """Current numpy MoE — Python loop over top-8 selected experts.
    Same algorithm as server_35b.moe_layer_forward, just inlined for clarity."""
    router_w = w["router"]
    eg = w["experts_gu"]
    ed = w["experts_d"]

    logits = h_np @ router_w.T
    lf = logits.astype(np.float64); lf -= lf.max()
    probs = (np.exp(lf) / np.exp(lf).sum(axis=-1, keepdims=True)).astype(np.float32)
    top_k_idxs = np.argsort(probs[0])[-TOP_K:][::-1].copy()
    top_k_vals = probs[0, top_k_idxs].copy()
    weights = top_k_vals / top_k_vals.sum()

    routed = np.zeros((1, HIDDEN), dtype=np.float32)
    for k_idx in range(TOP_K):
        e = int(top_k_idxs[k_idx])
        gate_up = h_np @ eg[e].T  # eg[e] shape: [2*MOE_INTER, HIDDEN]; eg[e].T: [HIDDEN, 2*MOE_INTER]
        gate = gate_up[:, :MOE_INTER]; up = gate_up[:, MOE_INTER:]
        mid = srv.silu(gate) * up
        out_e = mid @ ed[e].T  # ed[e] shape: [HIDDEN, MOE_INTER]; ed[e].T: [MOE_INTER, HIDDEN]
        routed += float(weights[k_idx]) * out_e

    # Shared expert
    s_gate = h_np @ w["shared_g"].T
    s_up = h_np @ w["shared_u"].T
    s_mid = srv.silu(s_gate) * s_up
    shared = s_mid @ w["shared_d"].T
    g_scalar = srv.sigmoid(h_np @ w["shared_gate"].T)
    shared *= g_scalar

    return routed + shared, top_k_idxs, weights


def moe_pattern_a_np(h_np, w):
    """Pattern A numpy MoE — run ALL 256 experts, mask by top-k routing.
    Algorithmically equivalent to moe_topk_np; verified by the test below."""
    router_w = w["router"]
    eg = w["experts_gu"]
    ed = w["experts_d"]

    # Router unchanged
    logits = h_np @ router_w.T
    lf = logits.astype(np.float64); lf -= lf.max()
    probs = (np.exp(lf) / np.exp(lf).sum(axis=-1, keepdims=True)).astype(np.float32)
    top_k_idxs = np.argsort(probs[0])[-TOP_K:][::-1].copy()  # [TOP_K]
    top_k_vals = probs[0, top_k_idxs].copy()                  # [TOP_K]
    weights = top_k_vals / top_k_vals.sum()                   # [TOP_K]

    # Build routing weight per expert: 0 for non-selected experts, weight for selected.
    # expert_id_table: [NUM_EXPERTS] = [0, 1, ..., 255]
    # top_k_idxs: [TOP_K]
    # mask[e, k] = 1 if expert_id_table[e] == top_k_idxs[k] else 0
    # routing_weight[e] = sum_k(mask[e, k] * weights[k])
    expert_id_table = np.arange(NUM_EXPERTS, dtype=np.int64)
    mask = (expert_id_table[:, None] == top_k_idxs[None, :]).astype(np.float32)  # [NUM_EXPERTS, TOP_K]
    routing_weight = (mask * weights[None, :]).sum(axis=-1)  # [NUM_EXPERTS]

    # Run ALL experts (most produce wasted compute since routing_weight is 0 for them)
    routed = np.zeros((1, HIDDEN), dtype=np.float32)
    for e in range(NUM_EXPERTS):
        gate_up = h_np @ eg[e].T
        gate = gate_up[:, :MOE_INTER]; up = gate_up[:, MOE_INTER:]
        mid = srv.silu(gate) * up
        out_e = mid @ ed[e].T
        routed += float(routing_weight[e]) * out_e

    # Shared expert — unchanged
    s_gate = h_np @ w["shared_g"].T
    s_up = h_np @ w["shared_u"].T
    s_mid = srv.silu(s_gate) * s_up
    shared = s_mid @ w["shared_d"].T
    g_scalar = srv.sigmoid(h_np @ w["shared_gate"].T)
    shared *= g_scalar

    return routed + shared, top_k_idxs, weights, routing_weight


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    layer_idx = 3  # First AT layer; MoE structure is the same on all 40 layers
    print(f"Loading weights for layer {layer_idx} …", flush=True)
    w = load_layer_weights(layer_idx)
    print(f"  router      {w['router'].shape}")
    print(f"  experts_gu  {w['experts_gu'].shape}")
    print(f"  experts_d   {w['experts_d'].shape}")
    print(f"  shared_g    {w['shared_g'].shape}")

    rng = np.random.default_rng(0)
    # Magnitudes consistent with real intermediate activations at L=3 (post input_layernorm)
    h = rng.normal(0, 50.0, size=(1, HIDDEN)).astype(np.float32)

    print()
    print("Running top-8 reference …", flush=True)
    out_topk, idxs, weights = moe_topk_np(h, w)
    print(f"  top_idxs: {idxs.tolist()}")
    print(f"  weights:  {weights.round(4).tolist()}")
    print(f"  out_topk:    |.|={np.linalg.norm(out_topk):.4f}  max|.|={np.abs(out_topk).max():.4f}")

    print()
    print("Running Pattern A (run all 256 experts, mask) …", flush=True)
    out_pa, idxs_pa, weights_pa, routing_w = moe_pattern_a_np(h, w)
    print(f"  routing_weight nonzero count: {(routing_w != 0).sum()} (expect {TOP_K})")
    print(f"  routing_weight sum: {routing_w.sum():.6f} (expect 1.0)")
    print(f"  out_pa:      |.|={np.linalg.norm(out_pa):.4f}  max|.|={np.abs(out_pa).max():.4f}")

    print()
    print("Comparing outputs …")
    c = cos(out_topk, out_pa)
    abs_err = np.abs(out_topk - out_pa).max()
    rel_err = np.linalg.norm(out_topk - out_pa) / np.linalg.norm(out_topk)
    print(f"  cos(top8, pattern_a):    {c:.10f}")
    print(f"  max |Δ|:                 {abs_err:.6e}")
    print(f"  rel ||Δ|| / ||top8||:    {rel_err:.6e}")

    assert idxs.tolist() == idxs_pa.tolist(), "top_idxs disagree"
    assert np.allclose(weights, weights_pa, atol=1e-6), "weights disagree"
    assert c > 0.99999, f"cos {c} below threshold 0.99999"
    assert rel_err < 1e-5, f"rel_err {rel_err} above threshold 1e-5"
    print()
    print("PASS ✓  Pattern A and top-8 are numerically equivalent.")


if __name__ == "__main__":
    main()
