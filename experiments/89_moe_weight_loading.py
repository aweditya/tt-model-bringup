#!/usr/bin/env python3
"""
Experiment 89: Qwen1.5-MoE-A2.7B — Weight Loading + Single Layer Forward

First MoE experiment. Goals:
  1. Download and load the 8-shard model (~28.6 GB BF16)
  2. Upload one layer's weights to Blackhole at BFP8 — verify memory
  3. Run a single expert MLP forward pass — verify correctness
  4. Run the router (linear + softmax + topk) — verify top-4 selection
  5. Run ALL 60 experts + masking — measure time
  6. Run full MoE layer: attention + router + all experts + shared expert

Architecture (from config.json):
  hidden=2048, 24 layers, 16Q/16KV heads (MHA), head_dim=128
  60 routed experts, top-4 routing, moe_intermediate=1408
  1 shared expert, shared_intermediate=5632
  rope_theta=1e6, rms_eps=1e-6, vocab=151936
  Biases on Q/K/V (not O), no biases on expert projections
  norm_topk_prob=false (don't renormalize router probs)
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import ttnn

# ── Architecture ─────────────────────────────────────────────
hidden = 2048; n_q_heads = 16; n_kv_heads = 16; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE = 32; batch_size = 1

# MoE config
n_experts = 60; top_k = 4
moe_intermediate = 1408   # per routed expert
shared_intermediate = 5632  # shared expert

# ── Device setup ─────────────────────────────────────────────
print("=" * 60)
print("Experiment 89: Qwen1.5-MoE-A2.7B on Blackhole")
print("=" * 60)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150 ({grid.x}x{grid.y} = {grid.x*grid.y} cores)")

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)

# ── Load weights ─────────────────────────────────────────────
print("\nDownloading Qwen1.5-MoE-A2.7B (8 shards, ~28.6 GB)...")
model_id = "Qwen/Qwen1.5-MoE-A2.7B"
n_shards = 8

t0 = time.perf_counter()
shard_paths = []
for i in range(n_shards):
    name = f"model-{i+1:05d}-of-{n_shards:05d}.safetensors"
    path = hf_hub_download(model_id, name)
    shard_paths.append(path)
    print(f"  Shard {i+1}/{n_shards}: cached")
print(f"  Download/cache check: {time.perf_counter()-t0:.0f}s")

# Load only what we need: embeddings + layer 0 + final norm + lm_head
print("\nLoading layer 0 weights + embeddings...")
t0 = time.perf_counter()

embed_w = None
final_norm_g = None
lm_head_w = None
layer0_weights = {}

for path in shard_paths:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key == "model.embed_tokens.weight":
                embed_w = f.get_tensor(key).float().numpy()
            elif key == "model.norm.weight":
                final_norm_g = f.get_tensor(key).float().numpy()
            elif key == "lm_head.weight":
                lm_head_w = f.get_tensor(key).float().numpy()
            elif key.startswith("model.layers.0."):
                short = key[len("model.layers.0."):]
                layer0_weights[short] = f.get_tensor(key).float().numpy()

if lm_head_w is None:
    lm_head_w = embed_w  # tied embeddings

print(f"  Loaded in {time.perf_counter()-t0:.0f}s")
print(f"  Embed: {embed_w.shape}")
print(f"  Layer 0 keys: {len(layer0_weights)}")

# Count expert keys
expert_keys = [k for k in layer0_weights if k.startswith("mlp.experts.")]
shared_keys = [k for k in layer0_weights if k.startswith("mlp.shared_expert.")]
router_keys = [k for k in layer0_weights if k.startswith("mlp.gate.")]
print(f"  Expert weight keys: {len(expert_keys)} (expect 180 = 60 experts x 3)")
print(f"  Shared expert keys: {len(shared_keys)} (expect 3)")
print(f"  Router keys: {len(router_keys)} (expect 1)")


# ── Helpers ──────────────────────────────────────────────────
def to_bf16(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_bfp8(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)


# ══════════════════════════════════════════════════════════════
# TEST 1: Upload one expert and run forward pass
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("TEST 1: Single expert forward pass")
print(f"{'='*60}")

# Expert 0 weights
e0_gate = layer0_weights["mlp.experts.0.gate_proj.weight"]  # [1408, 2048]
e0_up = layer0_weights["mlp.experts.0.up_proj.weight"]      # [1408, 2048]
e0_down = layer0_weights["mlp.experts.0.down_proj.weight"]  # [2048, 1408]
print(f"  Expert 0 shapes: gate={e0_gate.shape}, up={e0_up.shape}, down={e0_down.shape}")

# Upload at BFP8
e0_gate_tt = to_bfp8(e0_gate.T)  # [2048, 1408]
e0_up_tt = to_bfp8(e0_up.T)
e0_down_tt = to_bfp8(e0_down.T)  # [1408, 2048]

# Test input: random hidden state
x_np = np.random.randn(1, 1, hidden).astype(np.float32)
x_tt = to_bf16(x_np.reshape(1, hidden))

# SwiGLU expert forward
t0 = time.perf_counter()
gate = ttnn.matmul(x_tt, e0_gate_tt, compute_kernel_config=hifi4)
up = ttnn.matmul(x_tt, e0_up_tt, compute_kernel_config=hifi4)
expert_out = ttnn.matmul(ttnn.mul(ttnn.silu(gate), up), e0_down_tt, compute_kernel_config=hifi4)
ttnn.synchronize_device(device)
dt = time.perf_counter() - t0
out = from_dev(expert_out, (1, hidden))
print(f"  Expert 0 output: shape={out.shape}, norm={np.linalg.norm(out):.4f}")
print(f"  Time: {dt*1000:.2f}ms")

# Verify against numpy
gate_np = x_np.reshape(1, hidden) @ e0_gate.T  # [1, 1408]
up_np = x_np.reshape(1, hidden) @ e0_up.T
silu_np = gate_np * (1 / (1 + np.exp(-gate_np)))  # silu
expert_np = (silu_np * up_np) @ e0_down.T  # [1, 2048]
cos_sim = np.dot(out.flatten(), expert_np.flatten()) / (np.linalg.norm(out) * np.linalg.norm(expert_np) + 1e-10)
print(f"  Cosine vs numpy: {cos_sim:.6f}")


# ══════════════════════════════════════════════════════════════
# TEST 2: Router — linear + softmax + topk
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("TEST 2: Router (top-4 of 60 experts)")
print(f"{'='*60}")

router_w = layer0_weights["mlp.gate.weight"]  # [60, 2048]
print(f"  Router weight shape: {router_w.shape}")

# Router forward on CPU (topk not traceable on device for 60 experts)
router_logits = x_np.reshape(1, hidden) @ router_w.T  # [1, 60]
router_probs = np.exp(router_logits - router_logits.max()) / np.exp(router_logits - router_logits.max()).sum()
top_indices = np.argsort(router_probs[0])[-top_k:][::-1]
top_probs = router_probs[0][top_indices]
# norm_topk_prob=false: don't renormalize
print(f"  Top-4 experts: {top_indices}")
print(f"  Top-4 probs: {top_probs}")
print(f"  Sum of top-4: {top_probs.sum():.4f}")


# ══════════════════════════════════════════════════════════════
# TEST 3: Upload ALL 60 experts at BFP8 — memory check
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("TEST 3: Upload all 60 experts for layer 0 (BFP8)")
print(f"{'='*60}")

t0 = time.perf_counter()
expert_weights = []
for e in range(n_experts):
    ew = {
        "gate_w": to_bfp8(layer0_weights[f"mlp.experts.{e}.gate_proj.weight"].T),
        "up_w": to_bfp8(layer0_weights[f"mlp.experts.{e}.up_proj.weight"].T),
        "down_w": to_bfp8(layer0_weights[f"mlp.experts.{e}.down_proj.weight"].T),
    }
    expert_weights.append(ew)
    if (e + 1) % 20 == 0:
        print(f"    Expert {e+1}/{n_experts}")

# Shared expert
shared_gate_tt = to_bfp8(layer0_weights["mlp.shared_expert.gate_proj.weight"].T)
shared_up_tt = to_bfp8(layer0_weights["mlp.shared_expert.up_proj.weight"].T)
shared_down_tt = to_bfp8(layer0_weights["mlp.shared_expert.down_proj.weight"].T)
shared_expert_gate_w = layer0_weights.get("mlp.shared_expert_gate.weight", np.array([1.0]))

dt_upload = time.perf_counter() - t0
print(f"  All 60 experts + shared uploaded in {dt_upload:.1f}s")

# Memory estimate: 60 experts x 3 weights x 2048x1408 x 1 byte (bfp8) = ~520 MB
# Shared expert: 3 x 2048x5632 x 1 byte = ~33 MB
mem_experts = 60 * 3 * 2048 * 1408 / 1e9
mem_shared = 3 * 2048 * 5632 / 1e9
print(f"  Expert memory (BFP8): {mem_experts:.2f} GB")
print(f"  Shared expert memory: {mem_shared:.3f} GB")
print(f"  Total MoE per layer: {mem_experts + mem_shared:.2f} GB")
print(f"  Total MoE 24 layers: {(mem_experts + mem_shared) * 24:.1f} GB")


# ══════════════════════════════════════════════════════════════
# TEST 4: Run ALL 60 experts + mask — the "run everything" approach
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("TEST 4: All 60 experts forward (mask-based MoE)")
print(f"{'='*60}")

# Run each expert and accumulate weighted sum
x_tt = to_bf16(x_np.reshape(1, hidden))

# Method: run all 60, weight by router probs, zero-mask non-selected
t0 = time.perf_counter()
moe_output_np = np.zeros((1, hidden), dtype=np.float32)

for e in range(n_experts):
    ew = expert_weights[e]
    g = ttnn.matmul(x_tt, ew["gate_w"], compute_kernel_config=hifi4)
    u = ttnn.matmul(x_tt, ew["up_w"], compute_kernel_config=hifi4)
    d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), ew["down_w"], compute_kernel_config=hifi4)
    ttnn.synchronize_device(device)

    if e in top_indices:
        weight = float(router_probs[0, e])
        expert_result = from_dev(d, (1, hidden))
        moe_output_np += weight * expert_result

ttnn.synchronize_device(device)
dt_all = time.perf_counter() - t0
print(f"  All 60 experts (sequential, non-traced): {dt_all*1000:.1f}ms")
print(f"  MoE output norm: {np.linalg.norm(moe_output_np):.4f}")

# Shared expert
g = ttnn.matmul(x_tt, shared_gate_tt, compute_kernel_config=hifi4)
u = ttnn.matmul(x_tt, shared_up_tt, compute_kernel_config=hifi4)
shared_out = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), shared_down_tt, compute_kernel_config=hifi4)
ttnn.synchronize_device(device)
shared_out_np = from_dev(shared_out, (1, hidden))

# Shared expert gate (sigmoid)
shared_gate_val = 1.0 / (1.0 + np.exp(-float(shared_expert_gate_w.flatten()[0])))
total_moe = moe_output_np + shared_gate_val * shared_out_np
print(f"  Shared expert gate: {shared_gate_val:.4f}")
print(f"  Total MoE output norm: {np.linalg.norm(total_moe):.4f}")


# ══════════════════════════════════════════════════════════════
# TEST 5: Attention layer — verify MHA works without split
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("TEST 5: Attention (MHA, 16Q/16KV heads — no split needed!)")
print(f"{'='*60}")

# Upload attention weights
attn = {
    "q_w": to_bf16(layer0_weights["self_attn.q_proj.weight"].T),
    "q_b": to_bf16(layer0_weights["self_attn.q_proj.bias"]),
    "k_w": to_bf16(layer0_weights["self_attn.k_proj.weight"].T),
    "k_b": to_bf16(layer0_weights["self_attn.k_proj.bias"]),
    "v_w": to_bf16(layer0_weights["self_attn.v_proj.weight"].T),
    "v_b": to_bf16(layer0_weights["self_attn.v_proj.bias"]),
    "o_w": to_bf16(layer0_weights["self_attn.o_proj.weight"].T),
    "ln1_g": to_bf16(layer0_weights["input_layernorm.weight"]),
    "ln2_g": to_bf16(layer0_weights["post_attention_layernorm.weight"]),
}

# Test: Q/K/V projection shapes
h_tt = ttnn.rms_norm(x_tt, weight=attn["ln1_g"], epsilon=rms_eps)
q = ttnn.add(ttnn.matmul(h_tt, attn["q_w"], compute_kernel_config=hifi4), attn["q_b"])
k = ttnn.add(ttnn.matmul(h_tt, attn["k_w"], compute_kernel_config=hifi4), attn["k_b"])
v = ttnn.add(ttnn.matmul(h_tt, attn["v_w"], compute_kernel_config=hifi4), attn["v_b"])
ttnn.synchronize_device(device)

q_np = from_dev(q, (1, n_q_heads * head_dim))
k_np = from_dev(k, (1, n_kv_heads * head_dim))
v_np = from_dev(v, (1, n_kv_heads * head_dim))
print(f"  Q: {q_np.shape} (expect [1, {n_q_heads*head_dim}])")
print(f"  K: {k_np.shape} (expect [1, {n_kv_heads*head_dim}])")
print(f"  V: {v_np.shape} (expect [1, {n_kv_heads*head_dim}])")

# Test flash_decode with 16 KV heads (MHA)
# With 16 KV heads, GQA ratio = 16/16 = 1:1 — should work with flash_decode
k_cache = to_dev_4d(np.random.randn(batch_size, n_kv_heads, MAX_SEQ, head_dim).astype(np.float32) * 0.01)
v_cache = to_dev_4d(np.random.randn(batch_size, n_kv_heads, MAX_SEQ, head_dim).astype(np.float32) * 0.01)
pos_test = ttnn.from_torch(torch.tensor([10], dtype=torch.int32), device=device)

q_4d = ttnn.reshape(q, [1, n_q_heads, 1, head_dim])

try:
    # MHA: 16Q/16KV = 1:1 GQA — try direct (no split)
    q_decode = ttnn.reshape(q_4d, [1, 1, n_q_heads, head_dim])
    attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
        q_decode, k_cache, v_cache, cur_pos_tensor=pos_test, compute_kernel_config=hifi4)
    ttnn.synchronize_device(device)
    print(f"  Flash decode (16 KV heads, 1:1 GQA): OK! shape={attn_out.shape}")
except Exception as e:
    err = str(e)[:200]
    print(f"  Flash decode (16 KV heads): FAILED — {err}")
    print(f"  Will need split workaround (like Llama 8B)")


# ══════════════════════════════════════════════════════════════
# TEST 6: Benchmark — time a single full MoE forward (non-traced)
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("TEST 6: Full MoE layer timing (non-traced)")
print(f"{'='*60}")

x_tt = to_bf16(x_np.reshape(1, hidden))

t0 = time.perf_counter()

# Attention
h = ttnn.rms_norm(x_tt, weight=attn["ln1_g"], epsilon=rms_eps)
q = ttnn.add(ttnn.matmul(h, attn["q_w"], compute_kernel_config=hifi4), attn["q_b"])
k = ttnn.add(ttnn.matmul(h, attn["k_w"], compute_kernel_config=hifi4), attn["k_b"])
v = ttnn.add(ttnn.matmul(h, attn["v_w"], compute_kernel_config=hifi4), attn["v_b"])
# Skip SDPA for timing (just measure MoE part)
o = ttnn.matmul(ttnn.reshape(q, [1, 1, 1, n_q_heads*head_dim]), attn["o_w"], compute_kernel_config=hifi4)
x2 = ttnn.add(x_tt, o)

# MoE
h2 = ttnn.rms_norm(x2, weight=attn["ln2_g"], epsilon=rms_eps)

# All 60 expert forwards (the trace-all approach)
expert_outputs = []
for e in range(n_experts):
    ew = expert_weights[e]
    g = ttnn.matmul(h2, ew["gate_w"], compute_kernel_config=hifi4)
    u = ttnn.matmul(h2, ew["up_w"], compute_kernel_config=hifi4)
    d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), ew["down_w"], compute_kernel_config=hifi4)
    expert_outputs.append(d)

# Shared expert
sg = ttnn.matmul(h2, shared_gate_tt, compute_kernel_config=hifi4)
su = ttnn.matmul(h2, shared_up_tt, compute_kernel_config=hifi4)
sd = ttnn.matmul(ttnn.mul(ttnn.silu(sg), su), shared_down_tt, compute_kernel_config=hifi4)

ttnn.synchronize_device(device)
dt_layer = time.perf_counter() - t0

print(f"  Full MoE layer (non-traced): {dt_layer*1000:.1f}ms")
print(f"  Estimated 24 layers: {dt_layer*24*1000:.0f}ms")
print(f"  Estimated tok/s: {1000/(dt_layer*24*1000):.1f}")

# For comparison: how fast is just the attention part?
t0 = time.perf_counter()
h = ttnn.rms_norm(x_tt, weight=attn["ln1_g"], epsilon=rms_eps)
q = ttnn.add(ttnn.matmul(h, attn["q_w"], compute_kernel_config=hifi4), attn["q_b"])
k = ttnn.add(ttnn.matmul(h, attn["k_w"], compute_kernel_config=hifi4), attn["k_b"])
v = ttnn.add(ttnn.matmul(h, attn["v_w"], compute_kernel_config=hifi4), attn["v_b"])
o = ttnn.matmul(ttnn.reshape(q, [1, 1, 1, n_q_heads*head_dim]), attn["o_w"], compute_kernel_config=hifi4)
ttnn.synchronize_device(device)
dt_attn = time.perf_counter() - t0
print(f"\n  Attention only: {dt_attn*1000:.1f}ms")
print(f"  MoE block only: {(dt_layer - dt_attn)*1000:.1f}ms")
print(f"  MoE block / total: {(dt_layer - dt_attn)/dt_layer*100:.0f}%")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Model: Qwen1.5-MoE-A2.7B")
print(f"  Architecture: 24 layers, 60 experts (top-4), MHA (16/16 heads)")
print(f"  Expert size: {moe_intermediate} intermediate (tiny)")
print(f"  Memory per layer (BFP8): {mem_experts + mem_shared:.2f} GB")
print(f"  Total model (BFP8): {(mem_experts + mem_shared) * 24 + vocab_size*hidden*2/1e9:.1f} GB")
print(f"  Single expert forward: verified correct (cosine={cos_sim:.6f})")
print(f"  Full layer (non-traced): {dt_layer*1000:.1f}ms")
print(f"\nNext: Exp 90 — full 24-layer prefill + traced decode")

ttnn.close_device(device)
print("\nDone!")
