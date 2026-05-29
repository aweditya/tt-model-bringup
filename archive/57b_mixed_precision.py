#!/usr/bin/env python3
"""
Experiment 57b: Mixed precision — bf8 MLP weights + bf16 attention weights.

From 57: full bf8 degrades at token 3 despite 0.999 per-op cosine.
From 46: SDPA is precision-sensitive (bf16 softmax was the sole error source).

Hypothesis: The attention path (Q/K/V projections → SDPA → O projection) is
precision-sensitive due to softmax. MLP weights (gate/up/down) should be safe
at bf8 since they don't involve attention.

Test configurations:
  A: All bf16 (baseline)
  B: All bf8 (known to fail)
  C: bf8 MLP + bf16 attention (hypothesis: works)
  D: bf16 MLP + bf8 attention (hypothesis: fails)
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE_SIZE = 32; batch_size = 1

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# Load model
print("Loading Qwen2.5-0.5B...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights["lm_head.weight"].T if "lm_head.weight" in all_weights else embed_w.T.copy()

layer_weights_np = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)
del all_weights

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

def to_bf16(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_bf8(arr):
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


# RoPE
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)
def get_rope_tables_half(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))
def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]


configs = {
    "A_all_bf16":    {"attn": to_bf16, "mlp": to_bf16, "lm_head": to_bf16},
    "C_mixed_bf8mlp": {"attn": to_bf16, "mlp": to_bf8,  "lm_head": to_bf16},
    "D_mixed_bf8attn": {"attn": to_bf8,  "mlp": to_bf16, "lm_head": to_bf16},
    "B_all_bf8":     {"attn": to_bf8,  "mlp": to_bf8,  "lm_head": to_bf8},
}


def upload_config(cfg):
    attn_fn, mlp_fn, lm_fn = cfg["attn"], cfg["mlp"], cfg["lm_head"]
    dev_layers = []
    for i in range(n_layers):
        lw = layer_weights_np[i]
        dev_layers.append({
            "ln1_g": to_bf16(lw["input_layernorm.weight"]),
            "q_w": attn_fn(lw["self_attn.q_proj.weight"].T),
            "q_b": to_bf16(lw["self_attn.q_proj.bias"]),
            "k_w": attn_fn(lw["self_attn.k_proj.weight"].T),
            "k_b": to_bf16(lw["self_attn.k_proj.bias"]),
            "v_w": attn_fn(lw["self_attn.v_proj.weight"].T),
            "v_b": to_bf16(lw["self_attn.v_proj.bias"]),
            "o_w": attn_fn(lw["self_attn.o_proj.weight"].T),
            "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
            "gate_w": mlp_fn(lw["mlp.gate_proj.weight"].T),
            "up_w": mlp_fn(lw["mlp.up_proj.weight"].T),
            "down_w": mlp_fn(lw["mlp.down_proj.weight"].T),
        })
    return dev_layers, to_bf16(final_norm_g), lm_fn(lm_head_w)


def generate(dev_layers, final_g, lm_head, tokens, n_gen=60):
    """Simple full-recompute generation for quality check."""
    B, T_init = 1, len(tokens)
    gen = list(tokens)

    for step in range(n_gen):
        T = len(gen)
        x_np = embed_w[gen].reshape(B, T, hidden)
        cos_t, sin_t = get_rope_tables_half(T)

        for i in range(n_layers):
            dl = dev_layers[i]
            x_tt = to_bf16(x_np.reshape(B * T, hidden))
            h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
            q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
            k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
            v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

            q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
            k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
            v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

            q_4d = apply_rope_half_np(q_np.reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
            k_4d = apply_rope_half_np(k_np.reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
            v_4d = v_np.reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)

            attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
                to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d),
                is_causal=True, compute_kernel_config=hifi4)
            attn_np = from_dev(attn_out_tt, (B,n_q_heads,T,head_dim)).transpose(0,2,1,3).reshape(B,T,hidden)

            o_tt = ttnn.matmul(to_bf16(attn_np.reshape(B*T,hidden)), dl["o_w"], compute_kernel_config=hifi4)
            x_tt2 = ttnn.add(x_tt, o_tt)
            h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
            gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
            up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
            swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
            down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
            out_tt = ttnn.add(x_tt2, down_tt)
            x_np = from_dev(out_tt, (B*T, hidden)).reshape(B, T, hidden)

        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        x_tt = ttnn.rms_norm(x_tt, weight=final_g, epsilon=rms_eps)
        logits_tt = ttnn.matmul(x_tt, lm_head, compute_kernel_config=hifi4)
        logits = from_dev(logits_tt, (B*T, vocab_size))[-1]
        gen.append(int(np.argmax(logits)))

    return gen


tokens = np.array(tokenizer.encode("The capital of France is"))
print(f'\nPrompt: "The capital of France is" ({len(tokens)} tokens)')
print(f"Generating 20 tokens with each config (full recompute for quality)...\n")

results = {}
for name, cfg in configs.items():
    print(f"--- {name} ---")
    t0 = time.perf_counter()
    dev_layers, final_g, lm_head = upload_config(cfg)
    dt_upload = time.perf_counter() - t0

    t0 = time.perf_counter()
    gen = generate(dev_layers, final_g, lm_head, tokens, n_gen=20)
    dt_gen = time.perf_counter() - t0

    text = tokenizer.decode(gen)
    results[name] = text
    print(f"  Upload: {dt_upload*1000:.0f}ms, Generate: {dt_gen*1000:.0f}ms")
    print(f"  Output: {text}")

    # Deallocate
    for dl in dev_layers:
        for v in dl.values():
            v.deallocate()
    final_g.deallocate(); lm_head.deallocate()
    print()


print("=" * 60)
print("SUMMARY")
print("=" * 60)
for name, text in results.items():
    quality = "GOOD" if "Paris" in text and "200000" not in text else "BAD" if "200000" in text else "OK"
    print(f"  {name:25s}: [{quality}] {text[:80]}...")


ttnn.close_device(device)
print("\nDone!")
