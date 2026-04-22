#!/usr/bin/env python3
"""
Experiment 57: bfloat8_b weight quantization for Qwen2.5-0.5B.

Hypothesis: bfloat8_b weights give ~10% speedup with acceptable precision.
From wiki 39: bf8 is 10% faster at model-relevant sizes (41.6 vs 38.3 TFLOPS).

Key question: what is the cosine similarity and generation quality when we
quantize ALL weight matrices to bfloat8_b but keep activations in bfloat16?

Tests:
  A: Per-layer cosine with bf8 weights vs bf16 reference
  B: Full-model logit cosine
  C: Generation quality comparison
  D: Speed comparison (traced decode bf8 vs bf16)
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="The capital of France is")
parser.add_argument("--tokens", type=int, default=80)
args = parser.parse_args()

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

# ── Load model ──
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


# ── Upload helpers ──
def to_dev_bf16(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_bf8(arr):
    """Upload weight as bfloat8_b."""
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
# TEST A: Upload weights in both formats and compare
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST A: Weight quantization impact per operation")
print("=" * 60)

lw = layer_weights_np[0]

# Test a single matmul: x @ q_w
x_np = np.random.randn(1, hidden).astype(np.float32)
x_tt = to_dev_bf16(x_np)

# bf16 weight
q_w_bf16 = to_dev_bf16(lw["self_attn.q_proj.weight"].T)
out_bf16 = from_dev(ttnn.matmul(x_tt, q_w_bf16, compute_kernel_config=hifi4), (1, n_q_heads * head_dim))

# bf8 weight
q_w_bf8 = to_dev_bf8(lw["self_attn.q_proj.weight"].T)
out_bf8 = from_dev(ttnn.matmul(x_tt, q_w_bf8, compute_kernel_config=hifi4), (1, n_q_heads * head_dim))

# numpy reference
out_ref = (x_np @ lw["self_attn.q_proj.weight"].T)

cos_bf16_ref = np.dot(out_bf16.flatten(), out_ref.flatten()) / (np.linalg.norm(out_bf16) * np.linalg.norm(out_ref))
cos_bf8_ref = np.dot(out_bf8.flatten(), out_ref.flatten()) / (np.linalg.norm(out_bf8) * np.linalg.norm(out_ref))
cos_bf8_bf16 = np.dot(out_bf8.flatten(), out_bf16.flatten()) / (np.linalg.norm(out_bf8) * np.linalg.norm(out_bf16))

print(f"  Q projection (896 → 896):")
print(f"    bf16 vs fp32 ref: {cos_bf16_ref:.6f}")
print(f"    bf8  vs fp32 ref: {cos_bf8_ref:.6f}")
print(f"    bf8  vs bf16:     {cos_bf8_bf16:.6f}")

# Test gate projection (896 → 4864, the hottest matmul)
gate_w_bf16 = to_dev_bf16(lw["mlp.gate_proj.weight"].T)
gate_w_bf8 = to_dev_bf8(lw["mlp.gate_proj.weight"].T)
out_gate_bf16 = from_dev(ttnn.matmul(x_tt, gate_w_bf16, compute_kernel_config=hifi4), (1, 4864))
out_gate_bf8 = from_dev(ttnn.matmul(x_tt, gate_w_bf8, compute_kernel_config=hifi4), (1, 4864))
out_gate_ref = x_np @ lw["mlp.gate_proj.weight"].T

cos_g16 = np.dot(out_gate_bf16.flatten(), out_gate_ref.flatten()) / (np.linalg.norm(out_gate_bf16) * np.linalg.norm(out_gate_ref))
cos_g8 = np.dot(out_gate_bf8.flatten(), out_gate_ref.flatten()) / (np.linalg.norm(out_gate_bf8) * np.linalg.norm(out_gate_ref))

print(f"  Gate projection (896 → 4864):")
print(f"    bf16 vs fp32 ref: {cos_g16:.6f}")
print(f"    bf8  vs fp32 ref: {cos_g8:.6f}")

q_w_bf16.deallocate(); q_w_bf8.deallocate()
gate_w_bf16.deallocate(); gate_w_bf8.deallocate()


# ══════════════════════════════════════════════════════════════
# TEST B: Full model with bf8 weights — single forward pass
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST B: Full model comparison — bf16 vs bf8 weights")
print("=" * 60)

R = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R[i + half_dim, i] = -1.0
    R[i, i + half_dim] = 1.0
R_tt = to_dev_bf16(R)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

def get_rope_tables_half(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))

def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]


def upload_weights(dtype_fn, label):
    """Upload all layer weights in given dtype."""
    t0 = time.perf_counter()
    dev_layers = []
    for i in range(n_layers):
        lw = layer_weights_np[i]
        # Norms and biases always bf16 (small, precision-sensitive)
        dev_layers.append({
            "ln1_g": to_dev_bf16(lw["input_layernorm.weight"]),
            "q_w": dtype_fn(lw["self_attn.q_proj.weight"].T),
            "q_b": to_dev_bf16(lw["self_attn.q_proj.bias"]),
            "k_w": dtype_fn(lw["self_attn.k_proj.weight"].T),
            "k_b": to_dev_bf16(lw["self_attn.k_proj.bias"]),
            "v_w": dtype_fn(lw["self_attn.v_proj.weight"].T),
            "v_b": to_dev_bf16(lw["self_attn.v_proj.bias"]),
            "o_w": dtype_fn(lw["self_attn.o_proj.weight"].T),
            "ln2_g": to_dev_bf16(lw["post_attention_layernorm.weight"]),
            "gate_w": dtype_fn(lw["mlp.gate_proj.weight"].T),
            "up_w": dtype_fn(lw["mlp.up_proj.weight"].T),
            "down_w": dtype_fn(lw["mlp.down_proj.weight"].T),
        })
    final_g = to_dev_bf16(final_norm_g)
    lm_head = dtype_fn(lm_head_w)
    dt = time.perf_counter() - t0
    print(f"  {label} upload: {dt*1000:.0f}ms")
    return dev_layers, final_g, lm_head


def prefill_and_get_logits(dev_layers, final_g, lm_head, token_ids):
    """Run prefill and return final logits."""
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_half(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev_bf16(x_np.reshape(B * T, hidden))
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

        q_4d = apply_rope_half_np(q_np.reshape(B, T, n_q_heads, head_dim).transpose(0,2,1,3), cos_t, sin_t)
        k_4d = apply_rope_half_np(k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3), cos_t, sin_t)
        v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d),
            is_causal=True, compute_kernel_config=hifi4)
        attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim)).transpose(0,2,1,3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev_bf16(attn_np.reshape(B*T, hidden)), dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B * T, hidden)).reshape(B, T, hidden)

    x_tt = to_dev_bf16(x_np.reshape(B * T, hidden))
    x_tt = ttnn.rms_norm(x_tt, weight=final_g, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (B * T, vocab_size))[-1]


# Run with bf16 weights
print("\n  --- bf16 weights ---")
dev_bf16, final_bf16, lm_head_bf16 = upload_weights(to_dev_bf16, "bf16")
tokens = np.array(tokenizer.encode(args.prompt))
logits_bf16 = prefill_and_get_logits(dev_bf16, final_bf16, lm_head_bf16, tokens)
next_bf16 = int(np.argmax(logits_bf16))
print(f"  Next token (bf16): {next_bf16} = '{tokenizer.decode([next_bf16])}'")

# Deallocate bf16 weights
for dl in dev_bf16:
    for v in dl.values():
        v.deallocate()
final_bf16.deallocate(); lm_head_bf16.deallocate()

# Run with bf8 weights
print("\n  --- bf8 weights ---")
dev_bf8, final_bf8, lm_head_bf8 = upload_weights(to_dev_bf8, "bf8")
logits_bf8 = prefill_and_get_logits(dev_bf8, final_bf8, lm_head_bf8, tokens)
next_bf8 = int(np.argmax(logits_bf8))
print(f"  Next token (bf8):  {next_bf8} = '{tokenizer.decode([next_bf8])}'")

# Compare
cos_logits = np.dot(logits_bf16, logits_bf8) / (np.linalg.norm(logits_bf16) * np.linalg.norm(logits_bf8))
print(f"\n  Logit cosine (bf16 vs bf8): {cos_logits:.6f}")
print(f"  Top-1 match: {'YES' if next_bf16 == next_bf8 else 'NO'}")
print(f"  bf16 top-5: {np.argsort(logits_bf16)[-5:][::-1]}")
print(f"  bf8  top-5: {np.argsort(logits_bf8)[-5:][::-1]}")

# Memory savings
bf16_bytes = sum(v.nbytes for lw in layer_weights_np for v in lw.values()) * 2  # bf16
bf8_bytes = sum(v.nbytes for lw in layer_weights_np for v in lw.values())  # bf8
print(f"\n  Memory: bf16={bf16_bytes/1e6:.1f}MB, bf8={bf8_bytes/1e6:.1f}MB (estimated)")


# ══════════════════════════════════════════════════════════════
# TEST C: Generation quality comparison
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST C: Generation quality (greedy, first 30 tokens)")
print("=" * 60)

# Generate with bf8 (already loaded)
gen_tokens_bf8 = list(tokens) + [next_bf8]
# Simple non-traced generation for quality check
for step in range(30):
    x_np = embed_w[gen_tokens_bf8[-1]:gen_tokens_bf8[-1]+1].reshape(1, 1, hidden)
    x_tt = to_dev_bf16(x_np.reshape(1, hidden))
    pos = len(gen_tokens_bf8) - 1
    cos_t, sin_t = get_rope_tables_half(pos + 1)

    for i in range(n_layers):
        dl = dev_bf8[i]
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_np = from_dev(q_tt, (1, 1, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (1, 1, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (1, 1, n_kv_heads * head_dim))

        q_4d = apply_rope_half_np(q_np.reshape(1, 1, n_q_heads, head_dim).transpose(0,2,1,3),
                                   cos_t[-1:], sin_t[-1:])
        k_4d = apply_rope_half_np(k_np.reshape(1, 1, n_kv_heads, head_dim).transpose(0,2,1,3),
                                   cos_t[-1:], sin_t[-1:])
        v_4d = v_np.reshape(1, 1, n_kv_heads, head_dim).transpose(0,2,1,3)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d),
            is_causal=True, compute_kernel_config=hifi4)
        attn_np = from_dev(attn_out_tt, (1, n_q_heads, 1, head_dim)).transpose(0,2,1,3).reshape(1, 1, hidden)

        o_tt = ttnn.matmul(to_dev_bf16(attn_np.reshape(1, hidden)), dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt2, down_tt)

    x_tt = ttnn.rms_norm(x_tt, weight=final_bf8, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_bf8, compute_kernel_config=hifi4)
    logits = from_dev(logits_tt, (1, vocab_size))[0]
    next_tok = int(np.argmax(logits))
    gen_tokens_bf8.append(next_tok)

print(f"  bf8 output: {tokenizer.decode(gen_tokens_bf8)}")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Per-op cosine (Q proj):     bf16={cos_bf16_ref:.6f}, bf8={cos_bf8_ref:.6f}")
print(f"  Per-op cosine (gate proj):  bf16={cos_g16:.6f}, bf8={cos_g8:.6f}")
print(f"  Full-model logit cosine:    {cos_logits:.6f}")
print(f"  Top-1 token match:          {'YES' if next_bf16 == next_bf8 else 'NO'}")
print(f"  Expected speedup:           ~10% (from bf8 TFLOPS advantage)")

ttnn.close_device(device)
print("\nDone!")
