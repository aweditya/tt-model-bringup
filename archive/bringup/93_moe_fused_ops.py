#!/usr/bin/env python3
"""
Experiment 93: Qwen1.5-MoE-A2.7B-Chat — Fused Ops for MoE Decode

Key improvements vs exp 92 (20.5 tok/s):
  1. ttnn.linear(x, w, activation="silu") — fuse silu into gate matmul
     Saves 1 dispatch per expert × 4 experts × 24 layers = 96 dispatches
     Plus same for shared expert = +24 more = 120 saved
  2. ttnn.linear(x, w, bias=b) — fuse bias add into Q/K/V projections
     Saves 3 dispatches × 24 layers = 72 saved
  3. Total: ~192 fewer dispatches × 30μs = ~5.8ms savings

Target: ~22-23 tok/s

Run: ssh tenstorrent 'cd tt-xla && python3 experiments/93_moe_fused_ops.py'
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from collections import defaultdict
import ttnn

# ── Architecture ─────────────────────────────────────────────
hidden = 2048; n_q_heads = 16; n_kv_heads = 16; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE = 32; batch_size = 1
n_experts = 60; n_exp_pad = 64  # tile-aligned
top_k = 4

# ── Device ───────────────────────────────────────────────────
print("=" * 60)
print("Exp 93: Qwen1.5-MoE-A2.7B-Chat — Fused Ops")
print("=" * 60)
device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150 ({grid.x}x{grid.y} cores)")

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)

# ── Download model shards ────────────────────────────────────
model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
n_shards = 8
print(f"\nDownloading {model_id} ({n_shards} shards)...")
shard_paths = [hf_hub_download(model_id, f"model-{i+1:05d}-of-{n_shards:05d}.safetensors")
               for i in range(n_shards)]

key_to_path = {}
for path in shard_paths:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            key_to_path[key] = path
print(f"  {len(key_to_path)} weight tensors across {n_shards} shards")

tokenizer = AutoTokenizer.from_pretrained(model_id)

# ── Helpers ──────────────────────────────────────────────────
def load_np(key):
    with safe_open(key_to_path[key], framework="pt") as f:
        return f.get_tensor(key).float().numpy()

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

# ── RoPE ─────────────────────────────────────────────────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

def get_rope_tables(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))

def apply_rope_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

# ── Load embeddings ──────────────────────────────────────────
print("\nLoading embeddings + lm_head...")
embed_w = load_np("model.embed_tokens.weight")
final_norm_g = load_np("model.norm.weight")
lm_head_w = load_np("lm_head.weight").T if "lm_head.weight" in key_to_path else embed_w.T.copy()
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)

# ── Expert padding mask (zero out positions 60-63 after softmax) ──
# No expert mask needed: router_w is [2048, 60], so matmul output has logical width 60.
# Tile layout pads internally to 64, but softmax/topk operate on the 60 logical positions.

# ── Upload all 24 layers ─────────────────────────────────────
print(f"\nUploading {n_layers} layers (BFP8 experts, bf16 attention)...")
t0_upload = time.perf_counter()
dev_layers = []
seg_w_np_cache = []  # numpy cache for gate (fallback)

for L in range(n_layers):
    p = f"model.layers.{L}."
    layer_keys = [k for k in key_to_path if k.startswith(p)]
    by_path = defaultdict(list)
    for k in layer_keys:
        by_path[key_to_path[k]].append(k)
    lw = {}
    for path, keys in by_path.items():
        with safe_open(path, framework="pt") as f:
            for k in keys:
                lw[k[len(p):]] = f.get_tensor(k).float().numpy()

    has_o_bias = "self_attn.o_proj.bias" in lw
    dl = {
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "q_b": to_bf16(lw["self_attn.q_proj.bias"]),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "k_b": to_bf16(lw["self_attn.k_proj.bias"]),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "v_b": to_bf16(lw["self_attn.v_proj.bias"]),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "o_b": to_bf16(lw["self_attn.o_proj.bias"]) if has_o_bias else None,
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "router_w": to_bf16(lw["mlp.gate.weight"].T),  # [2048, 60]
    }

    # Shared expert (BFP8)
    dl["s_gate_w"] = to_bfp8(lw["mlp.shared_expert.gate_proj.weight"].T)
    dl["s_up_w"] = to_bfp8(lw["mlp.shared_expert.up_proj.weight"].T)
    dl["s_down_w"] = to_bfp8(lw["mlp.shared_expert.down_proj.weight"].T)

    # Shared expert gate
    seg_key = "mlp.shared_expert_gate.weight"
    if seg_key in lw:
        dl["seg_w"] = to_bf16(lw[seg_key].T)  # [2048, 1]
        seg_w_np_cache.append(lw[seg_key].T.copy())
    else:
        dl["seg_w"] = None
        seg_w_np_cache.append(None)

    # 60 experts (BFP8)
    experts = []
    for e in range(n_experts):
        experts.append({
            "g": to_bfp8(lw[f"mlp.experts.{e}.gate_proj.weight"].T),
            "u": to_bfp8(lw[f"mlp.experts.{e}.up_proj.weight"].T),
            "d": to_bfp8(lw[f"mlp.experts.{e}.down_proj.weight"].T),
        })
    dl["experts"] = experts
    dev_layers.append(dl)
    del lw

    elapsed = time.perf_counter() - t0_upload
    eta = elapsed / (L + 1) * (n_layers - L - 1)
    print(f"  Layer {L+1}/{n_layers} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

print(f"  All layers uploaded in {time.perf_counter()-t0_upload:.0f}s")

# ── KV caches ────────────────────────────────────────────────
print("Creating KV caches...")
k_caches, v_caches = [], []
for _ in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(to_dev_4d(c.copy()))
    v_caches.append(to_dev_4d(c.copy()))

kv_sh = ((n_kv_heads + TILE - 1) // TILE) * TILE
kv_cg = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_sh, head_dim), core_grid=kv_cg,
    strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)


# ── Prefill (same as exp 91) ────────────────────────────────
def prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B * T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k = ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v = ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])
        q_np = apply_rope_np(
            from_dev(q, (B, T, n_q_heads * head_dim)).reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3),
            cos_t, sin_t)
        k_np = apply_rope_np(
            from_dev(k, (B, T, n_kv_heads * head_dim)).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3),
            cos_t, sin_t)
        v_np = from_dev(v, (B, T, n_kv_heads * head_dim)).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_np), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_np), batch_index=0)
        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn, (B, n_q_heads, T, head_dim)).transpose(0, 2, 1, 3).reshape(B, T, hidden)
        o = ttnn.matmul(to_bf16(a_np.reshape(B * T, hidden)), dl["o_w"], compute_kernel_config=hifi4)
        if dl["o_b"] is not None:
            o = ttnn.add(o, dl["o_b"])
        x2 = ttnn.add(x_tt, o)

        # MoE (CPU routing for prefill — multiple tokens, not optimized)
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        h2_np = from_dev(h2, (B * T, hidden))
        rl = ttnn.matmul(h2, dl["router_w"], compute_kernel_config=hifi4)
        rl_np = from_dev(rl, (B * T, n_experts))
        rl_np = rl_np - rl_np.max(axis=-1, keepdims=True)
        probs = np.exp(rl_np) / np.exp(rl_np).sum(axis=-1, keepdims=True)

        active = set()
        token_top4 = []
        for t in range(B * T):
            t4 = np.argsort(probs[t])[-top_k:]
            active.update(t4)
            token_top4.append(set(t4))

        moe_np = np.zeros((B * T, hidden), dtype=np.float32)
        h2_tt = to_bf16(h2_np)
        for e in active:
            w_e = np.zeros((B * T, 1), dtype=np.float32)
            for t in range(B * T):
                if e in token_top4[t]:
                    w_e[t, 0] = probs[t, e]
            ew = dl["experts"][e]
            g = ttnn.matmul(h2_tt, ew["g"], compute_kernel_config=hifi4)
            u = ttnn.matmul(h2_tt, ew["u"], compute_kernel_config=hifi4)
            d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), ew["d"], compute_kernel_config=hifi4)
            moe_np += w_e * from_dev(d, (B * T, hidden))

        sg = ttnn.matmul(h2_tt, dl["s_gate_w"], compute_kernel_config=hifi4)
        su = ttnn.matmul(h2_tt, dl["s_up_w"], compute_kernel_config=hifi4)
        sd = ttnn.matmul(ttnn.mul(ttnn.silu(sg), su), dl["s_down_w"], compute_kernel_config=hifi4)
        sd_np = from_dev(sd, (B * T, hidden))
        if dl["seg_w"] is not None:
            seg_logit = h2_np @ seg_w_np_cache[i]
            seg_val = 1.0 / (1.0 + np.exp(-seg_logit))
            moe_np += seg_val * sd_np
        else:
            moe_np += sd_np

        x2_np = from_dev(x2, (B * T, hidden))
        x_np = (x2_np + moe_np).reshape(B, T, hidden)
        if i < 3 or (i + 1) % 6 == 0:
            print(f"    Layer {i+1}/{n_layers}: norm={np.linalg.norm(x_np):.2f}")

    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B * T, hidden)), weight=final_g, epsilon=rms_eps)
    logits = from_dev(ttnn.matmul(x_tt, lm_h, compute_kernel_config=hifi4), (B * T, vocab_size))
    last = logits[-1]
    top5 = np.argsort(last)[-5:][::-1]
    print(f"    Top-5: {[(tokenizer.decode([t]), f'{last[t]:.1f}') for t in top5]}")
    return last


# ── Device-routed decode step ────────────────────────────────
def decode_step(token_id, pos):
    """
    Decode with on-device routing: softmax, topk, sigmoid all on device.
    Only CPU read: 4 expert indices + 4 prob weights = 32 bytes per layer.
    """
    x = to_bf16(embed_w[token_id:token_id + 1].reshape(1, 1, hidden))

    angles = pos * freqs
    cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    cos_tt = to_dev_4d(cos_np)
    sin_tt = to_dev_4d(sin_np)
    pos_tt = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)

    for i in range(n_layers):
        dl = dev_layers[i]

        # ── Attention (fused bias into linear) ────────────────
        h = ttnn.rms_norm(x, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.reshape(ttnn.linear(h, dl["q_w"], bias=dl["q_b"], compute_kernel_config=hifi4),
                         [1, n_q_heads, 1, head_dim])
        k = ttnn.reshape(ttnn.linear(h, dl["k_w"], bias=dl["k_b"], compute_kernel_config=hifi4),
                         [1, n_kv_heads, 1, head_dim])
        v = ttnn.reshape(ttnn.linear(h, dl["v_w"], bias=dl["v_b"], compute_kernel_config=hifi4),
                         [1, n_kv_heads, 1, head_dim])
        qr = ttnn.experimental.rotary_embedding(q, cos_tt, sin_tt)
        kr = ttnn.experimental.rotary_embedding(k, cos_tt, sin_tt)
        if list(qr.shape)[2] > 1:
            qr = ttnn.slice(qr, [0, 0, 0, 0], [1, n_q_heads, 1, head_dim])
        if list(kr.shape)[2] > 1:
            kr = ttnn.slice(kr, [0, 0, 0, 0], [1, n_kv_heads, 1, head_dim])
        ks = ttnn.to_memory_config(ttnn.reshape(kr, [1, 1, n_kv_heads, head_dim]), kv_cfg)
        vs = ttnn.to_memory_config(ttnn.reshape(v, [1, 1, n_kv_heads, head_dim]), kv_cfg)
        ttnn.experimental.paged_update_cache(k_caches[i], ks, update_idxs_tensor=pos_tt)
        ttnn.experimental.paged_update_cache(v_caches[i], vs, update_idxs_tensor=pos_tt)
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn.reshape(qr, [1, 1, n_q_heads, head_dim]), k_caches[i], v_caches[i],
            cur_pos_tensor=pos_tt, compute_kernel_config=hifi4)
        o = ttnn.matmul(ttnn.reshape(attn, [1, 1, 1, hidden]), dl["o_w"], compute_kernel_config=hifi4)
        if dl["o_b"] is not None:
            o = ttnn.add(o, dl["o_b"])
        x2 = ttnn.add(x, o)

        # ── MoE: device-side routing ─────────────────────────
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)

        # Router: matmul → softmax → mask padding → topk (ALL ON DEVICE)
        rl = ttnn.matmul(h2, dl["router_w"], compute_kernel_config=hifi4)  # [1,1,1,64]
        probs = ttnn.softmax(rl, dim=-1)  # [1,1,1,60] — on device!
        top4_vals, top4_idxs = ttnn.topk(probs, top_k)  # [1,1,1,4] each

        # Single sync: read only 8 values (32 bytes)
        ttnn.synchronize_device(device)
        top4_vals_np = from_dev(top4_vals, (top_k,))
        top4_idxs_np = from_dev(top4_idxs, (top_k,)).astype(int)

        # ── Top-4 experts: fused silu into gate matmul ──────────
        moe_acc = None
        for rank in range(top_k):
            e = top4_idxs_np[rank]
            prob = float(top4_vals_np[rank])
            ew = dl["experts"][e]
            g = ttnn.linear(h2, ew["g"], activation="silu", compute_kernel_config=hifi4)
            u = ttnn.matmul(h2, ew["u"], compute_kernel_config=hifi4)
            d = ttnn.matmul(ttnn.mul(g, u), ew["d"], compute_kernel_config=hifi4)
            weighted = ttnn.multiply(d, prob)
            if moe_acc is None:
                moe_acc = weighted
            else:
                moe_acc = ttnn.add(moe_acc, weighted)

        # ── Shared expert: fused silu into gate matmul ───────
        sg = ttnn.linear(h2, dl["s_gate_w"], activation="silu", compute_kernel_config=hifi4)
        su = ttnn.matmul(h2, dl["s_up_w"], compute_kernel_config=hifi4)
        sd = ttnn.matmul(ttnn.mul(sg, su), dl["s_down_w"], compute_kernel_config=hifi4)
        if dl["seg_w"] is not None:
            # Device-side gate: matmul + sigmoid — no CPU readback!
            seg_logit = ttnn.matmul(h2, dl["seg_w"], compute_kernel_config=hifi4)
            seg_val = ttnn.sigmoid(seg_logit)
            shared_gated = ttnn.mul(sd, seg_val)
            moe_acc = ttnn.add(moe_acc, shared_gated)
        else:
            moe_acc = ttnn.add(moe_acc, sd)

        # Residual ON DEVICE
        x = ttnn.add(x2, moe_acc)

    logits = ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps),
                         lm_h, compute_kernel_config=hifi4)
    ttnn.synchronize_device(device)
    return from_dev(logits, (1, vocab_size))[0]


# ── Reset KV caches ──────────────────────────────────────────
def reset_kv():
    for i in range(n_layers):
        c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
        ttnn.copy(to_dev_4d(c), k_caches[i])
        ttnn.copy(to_dev_4d(c), v_caches[i])


# ══════════════════════════════════════════════════════════════
# GENERATE
# ══════════════════════════════════════════════════════════════

try:
    device.enable_program_cache()
    print("\nProgram cache: enabled")
except Exception as e:
    print(f"\nProgram cache: {e}")

# First: test ttnn.topk + ttnn.softmax + ttnn.sigmoid
print("\nTesting device-side ops...")
test_t = to_dev_4d(np.random.randn(1, 1, 1, 64).astype(np.float32))
try:
    test_sm = ttnn.softmax(test_t, dim=-1)
    print("  ttnn.softmax: OK")
except Exception as e:
    print(f"  ttnn.softmax: FAILED — {e}")
    print("  Falling back to exp 91 approach")
    sys.exit(1)

try:
    vals, idxs = ttnn.topk(test_sm, 4)
    v_np = from_dev(vals, (4,))
    i_np = from_dev(idxs, (4,)).astype(int)
    print(f"  ttnn.topk(k=4): OK — indices={i_np}, vals={v_np}")
except Exception as e:
    print(f"  ttnn.topk: FAILED — {e}")
    print("  Falling back to exp 91 approach")
    sys.exit(1)

try:
    test_s = to_dev_4d(np.array([[[[0.5]]]], dtype=np.float32))
    sig = ttnn.sigmoid(test_s)
    sig_np = from_dev(sig, (1,))
    print(f"  ttnn.sigmoid: OK — sigmoid(0.5) = {sig_np[0]:.4f} (expected 0.6225)")
except Exception as e:
    print(f"  ttnn.sigmoid: FAILED — {e}")

# Test fused linear+silu
try:
    test_x = to_dev_4d(np.random.randn(1, 1, 1, 64).astype(np.float32))
    test_w = to_bf16(np.random.randn(64, 64).astype(np.float32))
    fused_out = ttnn.linear(test_x, test_w, activation="silu", compute_kernel_config=hifi4)
    print(f"  ttnn.linear(activation='silu'): OK — shape {list(fused_out.shape)}")
except Exception as e:
    print(f"  ttnn.linear(activation='silu'): FAILED — {e}")
    print("  Falling back to unfused approach")
    sys.exit(1)

# Test fused linear+bias
try:
    test_b = to_bf16(np.random.randn(64).astype(np.float32))
    fused_bias = ttnn.linear(test_x, test_w, bias=test_b, compute_kernel_config=hifi4)
    print(f"  ttnn.linear(bias=...): OK — shape {list(fused_bias.shape)}")
except Exception as e:
    print(f"  ttnn.linear(bias=...): FAILED — {e}")

print()

prompts = [
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a Python function to check if a number is prime.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nExplain quantum computing in one sentence.<|im_end|>\n<|im_start|>assistant\n",
]

for prompt in prompts:
    print(f"\n{'=' * 60}")
    display = prompt.replace('<|im_start|>', '').replace('<|im_end|>', '')
    display = display.replace('user\n', '').replace('\nassistant\n', '').strip()
    print(f"Prompt: \"{display}\"")
    print(f"{'=' * 60}")

    reset_kv()
    tokens = list(tokenizer.encode(prompt))
    print(f"  {len(tokens)} prompt tokens")

    t_pf = time.perf_counter()
    logits = prefill(np.array(tokens))
    dt_pf = time.perf_counter() - t_pf
    next_id = int(np.argmax(logits))
    gen = [next_id]
    print(f"  Prefill: {dt_pf:.1f}s | First token: {tokenizer.decode([next_id])!r}")

    pos = len(tokens)
    eos_ids = {tokenizer.eos_token_id, 151643, 151644, 151645}
    max_gen = 100
    times = []

    for step in range(max_gen):
        t0 = time.perf_counter()
        logits = decode_step(next_id, pos)
        dt = time.perf_counter() - t0
        next_id = int(np.argmax(logits))
        gen.append(next_id)
        times.append(dt)
        pos += 1
        tok_text = tokenizer.decode([next_id])
        if step < 5 or (step + 1) % 20 == 0:
            print(f"    Step {step + 1}: {tok_text!r} ({dt * 1000:.0f}ms)")
        if next_id in eos_ids:
            break

    text = tokenizer.decode(gen, skip_special_tokens=True)
    avg_ms = np.mean(times) * 1000
    tok_s = 1000 / avg_ms if avg_ms > 0 else 0

    if len(times) > 3:
        steady_ms = np.mean(times[2:]) * 1000
        steady_tok_s = 1000 / steady_ms
    else:
        steady_ms = avg_ms
        steady_tok_s = tok_s

    print(f"\n  Output: {text}")
    print(f"  {len(gen)} tokens | avg {avg_ms:.0f} ms/tok ({tok_s:.1f} tok/s)")
    print(f"  Steady-state: {steady_ms:.0f} ms/tok ({steady_tok_s:.1f} tok/s)")

# ── Summary ──────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("SUMMARY — Experiment 93: Fused Ops MoE Decode")
print(f"{'=' * 60}")
print(f"  Model: Qwen1.5-MoE-A2.7B-Chat (14.3B total, 2.7B active)")
print(f"  New in exp 93:")
print(f"    - ttnn.linear(activation='silu') fuses silu into gate matmul")
print(f"    - ttnn.linear(bias=b) fuses bias add into Q/K/V projections")
print(f"    - ~192 fewer dispatches vs exp 92 (~5.8ms savings expected)")
print(f"  Baseline exp 92: 20.5 tok/s")

ttnn.close_device(device)
print("\nDone!")
