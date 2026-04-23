#!/usr/bin/env python3
"""
Experiment 97: MoE swiglu Fusion — Concat gate+up, use ttnn.swiglu()

Building on exp 95 (22.7 tok/s, partial trace + fused ops).

Current expert MLP per expert (3 ops):
  g = ttnn.linear(h2, gate_w, activation="silu")   # matmul + silu fused
  u = ttnn.matmul(h2, up_w)                         # separate matmul
  d = ttnn.matmul(ttnn.mul(g, u), down_w)            # mul + matmul

With swiglu fusion (2 ops):
  gu = ttnn.matmul(h2, gate_up_w)                   # single matmul [hidden, 2*inter]
  d = ttnn.matmul(ttnn.swiglu(gu), down_w)           # swiglu + matmul

Savings: 1 op per expert × 4 experts × 24 layers = 96 ops
         1 op per shared expert × 24 layers = 24 ops
         Total: 120 ops × 30μs = 3.6ms → ~40ms → ~25 tok/s

Run: ssh tenstorrent 'cd tt-xla && python3 experiments/97_moe_swiglu.py'
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

# ── Architecture ─────────────────────────────────────────────
hidden = 2048; n_q_heads = 16; n_kv_heads = 16; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE = 32; batch_size = 1
n_experts = 60; top_k = 4

# ── Device ───────────────────────────────────────────────────
print("=" * 60)
print("Exp 97: MoE swiglu Fusion")
print("=" * 60)
device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150 ({grid.x}x{grid.y} cores)")

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)

# ── Download + helpers ───────────────────────────────────────
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

# ── Pre-test: does ttnn.swiglu work? ────────────────────────
print("\nPre-test: ttnn.swiglu()...")
try:
    # swiglu expects [batch, 2*dim] → splits in half, applies silu to first half, multiplies
    test_in = to_bf16(np.random.randn(1, 256).astype(np.float32))
    test_out = ttnn.swiglu(test_in)
    out_shape = list(test_out.shape)
    print(f"  Input: [1, 256] → Output: {out_shape}")
    assert out_shape[-1] == 128, f"Expected last dim 128, got {out_shape[-1]}"
    print("  ttnn.swiglu: OK!")
    SWIGLU_WORKS = True
except Exception as e:
    print(f"  ttnn.swiglu: FAILED — {e}")
    print("  Falling back to manual silu+mul")
    SWIGLU_WORKS = False

# ── Load embeddings ──────────────────────────────────────────
print("\nLoading embeddings + lm_head...")
embed_w = load_np("model.embed_tokens.weight")
final_norm_g = load_np("model.norm.weight")
lm_head_w = load_np("lm_head.weight").T if "lm_head.weight" in key_to_path else embed_w.T.copy()
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)

# ── Upload all 24 layers (with concatenated gate+up weights) ─
print(f"\nUploading {n_layers} layers (swiglu-fused experts)...")
t0_upload = time.perf_counter()
dev_layers = []
seg_w_np_cache = []

for li in range(n_layers):
    pfx = f"model.layers.{li}."
    dl = {}
    dl["ln1_g"] = to_bf16(load_np(pfx + "input_layernorm.weight"))
    dl["ln2_g"] = to_bf16(load_np(pfx + "post_attention_layernorm.weight"))
    dl["q_w"] = to_bf16(load_np(pfx + "self_attn.q_proj.weight").T)
    dl["q_b"] = to_bf16(load_np(pfx + "self_attn.q_proj.bias"))
    dl["k_w"] = to_bf16(load_np(pfx + "self_attn.k_proj.weight").T)
    dl["k_b"] = to_bf16(load_np(pfx + "self_attn.k_proj.bias"))
    dl["v_w"] = to_bf16(load_np(pfx + "self_attn.v_proj.weight").T)
    dl["v_b"] = to_bf16(load_np(pfx + "self_attn.v_proj.bias"))
    dl["o_w"] = to_bf16(load_np(pfx + "self_attn.o_proj.weight").T)
    o_bias_key = pfx + "self_attn.o_proj.bias"
    dl["o_b"] = to_bf16(load_np(o_bias_key)) if o_bias_key in key_to_path else None

    dl["router_w"] = to_bf16(load_np(pfx + "mlp.gate.weight").T)

    # ── KEY CHANGE: concatenate gate+up weights for swiglu ───
    experts = []
    for e in range(n_experts):
        ep = pfx + f"mlp.experts.{e}."
        gate_w = load_np(ep + "gate_proj.weight").T  # [hidden, inter]
        up_w = load_np(ep + "up_proj.weight").T      # [hidden, inter]
        gate_up = np.concatenate([gate_w, up_w], axis=1)  # [hidden, 2*inter]
        experts.append({
            "gu": to_bfp8(gate_up),
            "d": to_bfp8(load_np(ep + "down_proj.weight").T),
        })
    dl["experts"] = experts

    # Shared expert: also concatenate gate+up
    sp = pfx + "mlp.shared_expert."
    s_gate = load_np(sp + "gate_proj.weight").T
    s_up = load_np(sp + "up_proj.weight").T
    dl["s_gu_w"] = to_bfp8(np.concatenate([s_gate, s_up], axis=1))
    dl["s_down_w"] = to_bfp8(load_np(sp + "down_proj.weight").T)

    seg_key = pfx + "mlp.shared_expert_gate.weight"
    if seg_key in key_to_path:
        dl["seg_w"] = to_bf16(load_np(seg_key).T)
        seg_w_np_cache.append(load_np(seg_key).T.copy())
    else:
        dl["seg_w"] = None
        seg_w_np_cache.append(None)

    dev_layers.append(dl)
    elapsed = time.perf_counter() - t0_upload
    rem = elapsed / (li + 1) * (n_layers - li - 1)
    print(f"  Layer {li+1}/{n_layers} ({elapsed:.0f}s elapsed, ~{rem:.0f}s remaining)")

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


# ── Prefill ─────────────────────────────────────────────────
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

        # MoE (CPU routing for prefill)
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
            # swiglu path for prefill too
            gu = ttnn.matmul(h2_tt, ew["gu"], compute_kernel_config=hifi4)
            if SWIGLU_WORKS:
                act = ttnn.swiglu(gu)
            else:
                inter = list(gu.shape)[-1] // 2
                g_half = ttnn.slice(gu, [0, 0], [list(gu.shape)[0], inter])
                u_half = ttnn.slice(gu, [0, inter], [list(gu.shape)[0], 2 * inter])
                act = ttnn.mul(ttnn.silu(g_half), u_half)
            d = ttnn.matmul(act, ew["d"], compute_kernel_config=hifi4)
            moe_np += w_e * from_dev(d, (B * T, hidden))

        # Shared expert with swiglu
        sgu = ttnn.matmul(h2_tt, dl["s_gu_w"], compute_kernel_config=hifi4)
        if SWIGLU_WORKS:
            s_act = ttnn.swiglu(sgu)
        else:
            s_inter = list(sgu.shape)[-1] // 2
            sg = ttnn.slice(sgu, [0, 0], [list(sgu.shape)[0], s_inter])
            su = ttnn.slice(sgu, [0, s_inter], [list(sgu.shape)[0], 2 * s_inter])
            s_act = ttnn.mul(ttnn.silu(sg), su)
        sd = ttnn.matmul(s_act, dl["s_down_w"], compute_kernel_config=hifi4)
        sd_np = from_dev(sd, (B * T, hidden))
        if dl["seg_w"] is not None:
            seg_logit = h2_np @ seg_w_np_cache[i]
            seg_val = 1.0 / (1.0 + np.exp(-seg_logit))
            moe_np += seg_val * sd_np
        else:
            moe_np += sd_np

        x2_np = from_dev(x2, (B * T, hidden))
        x_np = (x2_np + moe_np).reshape(B, T, hidden)

    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B * T, hidden)), weight=final_g, epsilon=rms_eps)
    logits = from_dev(ttnn.matmul(x_tt, lm_h, compute_kernel_config=hifi4), (B * T, vocab_size))
    return logits[-1]


# ══════════════════════════════════════════════════════════════
# CAPTURE ATTENTION TRACES (same as exp 95)
# ══════════════════════════════════════════════════════════════

try:
    device.enable_program_cache()
    print("\nProgram cache: enabled")
except:
    pass

print("\n── Capturing attention traces for 24 layers ──")

dummy_x = to_dev_4d(np.zeros((1, 1, 1, hidden), dtype=np.float32))
dummy_cos = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
dummy_sin = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
dummy_pos = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

# Warmup attention kernels
print("  Warming up attention kernels...")
for i in range(n_layers):
    dl = dev_layers[i]
    h = ttnn.rms_norm(dummy_x, weight=dl["ln1_g"], epsilon=rms_eps)
    q = ttnn.reshape(ttnn.linear(h, dl["q_w"], bias=dl["q_b"], compute_kernel_config=hifi4),
                     [1, n_q_heads, 1, head_dim])
    k = ttnn.reshape(ttnn.linear(h, dl["k_w"], bias=dl["k_b"], compute_kernel_config=hifi4),
                     [1, n_kv_heads, 1, head_dim])
    v = ttnn.reshape(ttnn.linear(h, dl["v_w"], bias=dl["v_b"], compute_kernel_config=hifi4),
                     [1, n_kv_heads, 1, head_dim])
    qr = ttnn.experimental.rotary_embedding(q, dummy_cos, dummy_sin)
    kr = ttnn.experimental.rotary_embedding(k, dummy_cos, dummy_sin)
    if list(qr.shape)[2] > 1:
        qr = ttnn.slice(qr, [0, 0, 0, 0], [1, n_q_heads, 1, head_dim])
    if list(kr.shape)[2] > 1:
        kr = ttnn.slice(kr, [0, 0, 0, 0], [1, n_kv_heads, 1, head_dim])
    ks = ttnn.to_memory_config(ttnn.reshape(kr, [1, 1, n_kv_heads, head_dim]), kv_cfg)
    vs = ttnn.to_memory_config(ttnn.reshape(v, [1, 1, n_kv_heads, head_dim]), kv_cfg)
    ttnn.experimental.paged_update_cache(k_caches[i], ks, update_idxs_tensor=dummy_pos)
    ttnn.experimental.paged_update_cache(v_caches[i], vs, update_idxs_tensor=dummy_pos)
    attn = ttnn.transformer.scaled_dot_product_attention_decode(
        ttnn.reshape(qr, [1, 1, n_q_heads, head_dim]), k_caches[i], v_caches[i],
        cur_pos_tensor=dummy_pos, compute_kernel_config=hifi4)
    o = ttnn.matmul(ttnn.reshape(attn, [1, 1, 1, hidden]), dl["o_w"], compute_kernel_config=hifi4)
    if dl["o_b"] is not None:
        o = ttnn.add(o, dl["o_b"])
    _ = ttnn.add(dummy_x, o)
ttnn.synchronize_device(device)
print("  Warmup complete")

# Capture attention traces
attn_traces = []
attn_x_ins = []
attn_x_outs = []
cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

for i in range(n_layers):
    dl = dev_layers[i]
    x_in = to_dev_4d(np.zeros((1, 1, 1, hidden), dtype=np.float32))
    attn_x_ins.append(x_in)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)

    h = ttnn.rms_norm(x_in, weight=dl["ln1_g"], epsilon=rms_eps)
    q = ttnn.reshape(ttnn.linear(h, dl["q_w"], bias=dl["q_b"], compute_kernel_config=hifi4),
                     [1, n_q_heads, 1, head_dim])
    k = ttnn.reshape(ttnn.linear(h, dl["k_w"], bias=dl["k_b"], compute_kernel_config=hifi4),
                     [1, n_kv_heads, 1, head_dim])
    v = ttnn.reshape(ttnn.linear(h, dl["v_w"], bias=dl["v_b"], compute_kernel_config=hifi4),
                     [1, n_kv_heads, 1, head_dim])
    qr = ttnn.experimental.rotary_embedding(q, cos_buf, sin_buf)
    kr = ttnn.experimental.rotary_embedding(k, cos_buf, sin_buf)
    if list(qr.shape)[2] > 1:
        qr = ttnn.slice(qr, [0, 0, 0, 0], [1, n_q_heads, 1, head_dim])
    if list(kr.shape)[2] > 1:
        kr = ttnn.slice(kr, [0, 0, 0, 0], [1, n_kv_heads, 1, head_dim])
    ks = ttnn.to_memory_config(ttnn.reshape(kr, [1, 1, n_kv_heads, head_dim]), kv_cfg)
    vs = ttnn.to_memory_config(ttnn.reshape(v, [1, 1, n_kv_heads, head_dim]), kv_cfg)
    ttnn.experimental.paged_update_cache(k_caches[i], ks, update_idxs_tensor=pos_buf)
    ttnn.experimental.paged_update_cache(v_caches[i], vs, update_idxs_tensor=pos_buf)
    attn = ttnn.transformer.scaled_dot_product_attention_decode(
        ttnn.reshape(qr, [1, 1, n_q_heads, head_dim]), k_caches[i], v_caches[i],
        cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
    o = ttnn.matmul(ttnn.reshape(attn, [1, 1, 1, hidden]), dl["o_w"], compute_kernel_config=hifi4)
    if dl["o_b"] is not None:
        o = ttnn.add(o, dl["o_b"])
    x_out = ttnn.add(x_in, o)

    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    attn_traces.append(trace_id)
    attn_x_outs.append(x_out)
    print(f"  Layer {i+1}/24 trace captured")

print(f"  All 24 attention traces captured!")


# ── Decode step: traced attention + swiglu MoE ───────────────
def decode_step(token_id, pos):
    x = to_dev_4d(embed_w[token_id:token_id + 1].reshape(1, 1, 1, hidden))

    angles = pos * freqs
    cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)

    ttnn.copy(to_dev_4d(cos_np), cos_buf)
    ttnn.copy(to_dev_4d(sin_np), sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)

    for i in range(n_layers):
        dl = dev_layers[i]

        # ── Traced attention ─────────────────────────────────
        ttnn.copy(x, attn_x_ins[i])
        ttnn.execute_trace(device, attn_traces[i], cq_id=0, blocking=False)
        x2 = attn_x_outs[i]

        # ── MoE: device routing + swiglu experts ─────────────
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)

        rl = ttnn.matmul(h2, dl["router_w"], compute_kernel_config=hifi4)
        probs = ttnn.softmax(rl, dim=-1)
        top4_vals, top4_idxs = ttnn.topk(probs, top_k)

        ttnn.synchronize_device(device)
        top4_vals_np = from_dev(top4_vals, (top_k,))
        top4_idxs_np = from_dev(top4_idxs, (top_k,)).astype(int)

        # Top-4 experts with swiglu fusion
        moe_acc = None
        for rank in range(top_k):
            e = top4_idxs_np[rank]
            prob = float(top4_vals_np[rank])
            ew = dl["experts"][e]

            # Single matmul → swiglu → down projection
            gu = ttnn.matmul(h2, ew["gu"], compute_kernel_config=hifi4)
            if SWIGLU_WORKS:
                act = ttnn.swiglu(gu)
            else:
                inter = list(gu.shape)[-1] // 2
                g_half = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 1, inter])
                u_half = ttnn.slice(gu, [0, 0, 0, inter], [1, 1, 1, 2 * inter])
                act = ttnn.mul(ttnn.silu(g_half), u_half)
            d = ttnn.matmul(act, ew["d"], compute_kernel_config=hifi4)
            weighted = ttnn.multiply(d, prob)
            if moe_acc is None:
                moe_acc = weighted
            else:
                moe_acc = ttnn.add(moe_acc, weighted)

        # Shared expert with swiglu
        sgu = ttnn.matmul(h2, dl["s_gu_w"], compute_kernel_config=hifi4)
        if SWIGLU_WORKS:
            s_act = ttnn.swiglu(sgu)
        else:
            s_inter = list(sgu.shape)[-1] // 2
            sg = ttnn.slice(sgu, [0, 0, 0, 0], [1, 1, 1, s_inter])
            su = ttnn.slice(sgu, [0, 0, 0, s_inter], [1, 1, 1, 2 * s_inter])
            s_act = ttnn.mul(ttnn.silu(sg), su)
        sd = ttnn.matmul(s_act, dl["s_down_w"], compute_kernel_config=hifi4)
        if dl["seg_w"] is not None:
            seg_logit = ttnn.matmul(h2, dl["seg_w"], compute_kernel_config=hifi4)
            seg_val = ttnn.sigmoid(seg_logit)
            shared_gated = ttnn.mul(sd, seg_val)
            moe_acc = ttnn.add(moe_acc, shared_gated)
        else:
            moe_acc = ttnn.add(moe_acc, sd)

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
print("SUMMARY — Experiment 97: swiglu Fusion MoE Decode")
print(f"{'=' * 60}")
print(f"  Model: Qwen1.5-MoE-A2.7B-Chat (14.3B total, 2.7B active)")
print(f"  New in exp 97:")
print(f"    - Concatenated gate+up weights → single matmul per expert")
print(f"    - ttnn.swiglu() fuses silu+mul into one op")
print(f"    - Saves ~120 dispatches per token (~3.6ms)")
print(f"    - Combined with partial trace from exp 95")
print(f"  Baseline exp 95: 22.7 tok/s (44ms/tok)")
print(f"  Expected: ~25 tok/s (~40ms/tok)")
print(f"  swiglu available: {SWIGLU_WORKS}")

for trace_id in attn_traces:
    ttnn.release_trace(device, trace_id)

ttnn.close_device(device)
print("\nDone!")
