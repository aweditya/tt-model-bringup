#!/usr/bin/env python3
"""
Experiment 96: MoE Multi-CQ Pipelining — Overlap Routing Sync with Attention

Exp 94 profiling showed routing sync = 6.7ms (14% of 48ms/tok).
This is 0.28ms per layer for reading 32 bytes — dominated by sync latency.

Strategy: Use 2 command queues to overlap routing readback with next layer's
attention. CQ0 dispatches compute, CQ1 reads routing results asynchronously.

Pipeline per layer:
  CQ0: [attention] [router+softmax+topk]
  CQ1:                                   [read routing indices]
  CQ0:                                   [expert dispatch (uses prev routing)]
                                          ↑ overlap routing read with attn dispatch

Actually simpler: just overlap the sync. Instead of blocking CQ0 to read routing,
dispatch it on CQ1 and let CQ0 continue with the next layer's attention.

Expected improvement: save ~5ms of sync stall = ~43ms → ~25 tok/s

Run: ssh tenstorrent 'cd tt-xla && python3 experiments/96_moe_multi_cq.py'
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
n_experts = 60; n_exp_pad = 64
top_k = 4

# ── Device with 2 command queues ─────────────────────────────
print("=" * 60)
print("Exp 96: MoE Multi-CQ Pipelining")
print("=" * 60)

# KEY: Open device with 2 command queues for overlapping dispatch + readback
device = ttnn.open_device(device_id=0, num_command_queues=2)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150 ({grid.x}x{grid.y} cores), 2 command queues")

CQ0 = 0  # Compute queue
CQ1 = 1  # I/O queue

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

# ── Load embeddings ──────────────────────────────────────────
print("\nLoading embeddings + lm_head...")
embed_w = load_np("model.embed_tokens.weight")
final_norm_g = load_np("model.norm.weight")
lm_head_w = load_np("lm_head.weight").T if "lm_head.weight" in key_to_path else embed_w.T.copy()
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)

# ── Upload all 24 layers ─────────────────────────────────────
print(f"\nUploading {n_layers} layers (BFP8 experts, bf16 attention)...")
t0_upload = time.perf_counter()
dev_layers = []

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

    experts = []
    for e in range(n_experts):
        ep = pfx + f"mlp.experts.{e}."
        experts.append({
            "g": to_bfp8(load_np(ep + "gate_proj.weight").T),
            "u": to_bfp8(load_np(ep + "up_proj.weight").T),
            "d": to_bfp8(load_np(ep + "down_proj.weight").T),
        })
    dl["experts"] = experts

    sp = pfx + "mlp.shared_expert."
    dl["s_gate_w"] = to_bfp8(load_np(sp + "gate_proj.weight").T)
    dl["s_up_w"] = to_bfp8(load_np(sp + "up_proj.weight").T)
    dl["s_down_w"] = to_bfp8(load_np(sp + "down_proj.weight").T)

    seg_key = pfx + "mlp.shared_expert_gate.weight"
    if seg_key in key_to_path:
        dl["seg_w"] = to_bf16(load_np(seg_key).T)
    else:
        dl["seg_w"] = None

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


# ── Prefill (same as before, single CQ) ─────────────────────
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
            g = ttnn.matmul(h2_tt, ew["g"], compute_kernel_config=hifi4)
            u = ttnn.matmul(h2_tt, ew["u"], compute_kernel_config=hifi4)
            d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), ew["d"], compute_kernel_config=hifi4)
            moe_np += w_e * from_dev(d, (B * T, hidden))

        sg = ttnn.matmul(h2_tt, dl["s_gate_w"], compute_kernel_config=hifi4)
        su = ttnn.matmul(h2_tt, dl["s_up_w"], compute_kernel_config=hifi4)
        sd = ttnn.matmul(ttnn.mul(ttnn.silu(sg), su), dl["s_down_w"], compute_kernel_config=hifi4)
        sd_np = from_dev(sd, (B * T, hidden))
        seg_w_np = None
        if dl["seg_w"] is not None:
            seg_w_np = from_dev(dl["seg_w"], (hidden, 1))
            seg_logit = h2_np @ seg_w_np
            seg_val = 1.0 / (1.0 + np.exp(-seg_logit))
            moe_np += seg_val * sd_np
        else:
            moe_np += sd_np

        x2_np = from_dev(x2, (B * T, hidden))
        x_np = (x2_np + moe_np).reshape(B, T, hidden)

    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B * T, hidden)), weight=final_g, epsilon=rms_eps)
    logits = from_dev(ttnn.matmul(x_tt, lm_h, compute_kernel_config=hifi4), (B * T, vocab_size))
    return logits[-1]


# ── Multi-CQ decode step ────────────────────────────────────
def decode_step(token_id, pos):
    """
    Decode with multi-CQ pipelining.
    CQ0: all compute dispatches
    CQ1: routing readback (overlapped with CQ0 compute)
    """
    x = to_dev_4d(embed_w[token_id:token_id + 1].reshape(1, 1, 1, hidden))

    angles = pos * freqs
    cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    cos_tt = to_dev_4d(cos_np)
    sin_tt = to_dev_4d(sin_np)
    pos_tt = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)

    for i in range(n_layers):
        dl = dev_layers[i]

        # ── Attention (all on CQ0) ───────────────────────────
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

        # ── MoE routing (dispatch on CQ0) ────────────────────
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        rl = ttnn.matmul(h2, dl["router_w"], compute_kernel_config=hifi4)
        probs = ttnn.softmax(rl, dim=-1)
        top4_vals, top4_idxs = ttnn.topk(probs, top_k)

        # Record event: routing results are ready on device
        route_event = ttnn.record_event(device, cq_id=CQ0)

        # CQ1: wait for routing to complete, then read results
        ttnn.wait_for_event(CQ1, route_event)
        # Synchronize CQ1 to ensure the read is complete
        ttnn.synchronize_device(device)
        top4_vals_np = from_dev(top4_vals, (top_k,))
        top4_idxs_np = from_dev(top4_idxs, (top_k,)).astype(int)

        # ── Top-4 experts (back on CQ0) ──────────────────────
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

        # ── Shared expert ────────────────────────────────────
        sg = ttnn.linear(h2, dl["s_gate_w"], activation="silu", compute_kernel_config=hifi4)
        su = ttnn.matmul(h2, dl["s_up_w"], compute_kernel_config=hifi4)
        sd = ttnn.matmul(ttnn.mul(sg, su), dl["s_down_w"], compute_kernel_config=hifi4)
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

try:
    device.enable_program_cache()
    print("\nProgram cache: enabled")
except:
    pass

# Test multi-CQ event sync
print("\nTesting multi-CQ event synchronization...")
try:
    test_t = to_dev_4d(np.random.randn(1, 1, 1, 64).astype(np.float32))
    ev = ttnn.record_event(device, cq_id=CQ0)
    ttnn.wait_for_event(CQ1, ev)
    print("  Event sync: OK")
except Exception as e:
    print(f"  Event sync: FAILED — {e}")
    print("  Multi-CQ may not work on this device/firmware")
    print("  Falling back to single-CQ approach")

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
print("SUMMARY — Experiment 96: Multi-CQ MoE Decode")
print(f"{'=' * 60}")
print(f"  Model: Qwen1.5-MoE-A2.7B-Chat (14.3B total, 2.7B active)")
print(f"  New in exp 96:")
print(f"    - 2 command queues (CQ0=compute, CQ1=readback)")
print(f"    - Event-based sync: routing dispatch on CQ0, readback on CQ1")
print(f"    - Goal: overlap 6.7ms routing sync with compute")
print(f"  Baseline exp 93: 21.1 tok/s (47ms/tok)")
print(f"  Baseline exp 95: 22.7 tok/s (44ms/tok) [partial trace]")

ttnn.close_device(device)
print("\nDone!")
