"""
Experiment 38: Full Qwen2.5-0.5B forward pass on Blackhole.

All 24 transformer layers + embedding + final RMSNorm + lm_head.
Compares logits against HuggingFace reference, reports top-5 predictions,
and benchmarks end-to-end latency.

Architecture (Qwen2.5-0.5B):
  - 24 layers, hidden=896, intermediate=4864
  - 14 Q heads, 2 KV heads, head_dim=64
  - RoPE theta=1000000.0, RMSNorm eps=1e-6
  - SwiGLU MLP, GQA attention with Q/K biases
  - Tied embedding / lm_head weights
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import time
import torch

from safetensors import safe_open
from huggingface_hub import hf_hub_download

import ttnn
from tt_jax import tensors

# ── Model config ────────────────────────────────────────────
hidden = 896
intermediate = 4864
n_q_heads = 14
n_kv_heads = 2
head_dim = 64
rms_eps = 1e-6
rope_theta = 1000000.0
n_layers = 24
vocab_size = 151936

# ── Load weights from safetensors ───────────────────────────
print("Downloading Qwen2.5-0.5B weights...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
print(f"  Model path: {model_path}")

print("Loading all weights...")
all_weights = {}
with safe_open(model_path, framework="numpy") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key)

print(f"  Total tensors: {len(all_weights)}")

# Check if lm_head is tied to embedding
has_lm_head = "lm_head.weight" in all_weights
print(f"  lm_head.weight present: {has_lm_head}")
if not has_lm_head:
    print("  -> Will reuse model.embed_tokens.weight for lm_head")

# ── Extract global weights ──────────────────────────────────
embed_w = all_weights["model.embed_tokens.weight"].astype(np.float32)  # (vocab, hidden)
final_norm_g = all_weights["model.norm.weight"].astype(np.float32)     # (hidden,)
if has_lm_head:
    lm_head_w = all_weights["lm_head.weight"].astype(np.float32).T    # (hidden, vocab)
else:
    lm_head_w = embed_w.T.copy()                                       # (hidden, vocab) — tied

print(f"  embed_w: {embed_w.shape}")
print(f"  final_norm_g: {final_norm_g.shape}")
print(f"  lm_head_w: {lm_head_w.shape}")

# ── Extract per-layer weights ───────────────────────────────
layer_weights = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {}
    for key, val in all_weights.items():
        if key.startswith(prefix):
            short = key[len(prefix):]
            lw[short] = val.astype(np.float32)
    layer_weights.append(lw)
    if i == 0:
        print(f"  Layer 0 keys: {sorted(lw.keys())}")

print(f"  Loaded weights for {len(layer_weights)} layers")

# Free raw weights dict
del all_weights

# ── Tokenize input ──────────────────────────────────────────
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="np")[0]  # (seq_len,)
seq_len = len(input_ids)
batch = 1
print(f"\nPrompt: '{prompt}'")
print(f"Token IDs: {input_ids} (len={seq_len})")

# ── HuggingFace reference (optional) ─────────────────────────
print("\n" + "=" * 60)
print("Running HuggingFace reference...")
print("=" * 60)

hf_logits = None
hf_top5_idx = None
try:
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32)
    hf_model.eval()

    with torch.no_grad():
        hf_input = torch.tensor(input_ids).unsqueeze(0)
        hf_output = hf_model(hf_input)
        hf_logits = hf_output.logits[0, -1].numpy()

    hf_top5_idx = np.argsort(hf_logits)[-5:][::-1]
    print("  HuggingFace top-5 next tokens:")
    for rank, idx in enumerate(hf_top5_idx):
        token_str = tokenizer.decode([idx])
        print(f"    {rank+1}. '{token_str}' (id={idx}, logit={hf_logits[idx]:.4f})")
    del hf_model
except Exception as e:
    print(f"  HF reference SKIPPED: {e}")
    print("  Will still run TT-NN forward and report top-5 predictions.")

# ── Open device ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Setting up TT-NN device...")
print("=" * 60)

device = ttnn.open_device(device_id=0)

def to_dev(arr):
    """Send numpy array to device as bfloat16 tile-layout tensor."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    """Send 4D numpy array to device."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    """Retrieve tensor from device as numpy."""
    t = ttnn.to_torch(tensor).float()
    try:
        return t.reshape(shape).numpy()
    except RuntimeError:
        return t.squeeze().numpy().reshape(shape)

def cosine(a, b):
    return np.dot(a.flatten(), b.flatten()) / (
        np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-8)

# ── Upload ALL weights to device ────────────────────────────
print("Uploading weights to device (all 24 layers + head)...")
t_upload_start = time.perf_counter()

# Per-layer device weights
dev_layers = []
for i in range(n_layers):
    lw = layer_weights[i]
    dl = {
        "ln1_g": to_dev(lw["input_layernorm.weight"]),
        "q_w": to_dev(lw["self_attn.q_proj.weight"].T),
        "q_b": to_dev(lw["self_attn.q_proj.bias"]),
        "k_w": to_dev(lw["self_attn.k_proj.weight"].T),
        "k_b": to_dev(lw["self_attn.k_proj.bias"]),
        "v_w": to_dev(lw["self_attn.v_proj.weight"].T),
        "v_b": to_dev(lw["self_attn.v_proj.bias"]),
        "o_w": to_dev(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_dev(lw["post_attention_layernorm.weight"]),
        "gate_w": to_dev(lw["mlp.gate_proj.weight"].T),
        "up_w": to_dev(lw["mlp.up_proj.weight"].T),
        "down_w": to_dev(lw["mlp.down_proj.weight"].T),
    }
    dev_layers.append(dl)
    if (i + 1) % 6 == 0 or i == 0:
        print(f"    Layer {i} uploaded")

# Global weights on device
final_norm_g_tt = to_dev(final_norm_g)
lm_head_w_tt = to_dev(lm_head_w)

t_upload = time.perf_counter() - t_upload_start
print(f"  All weights uploaded in {t_upload:.1f}s")

# Free CPU layer weights
del layer_weights

# ── Precompute RoPE tables ──────────────────────────────────
# Max seq_len we'll need — use the tokenized prompt length
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
positions = np.arange(seq_len, dtype=np.float32)
angles = np.outer(positions, freqs)  # (seq_len, head_dim/2)
cos_table = np.cos(angles).astype(np.float32)
sin_table = np.sin(angles).astype(np.float32)

def apply_rope(x_4d, n_heads):
    """Apply RoPE via CPU. x: (1, n_heads, seq_len, head_dim) numpy."""
    T = x_4d.shape[2]
    x_even = x_4d[..., 0::2]
    x_odd = x_4d[..., 1::2]
    cos_t = cos_table[None, None, :T, :]
    sin_t = sin_table[None, None, :T, :]
    out = np.zeros_like(x_4d)
    out[..., 0::2] = x_even * cos_t - x_odd * sin_t
    out[..., 1::2] = x_even * sin_t + x_odd * cos_t
    return out

# ── Full forward pass ───────────────────────────────────────
def qwen_forward(input_ids_np):
    """Full Qwen2.5-0.5B forward pass. Returns logits (1, seq_len, vocab)."""
    B = 1
    T = len(input_ids_np)

    # Embedding lookup (CPU — embedding is just a table lookup)
    x_np = embed_w[input_ids_np].reshape(B, T, hidden)  # (1, T, 896)

    # 24 transformer layers
    for i in range(n_layers):
        dl = dev_layers[i]

        x_tt = to_dev(x_np)

        # --- Self-attention ---
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)

        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"]), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"]), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"]), dl["v_b"])

        # Pull to CPU for reshape + RoPE
        q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

        q_4d = q_np.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k_4d = k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q_4d = apply_rope(q_4d, n_q_heads)
        k_4d = apply_rope(k_4d, n_kv_heads)

        q_dev = to_dev_4d(q_4d)
        k_dev = to_dev_4d(k_4d)
        v_dev = to_dev_4d(v_4d)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            q_dev, k_dev, v_dev, is_causal=True
        )

        attn_out_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
        attn_out_np = attn_out_np.transpose(0, 2, 1, 3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev(attn_out_np), dl["o_w"])

        # Residual
        x_tt2 = ttnn.add(x_tt, o_tt)

        # --- MLP ---
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)

        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"])
        up_tt = ttnn.matmul(h2_tt, dl["up_w"])
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"])

        out_tt = ttnn.add(x_tt2, down_tt)

        # Pull back to CPU for next layer
        x_np = from_dev(out_tt, (B, T, hidden))

        if (i + 1) % 6 == 0 or i == 0:
            print(f"    Layer {i:2d} done | x norm: {np.linalg.norm(x_np):.4f}")

    # Final RMSNorm + lm_head projection
    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt)

    logits_np = from_dev(logits_tt, (B, T, vocab_size))
    return logits_np


# ── Run full forward pass ───────────────────────────────────
print("\n" + "=" * 60)
print(f"Running full 24-layer forward pass on TT-NN...")
print(f"  Prompt: '{prompt}' ({seq_len} tokens)")
print("=" * 60)

t0 = time.perf_counter()
tt_logits = qwen_forward(input_ids)
t_fwd = time.perf_counter() - t0
print(f"\n  Forward pass time: {t_fwd*1000:.1f}ms")

# Last-token logits
tt_last_logits = tt_logits[0, -1]  # (vocab,)

# ── Compare against HuggingFace ─────────────────────────────
print("\n" + "=" * 60)
print("Comparison: TT-NN vs HuggingFace")
print("=" * 60)

# Top-5 from TT-NN (always available)
tt_top5_idx = np.argsort(tt_last_logits)[-5:][::-1]
print("\n  TT-NN top-5 next tokens:")
for rank, idx in enumerate(tt_top5_idx):
    token_str = tokenizer.decode([idx])
    print(f"    {rank+1}. '{token_str}' (id={idx}, logit={tt_last_logits[idx]:.4f})")

tt_top1 = tt_top5_idx[0]
print(f"\n  TT-NN prediction: '{tokenizer.decode([tt_top1])}'")

logit_cos = None
top1_match = None
overlap = None
hf_top1 = None

if hf_logits is not None:
    logit_cos = cosine(tt_last_logits, hf_logits)
    print(f"\n  Last-token logit cosine similarity: {logit_cos:.6f}")

    max_err = np.abs(tt_last_logits - hf_logits).max()
    mean_err = np.abs(tt_last_logits - hf_logits).mean()
    print(f"  Max absolute error:  {max_err:.4f}")
    print(f"  Mean absolute error: {mean_err:.4f}")

    print("\n  HuggingFace top-5 (for comparison):")
    for rank, idx in enumerate(hf_top5_idx):
        token_str = tokenizer.decode([idx])
        print(f"    {rank+1}. '{token_str}' (id={idx}, logit={hf_logits[idx]:.4f})")

    hf_top1 = hf_top5_idx[0]
    top1_match = tt_top1 == hf_top1
    print(f"\n  Top-1 match: {'YES' if top1_match else 'NO'} "
          f"(TT={tokenizer.decode([tt_top1])!r}, HF={tokenizer.decode([hf_top1])!r})")

    tt_top5_set = set(tt_top5_idx.tolist())
    hf_top5_set = set(hf_top5_idx.tolist())
    overlap = len(tt_top5_set & hf_top5_set)
    print(f"  Top-5 overlap: {overlap}/5")

    # Per-token logit cosine across sequence
    print("\n  Per-token logit cosine similarity:")
    try:
        with torch.no_grad():
            hf_full_logits = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32
            )(torch.tensor(input_ids).unsqueeze(0)).logits[0].numpy()
        for t in range(seq_len):
            tok_cos = cosine(tt_logits[0, t], hf_full_logits[t])
            tok_str = tokenizer.decode([input_ids[t]])
            print(f"    Token {t}: '{tok_str:>12s}' -> cosine {tok_cos:.6f}")
        del hf_full_logits
    except Exception as e:
        print(f"    Skipped per-token comparison: {e}")
else:
    print("\n  (HuggingFace reference unavailable — skipping comparison)")

# ── Benchmark ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Latency benchmark (3 iterations)")
print("=" * 60)

times = []
for run in range(3):
    t0 = time.perf_counter()
    _ = qwen_forward(input_ids)
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    print(f"  Run {run+1}: {elapsed*1000:.1f}ms")

times_ms = [t * 1000 for t in times]
print(f"\n  Mean: {np.mean(times_ms):.1f}ms")
print(f"  Min:  {np.min(times_ms):.1f}ms")
print(f"  Per-layer avg: {np.mean(times_ms)/24:.1f}ms")

# ── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
summary = f"""
  Model:              Qwen2.5-0.5B (full 24 layers)
  Prompt:             '{prompt}' ({seq_len} tokens)
  TT-NN prediction:   '{tokenizer.decode([tt_top1])}'
  Forward latency:    {np.mean(times_ms):.1f}ms
  Per-layer latency:  {np.mean(times_ms)/24:.1f}ms
  Upload time:        {t_upload:.1f}s"""

if logit_cos is not None:
    summary += f"""
  Logit cosine:       {logit_cos:.6f}
  Top-1 match:        {'YES' if top1_match else 'NO'}
  Top-5 overlap:      {overlap}/5
  HF prediction:      '{tokenizer.decode([hf_top1])}'
  Status:             {'PASS' if logit_cos > 0.95 else 'NEEDS INVESTIGATION'} (cosine > 0.95)"""
else:
    summary += """
  HF comparison:      SKIPPED (transformers not available)"""

print(summary)

ttnn.close_device(device)
print("Done!")
