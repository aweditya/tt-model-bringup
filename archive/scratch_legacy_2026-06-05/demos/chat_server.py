#!/usr/bin/env python3
"""
Chat server: Qwen2.5-0.5B on Tenstorrent Blackhole P150 via TT-NN.

Serves an HTTP API for text generation using traced decode on device.
Based on experiments/53e_traced_paged_decode.py and experiments/54b_traced_sampling.py.

Usage:
    python3 ~/tt-xla/demos/chat_server.py [--port 8080]

API:
    POST /generate  {"prompt": "Hello!", "max_tokens": 100, "temperature": 0.7, "top_k": 50}
    GET  /health    Returns {"status": "ready"} when model is loaded
"""

import sys, os, time, json, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import argparse
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Qwen2.5-0.5B chat server on Blackhole")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ── Model constants ──────────────────────────────────────────
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE_SIZE = 32; batch_size = 1

# ── Device setup ─────────────────────────────────────────────
print("Opening device...")
hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# ── Load model ───────────────────────────────────────────────
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

# ── Device helpers ───────────────────────────────────────────
def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# ── RoPE setup ───────────────────────────────────────────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

def get_rope_tables_half(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))

def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

# ── Upload weights ───────────────────────────────────────────
print("Uploading weights to device...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dev_layers.append({
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
    })
final_norm_g_tt = to_dev(final_norm_g)
lm_head_w_tt = to_dev(lm_head_w)
del layer_weights_np
print(f"  Uploaded in {(time.perf_counter()-t0)*1000:.0f}ms")

# ── KV caches ────────────────────────────────────────────────
k_caches, v_caches = [], []
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(to_dev_4d(c.copy()))
    v_caches.append(to_dev_4d(c.copy()))

kv_shard_height = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
kv_core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_mem_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_shard_height, head_dim),
    core_grid=kv_core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)

# ── Input buffers for trace ──────────────────────────────────
embed_buf = to_dev(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

def update_buffers(token_id, pos):
    """Update all input buffers before trace replay."""
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_dev(x_np), embed_buf)

    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)

    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)

def reset_kv_caches():
    """Zero out KV caches between requests."""
    for i in range(n_layers):
        c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
        ttnn.copy(to_dev_4d(c), k_caches[i])
        ttnn.copy(to_dev_4d(c), v_caches[i])

# ── Sampling ─────────────────────────────────────────────────
def sample_top_k(logits, temp=0.7, top_k=50):
    """Temperature-scaled top-k sampling. CPU-side, outside trace."""
    if temp <= 0:
        return int(np.argmax(logits))
    logits = logits / temp
    top_idx = np.argpartition(logits, -top_k)[-top_k:]
    top_logits = logits[top_idx]
    probs = np.exp(top_logits - np.max(top_logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(top_idx, p=probs))

# ── Prefill (CPU RoPE) ──────────────────────────────────────
def prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_half(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np.reshape(B * T, hidden))
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

        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_4d), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_4d), batch_index=0)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d),
            is_causal=True, compute_kernel_config=hifi4)
        attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim)).transpose(0,2,1,3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev(attn_np.reshape(B*T, hidden)), dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B * T, hidden)).reshape(B, T, hidden)

    x_tt = to_dev(x_np.reshape(B * T, hidden))
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (B * T, vocab_size))[-1]

# ── Decode step (traceable) ─────────────────────────────────
def decode_forward():
    """Full 24-layer decode using buffer references. All dynamic values via buffers."""
    x_tt = embed_buf

    for i in range(n_layers):
        dl = dev_layers[i]
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)

        q_4d = ttnn.reshape(ttnn.linear(h_tt, dl["q_w"], bias=dl["q_b"], compute_kernel_config=hifi4),
                            [1, n_q_heads, 1, head_dim])
        k_4d = ttnn.reshape(ttnn.linear(h_tt, dl["k_w"], bias=dl["k_b"], compute_kernel_config=hifi4),
                            [1, n_kv_heads, 1, head_dim])
        v_4d = ttnn.reshape(ttnn.linear(h_tt, dl["v_w"], bias=dl["v_b"], compute_kernel_config=hifi4),
                            [1, n_kv_heads, 1, head_dim])

        # Native RoPE (half-format for Qwen)
        q_roped = ttnn.experimental.rotary_embedding(q_4d, rope_cos_buf, rope_sin_buf)
        k_roped = ttnn.experimental.rotary_embedding(k_4d, rope_cos_buf, rope_sin_buf)
        if list(q_roped.shape)[2] > 1:
            q_roped = ttnn.slice(q_roped, [0,0,0,0], [1,n_q_heads,1,head_dim])
        if list(k_roped.shape)[2] > 1:
            k_roped = ttnn.slice(k_roped, [0,0,0,0], [1,n_kv_heads,1,head_dim])

        k_for_cache = ttnn.reshape(k_roped, [1, 1, n_kv_heads, head_dim])
        v_for_cache = ttnn.reshape(v_4d, [1, 1, n_kv_heads, head_dim])
        k_sharded = ttnn.to_memory_config(k_for_cache, kv_mem_cfg)
        v_sharded = ttnn.to_memory_config(v_for_cache, kv_mem_cfg)
        ttnn.experimental.paged_update_cache(k_caches[i], k_sharded, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches[i], v_sharded, update_idxs_tensor=pos_buf)

        q_decode = ttnn.reshape(q_roped, [1, 1, n_q_heads, head_dim])
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)

        merged = ttnn.reshape(attn, [1, 1, 1, hidden])
        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, o_tt)

        # MLP with fused gate+silu
        h2_tt = ttnn.rms_norm(x_tt, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.linear(h2_tt, dl["gate_w"], activation="silu", compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        down_tt = ttnn.matmul(ttnn.mul(gate_tt, up_tt), dl["down_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, down_tt)

    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return logits_tt


# ══════════════════════════════════════════════════════════════
# Capture trace at startup
# ══════════════════════════════════════════════════════════════
print("Warming up decode path...")
# Run a dummy prefill + decode to populate program cache
dummy_tokens = tokenizer.encode("Hello")
_ = prefill(np.array(dummy_tokens))
update_buffers(dummy_tokens[-1], len(dummy_tokens) - 1)
_ = decode_forward()
ttnn.synchronize_device(device)

# Enable program cache
try:
    ttnn.device.enable_program_cache(device)
except AttributeError:
    try:
        device.enable_program_cache()
    except:
        print("  (program cache API not found, continuing)")

# Capture trace
print("Capturing trace...")
update_buffers(dummy_tokens[-1], len(dummy_tokens) - 1)
t_cap0 = time.perf_counter()
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward()
ttnn.end_trace_capture(device, trace_id, cq_id=0)
t_cap = time.perf_counter() - t_cap0
print(f"  Trace captured in {t_cap*1000:.0f}ms")

# Reset caches after warmup
reset_kv_caches()

# Lock to serialize requests (device is single-user)
generate_lock = threading.Lock()
request_count = 0


# ══════════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════════
def generate(prompt, max_tokens=100, temperature=0.7, top_k=50, seed=None):
    """Run prefill + traced decode for a prompt. Returns (text, timing_info)."""
    global trace_id, logits_ref

    if seed is not None:
        np.random.seed(seed)

    # Release old trace, reset caches, re-capture for fresh KV state
    ttnn.release_trace(device, trace_id)
    reset_kv_caches()

    tokens_list = tokenizer.encode(prompt)
    if len(tokens_list) >= MAX_SEQ - 1:
        tokens_list = tokens_list[:MAX_SEQ - 2]
    max_gen = min(max_tokens, MAX_SEQ - len(tokens_list) - 1)

    # Prefill
    t_prefill_start = time.perf_counter()
    logits = prefill(np.array(tokens_list))
    t_prefill = time.perf_counter() - t_prefill_start

    next_id = sample_top_k(logits, temp=temperature, top_k=top_k)
    tokens_list.append(next_id)
    generated_ids = [next_id]

    # Warmup decode + re-capture trace (caches were reset and re-prefilled)
    update_buffers(next_id, len(tokens_list) - 1)
    _ = decode_forward()
    ttnn.synchronize_device(device)

    update_buffers(next_id, len(tokens_list) - 1)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    logits_ref = decode_forward()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    # Traced decode loop
    decode_times = []
    t_decode_start = time.perf_counter()
    for step in range(max_gen - 1):
        pos = len(tokens_list) - 1
        update_buffers(next_id, pos)

        t0 = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        dt = time.perf_counter() - t0
        decode_times.append(dt)

        logits = from_dev(logits_ref, (1, 1, vocab_size))[0, 0]
        next_id = sample_top_k(logits, temp=temperature, top_k=top_k)
        tokens_list.append(next_id)
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    t_decode_total = time.perf_counter() - t_decode_start
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Timing stats
    n_gen = len(generated_ids)
    avg_ms = np.mean(decode_times) * 1000 if decode_times else 0
    tok_per_sec = 1000.0 / avg_ms if avg_ms > 0 else 0

    timing = {
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "generated_tokens": n_gen,
        "prefill_ms": round(t_prefill * 1000, 1),
        "decode_total_ms": round(t_decode_total * 1000, 1),
        "avg_decode_ms_per_token": round(avg_ms, 2),
        "tokens_per_sec": round(tok_per_sec, 1),
    }

    return generated_text, timing


# ══════════════════════════════════════════════════════════════
# HTTP Server
# ══════════════════════════════════════════════════════════════
class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *a):
        # Compact logging
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {format % a}\n")

    def _send_json(self, code, obj):
        body = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ready", "model": "Qwen2.5-0.5B", "device": "Blackhole P150"})
        else:
            self._send_json(404, {"error": "Not found. Try POST /generate or GET /health"})

    def do_POST(self):
        global request_count
        if self.path != "/generate":
            self._send_json(404, {"error": "Not found. Try POST /generate"})
            return

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "Empty request body"})
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        prompt = body.get("prompt", "")
        if not prompt:
            self._send_json(400, {"error": "Missing 'prompt' field"})
            return

        max_tokens = min(body.get("max_tokens", 100), MAX_SEQ - 10)
        temperature = body.get("temperature", 0.7)
        top_k = body.get("top_k", 50)
        seed = body.get("seed", None)

        # Serialize generation (device is single-user)
        acquired = generate_lock.acquire(timeout=30)
        if not acquired:
            self._send_json(503, {"error": "Server busy, try again"})
            return

        try:
            request_count += 1
            req_id = request_count
            print(f"\n[Request {req_id}] prompt={prompt!r:.60s} max_tokens={max_tokens} temp={temperature}")

            t_total_start = time.perf_counter()
            text, timing = generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                    top_k=top_k, seed=seed)
            t_total = time.perf_counter() - t_total_start

            print(f"[Request {req_id}] {timing['generated_tokens']} tokens in {t_total*1000:.0f}ms "
                  f"({timing['tokens_per_sec']:.1f} tok/sec)")

            self._send_json(200, {
                "text": text,
                "prompt": prompt,
                "timing": timing,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json(500, {"error": str(e)})
        finally:
            generate_lock.release()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", args.port), ChatHandler)
    print(f"\n{'='*60}")
    print(f"Chat server ready on http://0.0.0.0:{args.port}")
    print(f"  POST /generate  - Generate text")
    print(f"  GET  /health    - Health check")
    print(f"  Model: Qwen2.5-0.5B | Device: Blackhole P150")
    print(f"  Max sequence: {MAX_SEQ} tokens")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
        ttnn.close_device(device)
        print("Device closed. Goodbye!")
