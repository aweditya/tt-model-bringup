#!/usr/bin/env python3
"""Qwen2.5-0.5B layer-by-layer residual snapshot — CPU vs TT.

Investigation A: localize WHERE the TT-PJRT run diverges from JAX-CPU.
Same code as experiments/jax_qwen05b_pjrt.py, but with an instrumented
decode_step that returns the residual stream AFTER attention AND AFTER
MLP at every layer.

Workflow (two-pass):
  # 1. CPU run — ground truth (fp32). Writes residuals to .cache/qwen05b_residuals_cpu.npz
  python3 experiments/qwen05b_layer_debug.py --mode cpu

  # 2. TT run — bf16 on device. Reads CPU snapshot, computes per-layer cosine.
  TT_PJRT_USE_DEVICE=1 python3 experiments/qwen05b_layer_debug.py --mode tt

Or, in a single invocation (just runs CPU then TT and compares):
  python3 experiments/qwen05b_layer_debug.py --mode both

The decode step we time is the FIRST decode after the numpy prefill. The
caches contain positions [0, T_prompt); the new token's K is excluded
from this step's attention (same v0 quirk as jax_qwen05b_pjrt.py).

Output:
  - .cache/qwen05b/residuals_cpu.npz : (N_LAYERS, 2 [post_attn, post_mlp], 1, 1, HIDDEN)
                                       + final_pre_norm, final_post_norm, logits
  - .cache/qwen05b/residuals_tt.npz  : same
  - research/pjrt_layer_by_layer.md  : table written by `--mode compare` mode

We do NOT fix anything — only localize.
"""

import os
import sys
import argparse
import time

import numpy as np


# ============================================================
# CLI — choose CPU, TT, or both
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["cpu", "tt", "compare", "both"],
                    default="both",
                    help="cpu: fp32 ground truth; tt: bf16 device; "
                         "compare: read both npz and tabulate; both: do all")
parser.add_argument("--prompt", default="The capital of France is")
parser.add_argument("--max-seq", type=int, default=128)
parser.add_argument("--no-pjrt-trace", action="store_true",
                    help="Disable PJRT trace cache for TT mode")
args = parser.parse_args()


HERE = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(HERE, ".."))
PJRT_DIR = os.path.join(PROJ, "pjrt_plugin")
CACHE_DIR = os.path.join(PROJ, ".cache", "qwen05b")
os.makedirs(CACHE_DIR, exist_ok=True)
CPU_NPZ = os.path.join(CACHE_DIR, "residuals_cpu.npz")
TT_NPZ = os.path.join(CACHE_DIR, "residuals_tt.npz")


# ============================================================
# Model config — Qwen2.5-0.5B
# ============================================================
HIDDEN = 896
N_Q_HEADS = 14
N_KV_HEADS = 2
HEAD_DIM = 64
HALF_DIM = HEAD_DIM // 2
N_LAYERS = 24
VOCAB = 151936
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0
INTERMEDIATE = 4864
MAX_SEQ = args.max_seq


# ============================================================
# Weight load + numpy prefill (shared between modes)
# ============================================================

def _load_weights():
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import torch  # only for safetensors → numpy

    model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
    raw = {}
    with safe_open(model_path, framework="pt") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k).float().numpy()
    embed_w = raw["model.embed_tokens.weight"]
    final_norm_g = raw["model.norm.weight"]
    lm_head_w = raw.get("lm_head.weight", embed_w).T.copy()
    layers = []
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        L = {k[len(p):]: v for k, v in raw.items() if k.startswith(p)}
        layers.append({
            "ln1_g":  L["input_layernorm.weight"],
            "q_w":    L["self_attn.q_proj.weight"].T,
            "q_b":    L["self_attn.q_proj.bias"],
            "k_w":    L["self_attn.k_proj.weight"].T,
            "k_b":    L["self_attn.k_proj.bias"],
            "v_w":    L["self_attn.v_proj.weight"].T,
            "v_b":    L["self_attn.v_proj.bias"],
            "o_w":    L["self_attn.o_proj.weight"].T,
            "ln2_g":  L["post_attention_layernorm.weight"],
            "gate_w": L["mlp.gate_proj.weight"].T,
            "up_w":   L["mlp.up_proj.weight"].T,
            "down_w": L["mlp.down_proj.weight"].T,
        })
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    return embed_w, final_norm_g, lm_head_w, layers, tok


def _build_R(head_dim):
    half = head_dim // 2
    R = np.zeros((head_dim, head_dim), dtype=np.float32)
    for i in range(half):
        R[i + half, i] = -1.0
        R[i, i + half] = 1.0
    return R


# ============================================================
# Numpy reference (prefill) — produces deterministic kv caches
# ============================================================
def _numpy_prefill(token_ids, embed_w, final_norm_g, lm_head_w, layers):
    """Run prompt through model in fp32 numpy. Populate kv caches.
    Returns (last_logits, k_caches, v_caches, x_after_prompt).
    x_after_prompt is the embedding for the next token (post-argmax).
    """
    freqs = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))

    def _rope_tables(T):
        angles = np.outer(np.arange(T, dtype=np.float32), freqs)
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
        return cos.astype(np.float32), sin.astype(np.float32)

    def _rope(x, cos, sin):
        x1, x2 = x[..., :HALF_DIM], x[..., HALF_DIM:]
        rot = np.concatenate([-x2, x1], axis=-1)
        return x * cos + rot * sin

    def _rms(x, g):
        ms = np.mean(x * x, axis=-1, keepdims=True)
        return x * (1.0 / np.sqrt(ms + RMS_EPS)) * g

    def _silu(x): return x / (1.0 + np.exp(-x))
    def _softmax(x):
        m = np.max(x, axis=-1, keepdims=True)
        e = np.exp(x - m)
        return e / np.sum(e, axis=-1, keepdims=True)

    T = len(token_ids)
    x = embed_w[token_ids].reshape(1, T, HIDDEN)
    cos, sin = _rope_tables(T)
    cos_4d = cos[None, None]
    sin_4d = sin[None, None]
    kc = [np.zeros((1, N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32) for _ in range(N_LAYERS)]
    vc = [np.zeros((1, N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32) for _ in range(N_LAYERS)]
    for i in range(N_LAYERS):
        lw = layers[i]
        h = _rms(x, lw["ln1_g"])
        q = (h @ lw["q_w"] + lw["q_b"]).reshape(1, T, N_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = (h @ lw["k_w"] + lw["k_b"]).reshape(1, T, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = (h @ lw["v_w"] + lw["v_b"]).reshape(1, T, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        q = _rope(q, cos_4d, sin_4d)
        k = _rope(k, cos_4d, sin_4d)
        kc[i][:, :, :T, :] = k
        vc[i][:, :, :T, :] = v
        scale = 1.0 / np.sqrt(HEAD_DIM)
        groups = N_Q_HEADS // N_KV_HEADS
        k_b = np.repeat(k, groups, axis=1)
        v_b = np.repeat(v, groups, axis=1)
        scores = q @ k_b.transpose(0, 1, 3, 2) * scale
        mask = np.triu(np.ones((T, T), dtype=np.float32), k=1) * -1e9
        scores = scores + mask
        probs = _softmax(scores)
        attn = (probs @ v_b).transpose(0, 2, 1, 3).reshape(1, T, HIDDEN)
        o = attn @ lw["o_w"]
        x = x + o
        h2 = _rms(x, lw["ln2_g"])
        g_ = _silu(h2 @ lw["gate_w"])
        u_ = h2 @ lw["up_w"]
        d_ = (g_ * u_) @ lw["down_w"]
        x = x + d_
    x = _rms(x, final_norm_g)
    return (x[:, -1] @ lm_head_w).squeeze(), kc, vc


# ============================================================
# JAX decode step — instrumented to also return per-layer residuals
# ============================================================
def _build_instrumented_decode(weights_packed, R_const, jnp, jax):
    R_jax = jnp.asarray(R_const)
    scale_const = float(1.0 / np.sqrt(HEAD_DIM))
    groups = N_Q_HEADS // N_KV_HEADS

    layer_jax = weights_packed["layers"]
    final_g = weights_packed["final_g"]
    lm_head = weights_packed["lm_head"]

    def step(x, cos, sin, mask, k_caches, v_caches):
        post_attn = []
        post_mlp = []
        for i in range(N_LAYERS):
            lw = layer_jax[i]
            ms = jnp.mean(x * x, axis=-1, keepdims=True)
            h = x * jax.lax.rsqrt(ms + RMS_EPS) * lw["ln1_g"]

            q = h @ lw["q_w"] + lw["q_b"]
            k = h @ lw["k_w"] + lw["k_b"]
            v = h @ lw["v_w"] + lw["v_b"]

            q = q.reshape(1, N_Q_HEADS, 1, HEAD_DIM)
            k = k.reshape(1, N_KV_HEADS, 1, HEAD_DIM)
            v = v.reshape(1, N_KV_HEADS, 1, HEAD_DIM)

            q_rope = q * cos + (q @ R_jax) * sin
            k_rope = k * cos + (k @ R_jax) * sin

            k_full = k_caches[i]
            v_full = v_caches[i]
            k_b = jnp.repeat(k_full, groups, axis=1)
            v_b = jnp.repeat(v_full, groups, axis=1)

            scores = q_rope @ jnp.swapaxes(k_b, -1, -2)
            scores = scores * scale_const
            scores = scores + mask[None, None, None, :]
            probs = jax.nn.softmax(scores, axis=-1)
            attn = probs @ v_b
            attn = attn.transpose(0, 2, 1, 3).reshape(1, 1, HIDDEN)
            o = attn @ lw["o_w"]
            x = x + o
            post_attn.append(x)

            ms2 = jnp.mean(x * x, axis=-1, keepdims=True)
            h2 = x * jax.lax.rsqrt(ms2 + RMS_EPS) * lw["ln2_g"]
            g_ = jax.nn.silu(h2 @ lw["gate_w"])
            u_ = h2 @ lw["up_w"]
            d_ = (g_ * u_) @ lw["down_w"]
            x = x + d_
            post_mlp.append(x)

        pre_norm = x
        ms_f = jnp.mean(x * x, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(ms_f + RMS_EPS) * final_g
        post_norm = x
        logits = x @ lm_head

        # Stack post-attn and post-mlp residuals → (N_LAYERS, 1, 1, HIDDEN)
        post_attn_stack = jnp.stack(post_attn, axis=0).reshape(N_LAYERS, 1, 1, HIDDEN)
        post_mlp_stack = jnp.stack(post_mlp, axis=0).reshape(N_LAYERS, 1, 1, HIDDEN)
        return logits, post_attn_stack, post_mlp_stack, pre_norm, post_norm

    return step


# ============================================================
# Run for one mode (cpu or tt) — produce + save npz
# ============================================================
def _run_one_mode(mode):
    print(f"\n=== mode={mode} ===")
    if mode == "tt":
        os.environ["TT_PJRT_USE_DEVICE"] = "1"
        if args.no_pjrt_trace:
            os.environ["TT_PJRT_NO_TRACE"] = "1"
    else:
        os.environ.pop("TT_PJRT_USE_DEVICE", None)

    sys.path.insert(0, PJRT_DIR)
    import jax
    import jax.numpy as jnp

    if mode == "tt":
        import jax_plugins.tt as tt_plugin
        try:
            tt_plugin.initialize()
        except Exception as e:
            if "ALREADY_EXISTS" not in str(e):
                raise
        devs = jax.devices("tt")
        if not devs:
            print("No TT devices; falling back to CPU")
            target = jax.devices("cpu")[0]
        else:
            target = devs[0]
    else:
        target = jax.devices("cpu")[0]
    print(f"target device: {target}")

    embed_w, final_norm_g, lm_head_w, layers_w, tok = _load_weights()
    R = _build_R(HEAD_DIM)
    token_ids = list(tok.encode(args.prompt))
    print(f"prompt tokens: {len(token_ids)}")

    # numpy prefill (always fp32)
    last_logits, kc_np, vc_np = _numpy_prefill(
        np.array(token_ids), embed_w, final_norm_g, lm_head_w, layers_w
    )
    next_id = int(np.argmax(last_logits))
    print(f"first generated token: {repr(tok.decode([next_id]))}")
    pos = len(token_ids)  # the JAX call computes the residual for this position

    # Build inputs for decode step
    freqs = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    angles = pos * freqs
    cos_h = np.cos(angles).astype(np.float32)
    sin_h = np.sin(angles).astype(np.float32)
    cos_np = np.concatenate([cos_h, cos_h]).reshape(1, 1, 1, HEAD_DIM)
    sin_np = np.concatenate([sin_h, sin_h]).reshape(1, 1, 1, HEAD_DIM)
    mask_np = np.zeros((MAX_SEQ,), dtype=np.float32)
    mask_np[pos + 1:] = -1e9
    x_np = embed_w[next_id:next_id + 1].reshape(1, 1, HIDDEN).astype(np.float32)

    def put(a):
        arr = jnp.asarray(a, dtype=jnp.float32)
        if target is not None:
            arr = jax.device_put(arr, target)
        return arr

    weights_packed = {
        "layers": [{key: put(v) for key, v in lw.items()} for lw in layers_w],
        "final_g": put(final_norm_g),
        "lm_head": put(lm_head_w),
    }

    step_fn = _build_instrumented_decode(weights_packed, R, jnp, jax)
    step_jit = jax.jit(step_fn)

    x_d = put(x_np)
    cos_d = put(cos_np)
    sin_d = put(sin_np)
    mask_d = put(mask_np)
    kc_d = [put(c) for c in kc_np]
    vc_d = [put(c) for c in vc_np]

    t0 = time.perf_counter()
    logits_d, pa_d, pm_d, pre_d, post_d = step_jit(x_d, cos_d, sin_d, mask_d, kc_d, vc_d)
    logits_np_ = np.asarray(jax.device_get(logits_d))
    pa_np = np.asarray(jax.device_get(pa_d))
    pm_np = np.asarray(jax.device_get(pm_d))
    pre_norm_np = np.asarray(jax.device_get(pre_d))
    post_norm_np = np.asarray(jax.device_get(post_d))
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"step elapsed: {elapsed:.0f} ms")
    sampled = int(np.argmax(logits_np_.flatten()))
    print(f"sampled (this step): {repr(tok.decode([sampled]))}")

    out_path = TT_NPZ if mode == "tt" else CPU_NPZ
    np.savez(out_path,
             post_attn=pa_np, post_mlp=pm_np,
             pre_norm=pre_norm_np, post_norm=post_norm_np,
             logits=logits_np_,
             sampled_id=np.array([sampled]),
             first_next_id=np.array([next_id]))
    print(f"wrote {out_path}")
    return sampled


# ============================================================
# Compare CPU vs TT npz → table
# ============================================================
def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _max_abs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _compare():
    if not os.path.isfile(CPU_NPZ) or not os.path.isfile(TT_NPZ):
        print("Missing one of the snapshot files — run --mode cpu and --mode tt first")
        sys.exit(2)
    cpu = np.load(CPU_NPZ)
    tt = np.load(TT_NPZ)
    pa_c, pa_t = cpu["post_attn"], tt["post_attn"]
    pm_c, pm_t = cpu["post_mlp"], tt["post_mlp"]
    print("\nPer-layer cosine + max-abs-error (CPU fp32 vs TT bf16)")
    print(f"{'layer':>5}  {'after_attn':>22}  {'after_mlp':>22}")
    print(f"{'':>5}  {'cos':>10} {'max_abs':>10}  {'cos':>10} {'max_abs':>10}")
    first_drop = None
    rows = []
    for i in range(N_LAYERS):
        ca = _cosine(pa_c[i], pa_t[i])
        ma = _max_abs(pa_c[i], pa_t[i])
        cm = _cosine(pm_c[i], pm_t[i])
        mm = _max_abs(pm_c[i], pm_t[i])
        rows.append((i, ca, ma, cm, mm))
        print(f"{i:>5}  {ca:>10.6f} {ma:>10.4f}  {cm:>10.6f} {mm:>10.4f}")
        if first_drop is None and min(ca, cm) < 0.99:
            first_drop = i
    print()
    print(f"final pre-norm  cos={_cosine(cpu['pre_norm'], tt['pre_norm']):.6f}  "
          f"max_abs={_max_abs(cpu['pre_norm'], tt['pre_norm']):.4f}")
    print(f"final post-norm cos={_cosine(cpu['post_norm'], tt['post_norm']):.6f}  "
          f"max_abs={_max_abs(cpu['post_norm'], tt['post_norm']):.4f}")
    print(f"logits          cos={_cosine(cpu['logits'], tt['logits']):.6f}  "
          f"max_abs={_max_abs(cpu['logits'], tt['logits']):.4f}")
    print(f"CPU argmax: {int(cpu['sampled_id'][0])}, TT argmax: {int(tt['sampled_id'][0])}")
    print(f"first layer where cos<0.99 (either col): {first_drop}")
    return rows, first_drop


# ============================================================
# Entrypoint
# ============================================================

if args.mode == "cpu":
    _run_one_mode("cpu")
elif args.mode == "tt":
    _run_one_mode("tt")
elif args.mode == "compare":
    _compare()
elif args.mode == "both":
    # Sequential in one process is tricky (PJRT state); print instructions.
    print("--mode both is intended for invoking the two modes separately:")
    print("  python3 experiments/qwen05b_layer_debug.py --mode cpu")
    print("  TT_PJRT_USE_DEVICE=1 python3 experiments/qwen05b_layer_debug.py --mode tt")
    print("  python3 experiments/qwen05b_layer_debug.py --mode compare")
