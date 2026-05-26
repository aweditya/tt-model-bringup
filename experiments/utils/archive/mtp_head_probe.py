"""mtp_head_probe.py - Numpy MTP head feasibility probe for Qwen3.6-27B speculative decoding.

Goal: measure draft-vs-verifier agreement rate. If >=30%, speculative decoding
(Branch D'3) is worth the integration cost.

Approach (Path B in task brief):
  1. Open ttnn device 1 (server holds device 0).
  2. Load full 27B weights onto device 1 (~60-90s).
  3. For each test prompt, greedy-decode K verifier tokens, capturing
     (residual_state_pre_final_norm, chosen_token_id) at each step.
  4. Pure-numpy MTP forward on (hidden_t, embed(token_{t+1})) -> draft token_{t+2}.
  5. Compare draft vs verifier's actual choice for token_{t+2}.

MTP architecture (per DeepSeek V3 reference + Qwen3.6 weight names):
    e_t1 = embed(token_{t+1})                        # [hidden]
    h_n  = RMSNorm(hidden_t, pre_fc_norm_hidden)
    e_n  = RMSNorm(e_t1,     pre_fc_norm_embedding)
    concat = [e_n, h_n]                              # [2*hidden]  (token first per DeepSeek)
    proj   = concat @ fc.T                            # [hidden]
    --- one transformer block ---
    a      = RMSNorm(proj, input_layernorm)
    Q,G    = a @ q_proj.T (split: Q=[N_Q, HD], G=[N_Q, HD])
    K      = a @ k_proj.T  ([N_KV, HD])
    V      = a @ v_proj.T  ([N_KV, HD])
    Q      = RMSNorm(Q, q_norm)
    K      = RMSNorm(K, k_norm)
    apply  partial-RoPE to Q[:, :ROTARY_DIM] and K[:, :ROTARY_DIM]
    GQA-replicate K,V to N_Q heads, single-token attention (no KV history in probe)
    attn   = softmax(Q @ K^T / sqrt(HD)) @ V          # [N_Q, HD]
    attn   = attn * sigmoid(G)
    proj   = proj + (attn.flatten() @ o_proj.T)
    --- MLP ---
    m      = RMSNorm(proj, post_attention_layernorm)
    proj   = proj + (silu(m @ gate_proj.T) * (m @ up_proj.T)) @ down_proj.T
    --- output ---
    out    = RMSNorm(proj, mtp.norm)
    logits = out @ lm_head.T

For a single-step MTP probe with no KV history, K/V is just the current token's
own K/V. This is the same setup MTP would see for the FIRST speculation slot
in a fresh prefill, plus it's what the original DeepSeek MTP recipe documents.
For consistency we don't carry MTP KV across steps - each (hidden_t, token_{t+1})
is treated as an independent prediction.

NOTE on token positions: HF and Qwen3.6 use position_idx aligned with the
absolute index in the prompt+generation stream. For the MTP attention, the
RoPE position used should match the verifier's position for token_{t+1}
(i.e. the next position after t). We use position = t+1.

Output: prints per-prompt token-by-token comparison and final aggregate match rate.

Run on qb1:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/utils/mtp_head_probe.py
"""
import argparse
import gc
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

import ttnn

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.expanduser("~"))

# Load 91f symbols (deltanet_step_ondevice, gated_attn_step_ondevice, mlp_step_ondevice,
# load_layer_weights_all, upload, hifi4)
_spec = importlib.util.spec_from_file_location(
    "_91f", os.path.expanduser("~/tt-xla/experiments/91f_qwen36_27b_full_ondevice.py"))
_91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_91f)
deltanet_step_ondevice = _91f.deltanet_step_ondevice
gated_attn_step_ondevice = _91f.gated_attn_step_ondevice
mlp_step_ondevice = _91f.mlp_step_ondevice
load_layer_weights_all = _91f.load_layer_weights_all
upload = _91f.upload
hifi4 = _91f.hifi4

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6
MAX_POS = 256

DEFAULT_PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "The largest planet in our solar system is",
    "Water boils at",
    "Albert Einstein was born in",
]


# ============================================================
# Pure numpy math primitives (fp32)
# ============================================================

def rms_norm_np(x, weight, eps=EPS):
    """x: [..., D], weight: [D]. Returns x * rsqrt(mean(x²) + eps) * weight.

    Note: caller is responsible for the (1.0 + raw) offset for Qwen3_5RMSNorm
    weights. This function applies weight directly.
    """
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(ms + eps)) * weight).astype(np.float32)


def silu_np(x):
    return (x * (1.0 / (1.0 + np.exp(-x.astype(np.float64))))).astype(np.float32)


def sigmoid_np(x):
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def softmax_np(x, axis=-1):
    x = x.astype(np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


# ============================================================
# MTP weight loading
# ============================================================

def load_mtp_weights():
    """Load all 15 MTP tensors as fp32 numpy. Apply Qwen3_5RMSNorm (1.0 + raw)
    offset to the layer-norm weights (matches 91f's loader)."""
    print("[mtp] loading 15 MTP tensors from safetensors (bf16 -> fp32)…")
    t0 = time.time()
    idx = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx) as f:
        weight_map = json.load(f)["weight_map"]
    mtp_keys = sorted(k for k in weight_map if k.startswith("mtp"))

    # Per Qwen3_5 convention: input_layernorm, post_attention_layernorm,
    # q_norm, k_norm, pre_fc_norm_*, and mtp.norm itself all use (1.0 + raw).
    # The DeepSeek MTP reference applies the SAME RMSNorm class for token_norm,
    # hidden_norm, head_norm. For Qwen3_5 specifically these are stored as
    # (raw) and we add 1.0 at load.
    RMSNORM_1PLUS_KEYS = {
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }

    by_shard = {}
    for k in mtp_keys:
        by_shard.setdefault(weight_map[k], []).append(k)

    raw = {}
    for shard, keys in by_shard.items():
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k).float().numpy()
                if "proj" in k:
                    t = t.T  # HF stores Linear as [out, in]; we use x @ W
                if k == "mtp.fc.weight":
                    t = t.T  # fc is [hidden, 2*hidden]; .T makes it [2*hidden, hidden] for x@W
                if k in RMSNORM_1PLUS_KEYS:
                    t = t + 1.0
                raw[k] = t.astype(np.float32).copy()

    # Re-key into a flat dict with friendly names
    w = {
        "fc":                 raw["mtp.fc.weight"],
        "pre_fc_norm_embed":  raw["mtp.pre_fc_norm_embedding.weight"],
        "pre_fc_norm_hidden": raw["mtp.pre_fc_norm_hidden.weight"],
        "input_layernorm":    raw["mtp.layers.0.input_layernorm.weight"],
        "q_proj":             raw["mtp.layers.0.self_attn.q_proj.weight"],
        "k_proj":             raw["mtp.layers.0.self_attn.k_proj.weight"],
        "v_proj":             raw["mtp.layers.0.self_attn.v_proj.weight"],
        "o_proj":             raw["mtp.layers.0.self_attn.o_proj.weight"],
        "q_norm":             raw["mtp.layers.0.self_attn.q_norm.weight"],
        "k_norm":             raw["mtp.layers.0.self_attn.k_norm.weight"],
        "post_attn_norm":     raw["mtp.layers.0.post_attention_layernorm.weight"],
        "gate_proj":          raw["mtp.layers.0.mlp.gate_proj.weight"],
        "up_proj":            raw["mtp.layers.0.mlp.up_proj.weight"],
        "down_proj":          raw["mtp.layers.0.mlp.down_proj.weight"],
        "final_norm":         raw["mtp.norm.weight"],
    }
    print(f"[mtp] loaded MTP weights in {time.time()-t0:.1f}s "
          f"(fc={w['fc'].shape}, q_proj={w['q_proj'].shape})")
    return w


# ============================================================
# Pure numpy MTP forward
# ============================================================

def mtp_forward_numpy(hidden_t_np, token_t1_emb_np, position_t1, w, lm_head_np,
                       cfg, rotary_dim):
    """Run MTP head forward on (hidden_t, embed(token_{t+1})) -> logits over vocab.

    hidden_t_np:      [hidden]  fp32  - residual stream from verifier at position t,
                                          BEFORE the final RMSNorm (per DeepSeek MTP spec).
    token_t1_emb_np:  [hidden]  fp32  - embedding of token_{t+1} (the actual one chosen
                                          by the verifier).
    position_t1:      int             - absolute position of token_{t+1} (for RoPE).
    w:                dict            - MTP weights (from load_mtp_weights).
    lm_head_np:       [hidden, vocab] fp32 - main model's LM head (shared with MTP).

    Returns: logits [vocab]
    """
    HIDDEN = cfg["hidden"]
    N_Q = cfg["n_q_heads"]
    N_KV = cfg["n_kv_heads"]
    HEAD_DIM = cfg["head_dim"]
    INTER = cfg["intermediate_size"]

    # 1) RMSNorm both inputs
    e_n = rms_norm_np(token_t1_emb_np, w["pre_fc_norm_embed"])  # [hidden]
    h_n = rms_norm_np(hidden_t_np,     w["pre_fc_norm_hidden"]) # [hidden]

    # 2) Concat (token first per DeepSeek reference) and project via fc
    cat = np.concatenate([e_n, h_n], axis=-1)  # [2*hidden]
    proj = cat @ w["fc"]                       # [hidden]

    # 3) Pre-norm for attention
    a = rms_norm_np(proj, w["input_layernorm"])

    # 4) Q + gate (q_proj fuses both), K, V projections
    qg = a @ w["q_proj"]              # [2 * N_Q * HEAD_DIM]
    qg = qg.reshape(N_Q, 2 * HEAD_DIM)
    q  = qg[:, :HEAD_DIM]              # [N_Q, HEAD_DIM]
    g  = qg[:, HEAD_DIM:]              # [N_Q, HEAD_DIM]
    k  = (a @ w["k_proj"]).reshape(N_KV, HEAD_DIM)
    v  = (a @ w["v_proj"]).reshape(N_KV, HEAD_DIM)

    # 5) Per-head RMSNorm of Q and K
    q = rms_norm_np(q, w["q_norm"])
    k = rms_norm_np(k, w["k_norm"])

    # 6) Partial RoPE on rotary_dim portion (theta = 10M per Qwen3.6 config)
    half = rotary_dim // 2
    theta = 10_000_000.0
    freqs = 1.0 / (theta ** (np.arange(half, dtype=np.float32) / half))
    angles = position_t1 * freqs
    cos = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)  # [rotary_dim]
    sin = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)

    def rope_apply(t, n_heads):
        rot = t[:, :rotary_dim]                          # [H, rotary_dim]
        passthru = t[:, rotary_dim:]                     # [H, HEAD_DIM - rotary_dim]
        x1 = rot[:, :half]; x2 = rot[:, half:]
        rotated_half = np.concatenate([-x2, x1], axis=-1)
        rot = rot * cos + rotated_half * sin
        return np.concatenate([rot, passthru], axis=-1)

    q = rope_apply(q, N_Q)
    k = rope_apply(k, N_KV)

    # 7) GQA: replicate K,V to N_Q heads
    n_rep = N_Q // N_KV
    k_full = np.repeat(k, n_rep, axis=0)  # [N_Q, HEAD_DIM]
    v_full = np.repeat(v, n_rep, axis=0)  # [N_Q, HEAD_DIM]

    # 8) Single-token attention (no KV history - this is a fresh MTP step).
    # scores = Q @ K^T / sqrt(HEAD_DIM); softmax => trivially 1.0 since cache_len=1.
    # attn = V (with softmax = 1 per row), then gated.
    # We compute it the long way for safety; will reduce to V exactly.
    scale = 1.0 / np.sqrt(HEAD_DIM)
    scores = np.einsum("hd,hd->h", q, k_full)[:, None] * scale  # [N_Q, 1]
    weights = softmax_np(scores, axis=-1)                         # [N_Q, 1] = 1
    attn = (weights[..., None] * v_full[:, None, :]).sum(axis=-2)  # [N_Q, HEAD_DIM]
    # Equivalent to attn = v_full here, but keep general for future KV extension.

    # 9) Sigmoid output gate
    attn = attn * sigmoid_np(g)

    # 10) Output projection + residual
    attn_flat = attn.reshape(-1)                  # [N_Q * HEAD_DIM]
    proj = proj + (attn_flat @ w["o_proj"])

    # 11) MLP block
    m = rms_norm_np(proj, w["post_attn_norm"])
    gate_out = m @ w["gate_proj"]
    up_out   = m @ w["up_proj"]
    inter = silu_np(gate_out) * up_out
    proj = proj + (inter @ w["down_proj"])

    # 12) Final MTP norm + LM head
    out = rms_norm_np(proj, w["final_norm"])
    logits = out @ lm_head_np                     # lm_head_np is [hidden, vocab]
    return logits


# ============================================================
# Verifier setup (full 27B on device 1)
# ============================================================

def load_embed_lm_head():
    """Load embedding + lm_head + final-norm. Mirrors 91l's loader.

    Important: lm_head_np is shaped [hidden, vocab] (the .T of HF's [vocab, hidden]).
    """
    print("[verifier] loading embed + lm_head + final_norm…")
    t0 = time.time()
    idx = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx) as f:
        weight_map = json.load(f)["weight_map"]
    needed = {
        "embed":      "model.language_model.embed_tokens.weight",
        "final_norm": "model.language_model.norm.weight",
        "lm_head":    "lm_head.weight",
    }
    by_shard = {}
    for key, tname in needed.items():
        by_shard.setdefault(weight_map[tname], []).append((key, tname))

    out = {}
    for shard, items in by_shard.items():
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            for key, tname in items:
                t = f.get_tensor(tname).float().numpy()
                if key == "lm_head":
                    t = t.T
                if key == "final_norm":
                    t = t + 1.0  # Qwen3_5RMSNorm
                out[key] = t.copy()
    print(f"[verifier] loaded in {time.time()-t0:.1f}s  "
          f"embed={out['embed'].shape}, lm_head={out['lm_head'].shape}")
    return out


def run_verifier_capture(device, prompt_ids, n_decode_steps, cfg, layer_weights,
                          final_norm_tt, lm_head_tt, ssm_states, conv_states,
                          kv_caches, embed_np):
    """Greedy-decode n_decode_steps starting from prompt_ids. At each generation
    step, capture the post-final-layer hidden state (BEFORE final RMSNorm) and
    the chosen next token.

    Returns:
      generated_ids: list[int] (length n_decode_steps)
      hidden_states_post_norm: list[np.ndarray [hidden]] (length n_decode_steps + len(prompt))
        - one entry per position 0 .. P+n_decode_steps-1
        - hidden_states_post_norm[t] = residual stream after layer 64, before final norm,
                                       when token at position t was input
    """
    HIDDEN = cfg["hidden"]
    NUM_LAYERS = 64
    ROTARY_DIM = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    half_rot = ROTARY_DIM // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))

    def rope_tt(pos):
        angles = pos * freqs
        cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
        sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
        return (upload(cos_np, device, dtype=ttnn.bfloat16),
                upload(sin_np, device, dtype=ttnn.bfloat16))

    def forward_token(token_id, cur_pos):
        """Returns (hidden_post_norm_np [hidden], logits_np [vocab]).

        Per mtp_smoke_hf_hidden.py: MTP expects POST-final-RMSNorm hidden state
        (HF returns this as the last entry of output_hidden_states / last_hidden_state).
        Ranks 2/2/2 with post-norm vs 4/11/2 with pre-norm on the cached prompt.
        """
        x_np = embed_np[token_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        cos_tt, sin_tt = rope_tt(cur_pos)
        cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                       device=device,
                                       layout=ttnn.ROW_MAJOR_LAYOUT)
        dn_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            layer_type, w_tt = layer_weights[i]
            if layer_type == "linear_attention":
                x_tt, ssm_states[dn_idx], conv_states[dn_idx] = deltanet_step_ondevice(
                    x_tt, w_tt, ssm_states[dn_idx], conv_states[dn_idx], cfg)
                dn_idx += 1
            else:
                kv_k, kv_v = kv_caches[attn_idx]
                x_tt, kv_k, kv_v = gated_attn_step_ondevice(
                    x_tt, w_tt, kv_k, kv_v, None, cur_pos_tt, cur_pos,
                    cos_tt, sin_tt, cfg, device)
                kv_caches[attn_idx] = [kv_k, kv_v]
                attn_idx += 1
            x_tt = mlp_step_ondevice(x_tt, w_tt)

        # Final norm => this is the POST-norm hidden state MTP expects.
        x_tt = ttnn.rms_norm(x_tt, weight=final_norm_tt, epsilon=EPS)
        ttnn.synchronize_device(device)
        hidden_post_norm = ttnn.to_torch(x_tt).float().numpy().flatten()[:HIDDEN]

        # LM head
        logits_tt = ttnn.linear(x_tt, lm_head_tt, compute_kernel_config=hifi4)
        ttnn.synchronize_device(device)
        VOCAB = int(lm_head_tt.shape[-1])
        logits = ttnn.to_torch(logits_tt).float().numpy().flatten()[:VOCAB]
        return hidden_post_norm, logits

    hidden_states_post_norm = []
    # Prefill: feed prompt tokens, capture each hidden state (and prefill's last logits)
    P = len(prompt_ids)
    last_logits = None
    for pos, tid in enumerate(prompt_ids):
        h, last_logits = forward_token(tid, pos)
        hidden_states_post_norm.append(h)

    # Decode: greedy
    generated_ids = []
    cur_pos = P
    for step in range(n_decode_steps):
        next_id = int(np.argmax(last_logits))
        generated_ids.append(next_id)
        # Feed this token at cur_pos (so its hidden_state is at index cur_pos)
        h, last_logits = forward_token(next_id, cur_pos)
        hidden_states_post_norm.append(h)
        cur_pos += 1
        if step < 5 or step % 5 == 0:
            print(f"    step {step:2d}: pos={cur_pos-1:2d} tok {next_id}")
    return generated_ids, hidden_states_post_norm


# ============================================================
# Driver
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=1,
                    help="ttnn device id (default: 1, since server uses 0)")
    ap.add_argument("--num-tokens", type=int, default=20,
                    help="How many verifier tokens to greedy-decode per prompt")
    ap.add_argument("--prompts", nargs="+", default=None,
                    help="Override default prompt list")
    ap.add_argument("--out-path", default=os.path.expanduser(
        "~/tt-xla/.cache/mtp_head_probe_results.json"))
    args = ap.parse_args()

    prompts = args.prompts or DEFAULT_PROMPTS
    print("=" * 72)
    print("MTP head probe (numpy MTP vs ttnn verifier)")
    print(f"  device_id={args.device_id}  num_tokens={args.num_tokens}")
    print(f"  prompts ({len(prompts)}):")
    for p in prompts:
        print(f"    {p!r}")
    print("=" * 72)

    # ---------- Config ----------
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        text_cfg = json.load(f)["text_config"]
    cfg = {
        "hidden":      text_cfg["hidden_size"],
        "n_k_heads":   text_cfg["linear_num_key_heads"],
        "n_v_heads":   text_cfg["linear_num_value_heads"],
        "k_dim":       text_cfg["linear_key_head_dim"],
        "v_dim":       text_cfg["linear_value_head_dim"],
        "conv_kernel": text_cfg["linear_conv_kernel_dim"],
        "n_q_heads":   text_cfg["num_attention_heads"],
        "n_kv_heads":  text_cfg["num_key_value_heads"],
        "head_dim":    text_cfg["head_dim"],
        "partial_rotary_factor": text_cfg["partial_rotary_factor"],
        "intermediate_size":     text_cfg["intermediate_size"],
    }
    NUM_LAYERS = text_cfg["num_hidden_layers"]
    HIDDEN = cfg["hidden"]
    ROTARY_DIM = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    CONV_DIM = 2 * cfg["n_k_heads"] * cfg["k_dim"] + cfg["n_v_heads"] * cfg["v_dim"]
    KEY_DIM = cfg["n_k_heads"] * cfg["k_dim"]

    # ---------- Tokenizer ----------
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # ---------- Load MTP weights (host numpy) ----------
    mtp_w = load_mtp_weights()
    embed_lmhead = load_embed_lm_head()
    embed_np = embed_lmhead["embed"]
    lm_head_np = embed_lmhead["lm_head"]            # [hidden, vocab]
    final_norm_np = embed_lmhead["final_norm"]

    # ---------- Device + layer weight upload ----------
    print(f"\n[device] opening device {args.device_id} + loading {NUM_LAYERS} layers (bf8)…")
    t0 = time.time()
    device = ttnn.open_device(device_id=args.device_id)
    final_norm_tt = upload(final_norm_np, device, dtype=ttnn.bfloat16)
    lm_head_tt = upload(lm_head_np, device, dtype=ttnn.bfloat8_b)

    layer_weights = []
    for i in range(NUM_LAYERS):
        layer_type = "linear_attention" if i % 4 != 3 else "full_attention"
        w_np = load_layer_weights_all(i, layer_type)
        w_tt = {}
        for k, arr in w_np.items():
            if k == "conv1d_weight" and arr.ndim == 3:
                arr = arr.squeeze(1)
            dt = ttnn.bfloat8_b if "proj" in k or k == "conv1d_weight" or "gate" in k else ttnn.bfloat16
            w_tt[k] = upload(arr, device, dtype=dt)
        layer_weights.append((layer_type, w_tt))
        del w_np
        gc.collect()
        if i % 16 == 0 or i == NUM_LAYERS - 1:
            print(f"    layer {i:2d}/{NUM_LAYERS-1} ({time.time()-t0:.1f}s)")
    print(f"[device] all {NUM_LAYERS} layers loaded in {time.time()-t0:.1f}s")

    # ---------- Per-prompt loop ----------
    all_results = []
    aggregate_matches = 0
    aggregate_total = 0

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n{'=' * 72}")
        print(f"Prompt {prompt_idx+1}/{len(prompts)}: {prompt!r}")
        print("=" * 72)
        prompt_ids = tok.encode(prompt)
        print(f"  prompt_ids ({len(prompt_ids)}): {prompt_ids}")
        P = len(prompt_ids)
        N = args.num_tokens
        assert P + N <= MAX_POS, f"P+N={P+N} > MAX_POS={MAX_POS}"

        # Reset all device-side recurrent state for this prompt
        print("[reset] fresh ssm/conv states + zero KV caches…")
        n_dn = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
        n_attn = NUM_LAYERS - n_dn
        ssm_states = [
            upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]), dtype=np.float32),
                   device, dtype=ttnn.float32)
            for _ in range(n_dn)
        ]
        conv_states = [
            upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                   device, dtype=ttnn.bfloat16)
            for _ in range(n_dn)
        ]
        kv_caches = []
        kv_zero = np.zeros((1, cfg["n_kv_heads"], MAX_POS, cfg["head_dim"]), dtype=np.float32)
        for _ in range(n_attn):
            kv_k = ttnn.from_torch(torch.from_numpy(kv_zero), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            kv_v = ttnn.from_torch(torch.from_numpy(kv_zero), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            kv_caches.append([kv_k, kv_v])

        # Verifier capture
        t_decode = time.time()
        generated_ids, hidden_states_post_norm = run_verifier_capture(
            device, prompt_ids, N, cfg, layer_weights,
            final_norm_tt, lm_head_tt, ssm_states, conv_states,
            kv_caches, embed_np)
        decode_time = time.time() - t_decode
        print(f"  verifier decode: {decode_time:.1f}s ({decode_time/(N)*1000:.0f} ms/tok)")

        # Free state
        for s in ssm_states: ttnn.deallocate(s)
        for s in conv_states: ttnn.deallocate(s)
        for kv_k, kv_v in kv_caches:
            ttnn.deallocate(kv_k); ttnn.deallocate(kv_v)
        del ssm_states, conv_states, kv_caches
        gc.collect()

        # All tokens in the unrolled stream:
        all_tokens = list(prompt_ids) + generated_ids   # length P + N
        # Number of MTP predictions we can validate: we need (hidden_t, token_{t+1})
        # to predict token_{t+2}. So t can range from 0 .. (P+N-2-1) inclusive,
        # i.e. predict positions 2..(P+N-1). That's P+N-2 predictions max.
        # But the LAST verifier token has no t+1 -- skip.
        prompt_text_tokens = tok.convert_ids_to_tokens(all_tokens)

        matches = 0
        total = 0
        details = []
        for t in range(len(all_tokens) - 2):
            tok_t1 = all_tokens[t + 1]
            tok_t2_actual = all_tokens[t + 2]
            hidden_t = hidden_states_post_norm[t]
            token_t1_emb = embed_np[tok_t1]
            # Position used for MTP RoPE = position of tok_{t+1} = t+1
            logits = mtp_forward_numpy(
                hidden_t, token_t1_emb, position_t1=t + 1,
                w=mtp_w, lm_head_np=lm_head_np, cfg=cfg, rotary_dim=ROTARY_DIM)
            tok_t2_pred = int(np.argmax(logits))
            match = (tok_t2_pred == tok_t2_actual)
            matches += int(match)
            total += 1
            details.append({
                "t": t,
                "tok_t1": tok_t1,
                "tok_t1_str": prompt_text_tokens[t + 1] if t + 1 < len(prompt_text_tokens) else None,
                "actual_t2": tok_t2_actual,
                "actual_t2_str": prompt_text_tokens[t + 2] if t + 2 < len(prompt_text_tokens) else None,
                "pred_t2": tok_t2_pred,
                "pred_t2_str": tok.decode([tok_t2_pred]),
                "match": match,
            })

        rate = matches / max(total, 1)
        print(f"  MTP match: {matches}/{total} = {rate*100:.1f}%")
        # First 10 details
        print(f"  {'t':>3s} {'tok_t1':>20s} {'actual_t2':>20s} {'pred_t2':>20s} {'OK':>3s}")
        for d in details[:15]:
            t_str = (d["tok_t1_str"] or "")[:18]
            a_str = (d["actual_t2_str"] or "")[:18]
            p_str = (d["pred_t2_str"] or "")[:18]
            ok = "Y" if d["match"] else " "
            print(f"  {d['t']:3d} {t_str:>20s} {a_str:>20s} {p_str:>20s} {ok:>3s}")

        all_results.append({
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "generated_ids": generated_ids,
            "match_count": matches,
            "total": total,
            "match_rate": rate,
            "details": details,
        })
        aggregate_matches += matches
        aggregate_total += total

    # ---------- Aggregate ----------
    print("\n" + "=" * 72)
    print("AGGREGATE")
    print("=" * 72)
    for r in all_results:
        print(f"  {r['match_rate']*100:5.1f}%  ({r['match_count']:3d}/{r['total']:3d})  "
              f"{r['prompt']!r}")
    agg_rate = aggregate_matches / max(aggregate_total, 1)
    print("-" * 72)
    print(f"  TOTAL: {aggregate_matches}/{aggregate_total} = {agg_rate*100:.1f}% match rate")

    if agg_rate >= 0.5:
        verdict = "STRONGLY RECOMMEND D'3 - high accept rate suggests ~1.5-2x speedup"
    elif agg_rate >= 0.3:
        verdict = "RECOMMEND D'3 with caveat - modest speedup, integration cost still ~1000 LOC"
    else:
        verdict = "DEFER D'3 - draft quality too low to justify ~1000 LOC integration"
    print(f"  VERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "device_id": args.device_id,
            "num_tokens_per_prompt": args.num_tokens,
            "prompts": prompts,
            "results": all_results,
            "aggregate_match_count": aggregate_matches,
            "aggregate_total": aggregate_total,
            "aggregate_match_rate": agg_rate,
            "verdict": verdict,
        }, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer,)) else o)
    print(f"  results -> {args.out_path}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
