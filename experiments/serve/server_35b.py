#!/usr/bin/env python3
"""Persistent JSON-socket server for Qwen3.6-35B-A3B on qb1 (1,4) mesh.

Wraps the B13 forward logic in a long-running process so we can:
  - Pay the 10s weight load cost once at bootstrap
  - Answer generate_35b requests with persistent fabric + weights
  - Drive long-context correctness probes (needle-haystack)
  - Profile + optimize incrementally

JSON-line wire protocol via `experiments/serve/protocol.py` (same as
server_tp.py). Endpoints:
  - status            → {loaded, n_layers, vocab, mesh_shape, …}
  - generate_35b      → streams tokens; returns final {generated_text, …}
  - reset_state       → clears DN + KV caches (called automatically at
                        start of each generate; expose for explicit reset too)
  - shutdown          → graceful exit

Run via `experiments/serve/scripts/serve_35b.sh start`.
"""
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import protocol as P  # noqa: E402

SNAPSHOT_ROOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.6-35B-A3B" / "snapshots"
SOCK_PATH = PROJECT_ROOT / ".cache" / "server_35b.sock"
LOG_PATH = PROJECT_ROOT / ".cache" / "server_35b.log"

# Model constants (35B-A3B text config)
HIDDEN = 2048
NUM_V_HEADS = 32
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
CONV_KERNEL = 4
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM_ATTN = 256
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS
PARTIAL_ROTARY = 0.25
ROTARY_DIM = int(HEAD_DIM_ATTN * PARTIAL_ROTARY)
NUM_EXPERTS = 256
TOP_K = 8
MOE_INTER = 512
EPS = 1e-6
NCHIPS = 4
VOCAB = 248320
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


# ── Math primitives (carried over from B13) ────────────────────────────
def silu(x): return x * (1.0 / (1.0 + np.exp(-x)))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def qwen35_rms_norm(x, w, eps=EPS):
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * (1.0 + w)


def rms_norm_head(x, w, eps=EPS):
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def load_t(key_to_shard, key):
    with safe_open(key_to_shard[key], framework="pt") as f:
        return f.get_tensor(key).float().numpy()


def load_layer_weights(key_to_shard, layer_idx):
    prefix = f"model.language_model.layers.{layer_idx}."
    return {k[len(prefix):]: load_t(key_to_shard, k)
            for k in key_to_shard if k.startswith(prefix)}


def build_key_to_shard():
    snap = next(SNAPSHOT_ROOT.glob("*"))
    out = {}
    for shard in sorted(snap.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                out[k] = shard
    return out


# ── Forward blocks (carried over from B13) ─────────────────────────────
def dn_layer_forward(h_np, sd, dn_state):
    conv_state_in, recurrent_state_in = dn_state
    in_proj_qkv = sd["linear_attn.in_proj_qkv.weight"]
    in_proj_z = sd["linear_attn.in_proj_z.weight"]
    in_proj_a = sd["linear_attn.in_proj_a.weight"]
    in_proj_b = sd["linear_attn.in_proj_b.weight"]
    conv1d_weight = sd["linear_attn.conv1d.weight"]
    A_log = sd["linear_attn.A_log"]
    dt_bias = sd["linear_attn.dt_bias"]
    norm_weight = sd["linear_attn.norm.weight"]
    out_proj = sd["linear_attn.out_proj.weight"]

    mixed_qkv = h_np @ in_proj_qkv.T
    z = (h_np @ in_proj_z.T).reshape(1, NUM_V_HEADS, HEAD_V_DIM)
    a = (h_np @ in_proj_a.T).reshape(NUM_V_HEADS)
    b = (h_np @ in_proj_b.T).reshape(NUM_V_HEADS)

    new_conv = np.zeros_like(conv_state_in)
    new_conv[:, :, :CONV_KERNEL-1] = conv_state_in[:, :, 1:]
    new_conv[:, :, CONV_KERNEL-1] = mixed_qkv
    conv_out = np.sum(new_conv * conv1d_weight[None, :, 0, :], axis=-1)
    silu_out = silu(conv_out)

    q_flat = silu_out[:, :KEY_DIM]
    k_flat = silu_out[:, KEY_DIM:2*KEY_DIM]
    v_flat = silu_out[:, 2*KEY_DIM:]
    q_h = q_flat.reshape(1, NUM_K_HEADS, HEAD_K_DIM)
    k_h = k_flat.reshape(1, NUM_K_HEADS, HEAD_K_DIM)
    v_h = v_flat.reshape(1, NUM_V_HEADS, HEAD_V_DIM)

    beta = sigmoid(b)
    softplus_ab = np.log1p(np.exp((a + dt_bias).astype(np.float64))).astype(np.float32)
    g_decay = np.exp(-np.exp(A_log) * softplus_ab)

    def l2norm(x, eps=1e-6):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    q_n = l2norm(q_h)
    k_n = l2norm(k_h)
    rep = NUM_V_HEADS // NUM_K_HEADS
    q_rep = np.repeat(q_n, rep, axis=1)
    k_rep = np.repeat(k_n, rep, axis=1)

    scale = 1.0 / np.sqrt(HEAD_K_DIM)
    q_scaled = q_rep * scale
    state = recurrent_state_in.copy()
    g_b = g_decay[None, :, None, None]
    beta_b = beta[None, :, None]
    state = state * g_b
    kv_mem = np.sum(state * k_rep[:, :, :, None], axis=-2)
    delta = (v_h - kv_mem) * beta_b
    state = state + k_rep[:, :, :, None] * delta[:, :, None, :]
    core_attn_out = np.sum(state * q_scaled[:, :, :, None], axis=-2)

    core_flat = core_attn_out.reshape(-1, HEAD_V_DIM)
    z_flat = z.reshape(-1, HEAD_V_DIM)
    var = np.mean(core_flat ** 2, axis=-1, keepdims=True)
    rsqrt = 1.0 / np.sqrt(var + EPS)
    normed = core_flat * rsqrt * norm_weight[None, :]
    silu_z = z_flat * sigmoid(z_flat)
    gated = (normed * silu_z).reshape(1, VALUE_DIM)
    output = gated @ out_proj.T
    return output, new_conv, state


def attn_layer_forward(h_np, sd, kv_cache, cos_hf, sin_hf):
    q_proj = sd["self_attn.q_proj.weight"]
    k_proj = sd["self_attn.k_proj.weight"]
    v_proj = sd["self_attn.v_proj.weight"]
    o_proj = sd["self_attn.o_proj.weight"]
    q_norm_w = sd["self_attn.q_norm.weight"]
    k_norm_w = sd["self_attn.k_norm.weight"]

    q_full = (h_np @ q_proj.T).reshape(1, NUM_Q_HEADS, HEAD_DIM_ATTN * 2)
    q = q_full[..., :HEAD_DIM_ATTN]
    gate_flat = q_full[..., HEAD_DIM_ATTN:].reshape(1, NUM_Q_HEADS * HEAD_DIM_ATTN)

    k_new = (h_np @ k_proj.T).reshape(1, NUM_KV_HEADS, HEAD_DIM_ATTN)
    v_new = (h_np @ v_proj.T).reshape(1, NUM_KV_HEADS, HEAD_DIM_ATTN)
    q = rms_norm_head(q, q_norm_w)
    k_new = rms_norm_head(k_new, k_norm_w)

    q_rot = q[..., :ROTARY_DIM]; q_pass = q[..., ROTARY_DIM:]
    q_rot = q_rot * cos_hf + rotate_half(q_rot) * sin_hf
    q = np.concatenate([q_rot, q_pass], axis=-1)
    k_rot = k_new[..., :ROTARY_DIM]; k_pass = k_new[..., ROTARY_DIM:]
    k_rot = k_rot * cos_hf + rotate_half(k_rot) * sin_hf
    k_new = np.concatenate([k_rot, k_pass], axis=-1)

    if kv_cache is None or kv_cache.get("K") is None or kv_cache["K"].shape[2] == 0:
        K_all = k_new[:, :, None, :]
        V_all = v_new[:, :, None, :]
    else:
        K_all = np.concatenate([kv_cache["K"], k_new[:, :, None, :]], axis=2)
        V_all = np.concatenate([kv_cache["V"], v_new[:, :, None, :]], axis=2)
    new_kv = {"K": K_all, "V": V_all}

    K_rep = np.repeat(K_all, GQA_GROUP, axis=1)
    V_rep = np.repeat(V_all, GQA_GROUP, axis=1)

    scale = 1.0 / np.sqrt(HEAD_DIM_ATTN)
    scores = np.einsum("bhd,bhtd->bht", q, K_rep) * scale
    weights_attn = scores - scores.max(axis=-1, keepdims=True)
    weights_attn = np.exp(weights_attn)
    weights_attn = weights_attn / weights_attn.sum(axis=-1, keepdims=True)
    attn_out = np.einsum("bht,bhtd->bhd", weights_attn, V_rep)
    attn_flat = attn_out.reshape(1, NUM_Q_HEADS * HEAD_DIM_ATTN)
    gated = attn_flat * sigmoid(gate_flat)
    output = gated @ o_proj.T
    return output, new_kv


def moe_layer_forward(h_np, sd):
    router_w = sd["mlp.gate.weight"]
    eg = sd["mlp.experts.gate_up_proj"]
    ed = sd["mlp.experts.down_proj"]
    sg_p = sd["mlp.shared_expert.gate_proj.weight"]
    su_p = sd["mlp.shared_expert.up_proj.weight"]
    sd_p = sd["mlp.shared_expert.down_proj.weight"]
    seg = sd["mlp.shared_expert_gate.weight"]

    logits = h_np @ router_w.T
    lf = logits.astype(np.float64); lf -= lf.max()
    probs = (np.exp(lf) / np.exp(lf).sum(axis=-1, keepdims=True)).astype(np.float32)
    top_k_idxs = np.argsort(probs[0])[-TOP_K:][::-1].copy()
    top_k_vals = probs[0, top_k_idxs].copy()
    weights = top_k_vals / top_k_vals.sum()

    routed = np.zeros((1, HIDDEN), dtype=np.float32)
    for k_idx in range(TOP_K):
        e = int(top_k_idxs[k_idx])
        gate_up = h_np @ eg[e].T
        gate = gate_up[:, :MOE_INTER]; up = gate_up[:, MOE_INTER:]
        mid = silu(gate) * up
        out_e = mid @ ed[e].T
        routed += float(weights[k_idx]) * out_e

    s_gate = h_np @ sg_p.T
    s_up = h_np @ su_p.T
    s_mid = silu(s_gate) * s_up
    shared = s_mid @ sd_p.T
    g_scalar = sigmoid(h_np @ seg.T)
    shared *= g_scalar
    return routed + shared


def layer_forward(h_np, sd, layer_type, dn_state, kv_cache, cos_hf, sin_hf):
    residual_1 = h_np
    input_ln_w = sd["input_layernorm.weight"]
    h_norm_1 = qwen35_rms_norm(h_np, input_ln_w)
    if layer_type == "linear_attention":
        mixer_out, new_conv, new_rec = dn_layer_forward(h_norm_1, sd, dn_state)
        new_dn = (new_conv, new_rec)
        new_kv = kv_cache
    else:
        mixer_out, new_kv = attn_layer_forward(h_norm_1, sd, kv_cache, cos_hf, sin_hf)
        new_dn = dn_state
    h_after_mixer = residual_1 + mixer_out

    residual_2 = h_after_mixer
    post_ln_w = sd["post_attention_layernorm.weight"]
    h_norm_2 = qwen35_rms_norm(h_after_mixer, post_ln_w)
    moe_out = moe_layer_forward(h_norm_2, sd)
    h_final = residual_2 + moe_out
    return h_final, new_dn, new_kv


# ── Persistent server state ────────────────────────────────────────────
class State:
    def __init__(self):
        self.mesh = None
        self.tokenizer = None
        self.text_cfg = None
        self.layer_types = None
        self.embed_w = None
        self.final_norm_w = None
        self.lm_head_w = None
        self.per_layer = None
        self.rotary = None
        self.dn_caches = None
        self.kv_caches = None

    def reset_caches(self):
        n = self.text_cfg.num_hidden_layers
        dn = []
        kv = []
        for L in range(n):
            if self.layer_types[L] == "linear_attention":
                cs = np.zeros((1, CONV_DIM, CONV_KERNEL), dtype=np.float32)
                rs = np.zeros((1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float32)
                dn.append((cs, rs))
                kv.append(None)
            else:
                dn.append(None)
                kv.append(None)
        self.dn_caches = dn
        self.kv_caches = kv


def bootstrap(state, log):
    log("[bootstrap] importing ttnn + opening (1,4) mesh on qb1…")
    import ttnn
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    state.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    log(f"  mesh: {state.mesh}")

    log("[bootstrap] config + tokenizer + rotary…")
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    state.text_cfg = cfg.text_config
    state.text_cfg.dtype = torch.bfloat16
    state.layer_types = list(state.text_cfg.layer_types)
    state.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTextRotaryEmbedding,
    )
    state.rotary = Qwen3_5MoeTextRotaryEmbedding(state.text_cfg)
    state.rotary.eval()
    log(f"  n_layers: {state.text_cfg.num_hidden_layers}")

    log("[bootstrap] loading top-level weights (embed + final_norm + lm_head)…")
    key_to_shard = build_key_to_shard()
    state.embed_w = load_t(key_to_shard, "model.language_model.embed_tokens.weight")
    state.final_norm_w = load_t(key_to_shard, "model.language_model.norm.weight")
    state.lm_head_w = load_t(key_to_shard, "lm_head.weight")
    log(f"  embed {state.embed_w.shape}, lm_head {state.lm_head_w.shape}")

    log("[bootstrap] loading 40 layer weights (~10s)…")
    t0 = time.time()
    state.per_layer = []
    for L in range(state.text_cfg.num_hidden_layers):
        state.per_layer.append(load_layer_weights(key_to_shard, L))
        if (L + 1) % 10 == 0:
            log(f"  layer {L+1}/{state.text_cfg.num_hidden_layers} loaded ({time.time()-t0:.1f}s)")
    log(f"  all-layer wall: {time.time()-t0:.1f}s")

    state.reset_caches()
    log("[bootstrap] ready.")


# ── Forward step (single token through 40 layers) ──────────────────────
def step_forward(state, h_np, pos):
    pos_t = torch.tensor([[pos]], dtype=torch.long)
    with torch.no_grad():
        cos_t, sin_t = state.rotary(torch.zeros(1, 1, HIDDEN, dtype=torch.bfloat16), pos_t)
    cos_hf = cos_t.detach().float().numpy().reshape(1, 1, ROTARY_DIM)
    sin_hf = sin_t.detach().float().numpy().reshape(1, 1, ROTARY_DIM)
    for L in range(state.text_cfg.num_hidden_layers):
        lt = state.layer_types[L]
        h_np, new_dn, new_kv = layer_forward(
            h_np, state.per_layer[L], lt,
            state.dn_caches[L], state.kv_caches[L],
            cos_hf, sin_hf,
        )
        if new_dn is not None:
            state.dn_caches[L] = new_dn
        if new_kv is not None:
            state.kv_caches[L] = new_kv
    return h_np


def logits_from_hidden(state, h_np):
    h_norm = qwen35_rms_norm(h_np, state.final_norm_w)
    return h_norm @ state.lm_head_w.T  # [1, VOCAB]


# ── RPC handlers ───────────────────────────────────────────────────────
def handle_status(state, args):
    return {
        "loaded": state.per_layer is not None,
        "n_layers": state.text_cfg.num_hidden_layers if state.text_cfg else None,
        "vocab": VOCAB,
        "mesh_shape": [1, NCHIPS],
        "current_kv_lens": [c["K"].shape[2] if c else 0 for c in state.kv_caches] if state.kv_caches else [],
    }


def handle_reset_state(state, args):
    state.reset_caches()
    return {"ok": True}


def handle_generate_35b(state, args):
    """Generator function: yields chunk dicts, ends with _final dict."""
    prompt = args.get("prompt")
    if not prompt:
        yield {"_final": True, "error": "missing prompt"}
        return
    max_tokens = int(args.get("max_tokens", 32))

    state.reset_caches()
    prompt_ids = state.tokenizer.encode(prompt)
    stop_on_eos = bool(args.get("stop_on_eos", True))

    # Prefill
    t0 = time.time()
    h = None
    for step, tid in enumerate(prompt_ids):
        h = state.embed_w[tid].reshape(1, HIDDEN).astype(np.float32)
        h = step_forward(state, h, step)
    prefill_ms = (time.time() - t0) * 1000.0

    # Decode loop
    generated_ids = []
    decode_times = []
    pos = len(prompt_ids)
    for _ in range(max_tokens):
        t1 = time.time()
        logits = logits_from_hidden(state, h)
        next_id = int(np.argmax(logits[0]))
        generated_ids.append(next_id)
        # Stream chunk
        chunk_text = state.tokenizer.decode([next_id])
        yield {"token_id": next_id, "token_text": chunk_text}
        # Next forward
        h = state.embed_w[next_id].reshape(1, HIDDEN).astype(np.float32)
        h = step_forward(state, h, pos)
        pos += 1
        decode_times.append(time.time() - t1)
        if stop_on_eos and next_id == state.tokenizer.eos_token_id:
            break

    generated_text = state.tokenizer.decode(generated_ids, skip_special_tokens=True)
    ms_per_tok = (sum(decode_times) / len(decode_times) * 1000.0) if decode_times else float("nan")
    yield {
        "_final": True,
        "n_prompt_tokens": len(prompt_ids),
        "prefill_ms": prefill_ms,
        "n_generated_tokens": len(generated_ids),
        "decode_ms_per_tok": ms_per_tok,
        "tokens_per_sec": 1000.0 / ms_per_tok if ms_per_tok else float("nan"),
        "generated_text": generated_text,
    }


CMD_TABLE = {
    "status": handle_status,
    "reset_state": handle_reset_state,
}
STREAMING_CMD_TABLE = {
    "generate_35b": handle_generate_35b,
}


def serve(state, log):
    SOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SOCK_PATH.exists():
        SOCK_PATH.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(SOCK_PATH))
    sock.listen(8)
    log(f"[serve] listening on {SOCK_PATH}")

    def _shutdown(*_):
        log("[serve] shutdown signal received")
        sock.close()
        if SOCK_PATH.exists():
            SOCK_PATH.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while True:
        try:
            conn, _ = sock.accept()
        except OSError:
            break
        try:
            raw = P.read_line(conn, max_bytes=64 << 20)
            if not raw:
                conn.close()
                continue
            req = P.parse_request(raw)
            log(f"[serve] cmd={req.cmd}")
            if req.cmd == "shutdown":
                conn.sendall(P.pack_result({"ok": True}))
                conn.close()
                _shutdown()
            if req.cmd in CMD_TABLE:
                result = CMD_TABLE[req.cmd](state, req.args)
                conn.sendall(P.pack_result(result))
            elif req.cmd in STREAMING_CMD_TABLE:
                gen = STREAMING_CMD_TABLE[req.cmd](state, req.args)
                for chunk in gen:
                    if chunk.get("_final"):
                        chunk = {k: v for k, v in chunk.items() if k != "_final"}
                        conn.sendall(P.pack_result(chunk))
                        break
                    else:
                        conn.sendall(P.pack_chunk(chunk))
            else:
                conn.sendall(P.pack_error(f"unknown cmd: {req.cmd}"))
        except Exception as e:
            try:
                conn.sendall(P.pack_error(f"{type(e).__name__}: {e}"))
            except Exception:
                pass
        finally:
            conn.close()


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_PATH, "a", buffering=1)

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)
        log_file.write(f"[{ts}] {msg}\n")

    log("=" * 60)
    log("server_35b starting")
    state = State()
    bootstrap(state, log)
    serve(state, log)


if __name__ == "__main__":
    main()
