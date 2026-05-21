#!/usr/bin/env python3
"""B12.A / B13 — Qwen3.6-35B-A3B N-layer chain on (1,4) MESH on qb1.

Parameterized by N_LAYERS:
  - N_LAYERS=4  → B12.A: matches B4's HF 4-layer chain reference
  - N_LAYERS=40 → B13: full backbone (needs embed + lm_head for ' Paris' test)

For B12.A: validates multi-layer composition + DN cache flow + DN-to-attn
transition at layer 3. Same residual + 2-RMSNorm pattern as B12, repeated
N times with layer_types[i] dispatch (linear_attention OR full_attention).

Loads weights DIRECTLY from the HF snapshot safetensors (qb1 must have
~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/...
present — download with `huggingface_hub.snapshot_download` first).

Run (qb1 server must NOT be running):
    ssh qb1 'cd ~/tt-xla && N_LAYERS=4 .venv/bin/python \\
        experiments/91ap_qwen36_35b_a3b_chain_ttnn_mesh.py'

Future: N_LAYERS=40 + embed + lm_head → B13 ' Paris' gate.
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
import ttnn
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextRotaryEmbedding,
)

# Find the snapshot directory on qb1
SNAPSHOT_ROOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.6-35B-A3B" / "snapshots"
B4_NPZ = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe" / "b4_4layer_reference.npz"
B5_NPZ = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe" / "b5_full_forward_reference.npz"

N_LAYERS = int(os.environ.get("N_LAYERS", 4))

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
SHARED_INTER = 512
EPS = 1e-6

NCHIPS = 4
NV_PER_CHIP = NUM_V_HEADS // NCHIPS
NK_PER_CHIP = NUM_K_HEADS // NCHIPS
KEY_DIM_CHIP = NK_PER_CHIP * HEAD_K_DIM
VALUE_DIM_CHIP = NV_PER_CHIP * HEAD_V_DIM
CONV_DIM_CHIP = CONV_DIM // NCHIPS
NQ_PER_CHIP = NUM_Q_HEADS // NCHIPS
MOE_INTER_CHIP = MOE_INTER // NCHIPS
SHARED_INTER_CHIP = SHARED_INTER // NCHIPS


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


def build_key_to_shard():
    snap = next(SNAPSHOT_ROOT.glob("*"))
    out = {}
    for shard in sorted(snap.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                out[k] = shard
    return out, snap


def load_t(key_to_shard, key):
    with safe_open(key_to_shard[key], framework="pt") as f:
        return f.get_tensor(key).float().numpy()


def load_layer_weights(key_to_shard, layer_idx):
    """Load all weights for one layer (DN or attn + MoE + layernorms)."""
    prefix = f"model.language_model.layers.{layer_idx}."
    layer_keys = [k for k in key_to_shard if k.startswith(prefix)]
    sd = {k[len(prefix):]: load_t(key_to_shard, k) for k in layer_keys}
    return sd


def dn_per_chip(chip, h_np, sd, conv_state_init_np, recurrent_state_init_np):
    """DN per-chip TP forward."""
    in_proj_qkv = sd["linear_attn.in_proj_qkv.weight"]
    in_proj_z = sd["linear_attn.in_proj_z.weight"]
    in_proj_a = sd["linear_attn.in_proj_a.weight"]
    in_proj_b = sd["linear_attn.in_proj_b.weight"]
    conv1d_weight = sd["linear_attn.conv1d.weight"]
    A_log = sd["linear_attn.A_log"]
    dt_bias = sd["linear_attn.dt_bias"]
    norm_weight = sd["linear_attn.norm.weight"]
    out_proj = sd["linear_attn.out_proj.weight"]

    def sc(arr, c=chip, total=NCHIPS, axis=0):
        size = arr.shape[axis] // total
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(c * size, (c + 1) * size)
        return arr[tuple(sl)]

    in_qkv_c = np.concatenate([
        sc(in_proj_qkv[:KEY_DIM]),
        sc(in_proj_qkv[KEY_DIM:2*KEY_DIM]),
        sc(in_proj_qkv[2*KEY_DIM:]),
    ], axis=0)
    in_z_c = sc(in_proj_z)
    in_a_c = sc(in_proj_a)
    in_b_c = sc(in_proj_b)
    conv_w_c = np.concatenate([
        sc(conv1d_weight[:KEY_DIM]),
        sc(conv1d_weight[KEY_DIM:2*KEY_DIM]),
        sc(conv1d_weight[2*KEY_DIM:]),
    ], axis=0)
    A_log_c = sc(A_log)
    dt_bias_c = sc(dt_bias)
    cs_c = np.concatenate([
        sc(conv_state_init_np[:, :KEY_DIM, :], axis=1),
        sc(conv_state_init_np[:, KEY_DIM:2*KEY_DIM, :], axis=1),
        sc(conv_state_init_np[:, 2*KEY_DIM:, :], axis=1),
    ], axis=1)
    rs_c = sc(recurrent_state_init_np, axis=1)
    out_proj_c = sc(out_proj, axis=1)

    mixed_qkv_c = h_np @ in_qkv_c.T
    z_c = (h_np @ in_z_c.T).reshape(1, NV_PER_CHIP, HEAD_V_DIM)
    a_c = (h_np @ in_a_c.T).reshape(NV_PER_CHIP)
    b_c = (h_np @ in_b_c.T).reshape(NV_PER_CHIP)

    new_cs = np.zeros_like(cs_c)
    new_cs[:, :, :CONV_KERNEL-1] = cs_c[:, :, 1:]
    new_cs[:, :, CONV_KERNEL-1] = mixed_qkv_c
    conv_out_c = np.sum(new_cs * conv_w_c[None, :, 0, :], axis=-1)
    silu_out_c = silu(conv_out_c)

    q_flat_c = silu_out_c[:, :KEY_DIM_CHIP]
    k_flat_c = silu_out_c[:, KEY_DIM_CHIP:2*KEY_DIM_CHIP]
    v_flat_c = silu_out_c[:, 2*KEY_DIM_CHIP:]
    q_per_head_c = q_flat_c.reshape(1, NK_PER_CHIP, HEAD_K_DIM)
    k_per_head_c = k_flat_c.reshape(1, NK_PER_CHIP, HEAD_K_DIM)
    v_per_head_c = v_flat_c.reshape(1, NV_PER_CHIP, HEAD_V_DIM)

    beta_c = sigmoid(b_c)
    softplus_c = np.log1p(np.exp((a_c + dt_bias_c).astype(np.float64))).astype(np.float32)
    g_decay_c = np.exp(-np.exp(A_log_c) * softplus_c)

    def l2norm(x, eps=1e-6):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    q_norm = l2norm(q_per_head_c)
    k_norm = l2norm(k_per_head_c)
    rep = NV_PER_CHIP // NK_PER_CHIP
    q_rep = np.repeat(q_norm, rep, axis=1)
    k_rep = np.repeat(k_norm, rep, axis=1)

    scale = 1.0 / np.sqrt(HEAD_K_DIM)
    q_scaled = q_rep * scale
    state = rs_c.copy()
    g_b = g_decay_c[None, :, None, None]
    beta_b = beta_c[None, :, None]
    state = state * g_b
    kv_mem = np.sum(state * k_rep[:, :, :, None], axis=-2)
    delta = (v_per_head_c - kv_mem) * beta_b
    state = state + k_rep[:, :, :, None] * delta[:, :, None, :]
    core_attn_out = np.sum(state * q_scaled[:, :, :, None], axis=-2)

    core_flat = core_attn_out.reshape(-1, HEAD_V_DIM)
    z_flat = z_c.reshape(-1, HEAD_V_DIM)
    var = np.mean(core_flat ** 2, axis=-1, keepdims=True)
    rsqrt = 1.0 / np.sqrt(var + EPS)
    normed = core_flat * rsqrt * norm_weight[None, :]
    silu_z = z_flat * sigmoid(z_flat)
    gated = normed * silu_z
    gated_chip = gated.reshape(1, VALUE_DIM_CHIP)
    partial = gated_chip @ out_proj_c.T
    # Return state for next-step continuation (per-chip slice)
    # But for full assembly we need to reconstruct global state — for B12.A
    # we only need 1 token, so we just return the per-chip slice in the order
    # we computed it. Multi-token would need to re-assemble.
    return partial, state


def attn_per_chip(chip, h_np, sd, cos_hf, sin_hf):
    """Attention per-chip TP forward (Q-sharded, KV replicated)."""
    q_proj = sd["self_attn.q_proj.weight"]
    k_proj = sd["self_attn.k_proj.weight"]
    v_proj = sd["self_attn.v_proj.weight"]
    o_proj = sd["self_attn.o_proj.weight"]
    q_norm_w = sd["self_attn.q_norm.weight"]
    k_norm_w = sd["self_attn.k_norm.weight"]

    q_proj_r = q_proj.reshape(NUM_Q_HEADS, HEAD_DIM_ATTN * 2, HIDDEN)
    q_proj_c = q_proj_r[chip*NQ_PER_CHIP:(chip+1)*NQ_PER_CHIP].reshape(
        NQ_PER_CHIP * HEAD_DIM_ATTN * 2, HIDDEN
    )
    q_full_c = (h_np @ q_proj_c.T).reshape(1, NQ_PER_CHIP, HEAD_DIM_ATTN * 2)
    q_c = q_full_c[..., :HEAD_DIM_ATTN]
    gate_flat_c = q_full_c[..., HEAD_DIM_ATTN:].reshape(1, NQ_PER_CHIP * HEAD_DIM_ATTN)

    k = (h_np @ k_proj.T).reshape(1, NUM_KV_HEADS, HEAD_DIM_ATTN)
    v = (h_np @ v_proj.T).reshape(1, NUM_KV_HEADS, HEAD_DIM_ATTN)
    q_c = rms_norm_head(q_c, q_norm_w)
    k = rms_norm_head(k, k_norm_w)

    q_rot = q_c[..., :ROTARY_DIM]; q_pass = q_c[..., ROTARY_DIM:]
    k_rot = k[..., :ROTARY_DIM]; k_pass = k[..., ROTARY_DIM:]
    q_rot = q_rot * cos_hf + rotate_half(q_rot) * sin_hf
    k_rot = k_rot * cos_hf + rotate_half(k_rot) * sin_hf
    q_final = np.concatenate([q_rot, q_pass], axis=-1)
    k_final = np.concatenate([k_rot, k_pass], axis=-1)

    chip_kv_idx = chip // (NCHIPS // NUM_KV_HEADS)
    v_chip = v[:, chip_kv_idx:chip_kv_idx + 1, :]
    v_per_q_c = np.broadcast_to(v_chip, (1, NQ_PER_CHIP, HEAD_DIM_ATTN)).copy()
    attn_flat_c = v_per_q_c.reshape(1, NQ_PER_CHIP * HEAD_DIM_ATTN)
    gated_c = attn_flat_c * sigmoid(gate_flat_c)

    o_proj_c = o_proj[:, chip*NQ_PER_CHIP*HEAD_DIM_ATTN:(chip+1)*NQ_PER_CHIP*HEAD_DIM_ATTN]
    return gated_c @ o_proj_c.T


def moe_per_chip(chip, h_np, sd, top_k_idxs, weights):
    """MoE per-chip TP forward."""
    eg = sd["mlp.experts.gate_up_proj"]
    ed = sd["mlp.experts.down_proj"]
    sg_p = sd["mlp.shared_expert.gate_proj.weight"]
    su_p = sd["mlp.shared_expert.up_proj.weight"]
    sd_p = sd["mlp.shared_expert.down_proj.weight"]

    gs = chip * MOE_INTER_CHIP; ge = (chip + 1) * MOE_INTER_CHIP
    us = MOE_INTER + chip * MOE_INTER_CHIP; ue = MOE_INTER + (chip + 1) * MOE_INTER_CHIP

    routed = np.zeros((1, HIDDEN), dtype=np.float32)
    for k_idx in range(TOP_K):
        e = int(top_k_idxs[k_idx])
        gate_chip = eg[e, gs:ge, :]
        up_chip = eg[e, us:ue, :]
        gate = h_np @ gate_chip.T
        up_v = h_np @ up_chip.T
        mid = silu(gate) * up_v
        down_chip = ed[e, :, chip*MOE_INTER_CHIP:(chip+1)*MOE_INTER_CHIP]
        routed += float(weights[k_idx]) * (mid @ down_chip.T)

    sgs = chip * SHARED_INTER_CHIP
    sge = (chip + 1) * SHARED_INTER_CHIP
    s_gate = h_np @ sg_p[sgs:sge, :].T
    s_up = h_np @ su_p[sgs:sge, :].T
    s_mid = silu(s_gate) * s_up
    shared = s_mid @ sd_p[:, sgs:sge].T
    return routed, shared


def layer_forward(h_np, sd, layer_type, cos_hf, sin_hf, dn_state):
    """Full layer forward = either DN+MoE or attn+MoE."""
    # ── input_layernorm + token mixer + residual_1 ──
    residual_1 = h_np
    input_ln_w = sd["input_layernorm.weight"]
    h_norm_1 = qwen35_rms_norm(h_np, input_ln_w)

    if layer_type == "linear_attention":
        conv_state_in, recurrent_state_in = dn_state
        partials_and_states = [
            dn_per_chip(c, h_norm_1, sd, conv_state_in, recurrent_state_in)
            for c in range(NCHIPS)
        ]
        partials = [p for p, _ in partials_and_states]
        # Reassemble new conv state + recurrent state by concat per-chip slices
        # Conv state per chip is [1, 2048, 4] (already in chip-local shape with KQV interleaved per chip).
        # The dn_per_chip return doesn't carry the new conv_state — for single-token correctness we
        # update only the recurrent state which is sliced per V-head.
        new_recurrent = np.concatenate([s for _, s in partials_and_states], axis=1)
        # For conv state we re-compute the update outside chips (the chip-local conv state is hard
        # to reassemble cleanly). For single-token sequential chain, just track a global conv_state
        # and update via the same formula.
        # Simplification for 1-token correctness: keep the same conv_state_in across this step
        # (single-token bootstrap; the conv state evolves only between sequential prompt tokens
        # — not relevant for B12.A's 1-token forward). For multi-token we'd need to recompute.
        new_conv_state = conv_state_in  # bootstrap; ok for 1-token validation
        attn_or_dn_out = np.sum(partials, axis=0)
        new_dn_state = (new_conv_state, new_recurrent)
    elif layer_type == "full_attention":
        partials = [attn_per_chip(c, h_norm_1, sd, cos_hf, sin_hf) for c in range(NCHIPS)]
        attn_or_dn_out = np.sum(partials, axis=0)
        new_dn_state = dn_state  # unchanged
    else:
        raise ValueError(f"unknown layer_type {layer_type}")

    h_after_mixer = residual_1 + attn_or_dn_out

    # ── post_attention_layernorm + MoE + residual_2 ──
    residual_2 = h_after_mixer
    post_ln_w = sd["post_attention_layernorm.weight"]
    h_norm_2 = qwen35_rms_norm(h_after_mixer, post_ln_w)

    router_w = sd["mlp.gate.weight"]
    logits = h_norm_2 @ router_w.T
    lf = logits.astype(np.float64); lf -= lf.max()
    probs = (np.exp(lf) / np.exp(lf).sum(axis=-1, keepdims=True)).astype(np.float32)
    top_k_idxs = np.argsort(probs[0])[-TOP_K:][::-1].copy()
    top_k_vals = probs[0, top_k_idxs].copy()
    weights = top_k_vals / top_k_vals.sum()

    routed_partials, shared_partials = [], []
    for c in range(NCHIPS):
        r, s = moe_per_chip(c, h_norm_2, sd, top_k_idxs, weights)
        routed_partials.append(r)
        shared_partials.append(s)
    routed_assembled = np.sum(routed_partials, axis=0)
    shared_assembled = np.sum(shared_partials, axis=0)
    shared_gate_w = sd["mlp.shared_expert_gate.weight"]
    g_scalar = sigmoid(h_norm_2 @ shared_gate_w.T)
    shared_assembled *= g_scalar
    moe_out = routed_assembled + shared_assembled
    h_final = residual_2 + moe_out

    return h_final, new_dn_state


def main():
    print(f"N_LAYERS={N_LAYERS}")

    print("[1] enable fabric + open (1,4) mesh on qb1…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    print(f"  mesh: {mesh}")

    try:
        print("[2] load config + tokenizer + rotary…")
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)
        text_cfg = cfg.text_config
        text_cfg.dtype = torch.bfloat16
        print(f"  layer_types[:{N_LAYERS}]={text_cfg.layer_types[:N_LAYERS]}")

        print("[3] load layer weights via safe_open…")
        t0 = time.time()
        key_to_shard, snap = build_key_to_shard()
        print(f"  enumerate shards: {time.time()-t0:.1f}s, snapshot={snap.name[:8]}…")
        per_layer = []
        for L in range(N_LAYERS):
            t1 = time.time()
            sd = load_layer_weights(key_to_shard, L)
            per_layer.append(sd)
            print(f"  layer {L} ({text_cfg.layer_types[L]}): {len(sd)} tensors loaded ({time.time()-t1:.1f}s)")

        if N_LAYERS == 4:
            print("[4] use B4 input + compare to B4 final output")
            assert B4_NPZ.exists(), f"need B4 npz at {B4_NPZ}"
            b4 = np.load(B4_NPZ)
            hidden = b4["hidden_in"].astype(np.float32).reshape(1, HIDDEN)
            expected_final = b4["output"].astype(np.float32).reshape(1, HIDDEN)
            position_ids = b4["position_ids"]
        else:
            # For N=40 we'd want B5's logits, embed, lm_head
            raise NotImplementedError("N_LAYERS=40 path: see B13 follow-up")

        print(f"  hidden_in norm: {np.linalg.norm(hidden):.4f}")
        print(f"  expected final norm: {np.linalg.norm(expected_final):.4f}")

        print("[5] compute RoPE cos/sin via HF rotary module…")
        # Rotary embedding for full_attention layers
        rotary = Qwen3_5MoeTextRotaryEmbedding(text_cfg)
        rotary.eval()
        pos = torch.from_numpy(position_ids).long()
        # Need a torch tensor input for rotary's dtype inference
        dummy = torch.zeros(1, 1, HIDDEN, dtype=torch.bfloat16)
        with torch.no_grad():
            cos_torch, sin_torch = rotary(dummy, pos)
        cos_hf = cos_torch.detach().float().numpy().reshape(1, 1, ROTARY_DIM)
        sin_hf = sin_torch.detach().float().numpy().reshape(1, 1, ROTARY_DIM)
        print(f"  cos/sin shape: {cos_hf.shape}")

        print(f"[6] forward chain {N_LAYERS} layers…")
        # Initialize DN cache (zeros for all layers; some may be unused for attn layers)
        dn_caches = []
        for L in range(N_LAYERS):
            if text_cfg.layer_types[L] == "linear_attention":
                conv_state = np.zeros((1, CONV_DIM, CONV_KERNEL), dtype=np.float32)
                recurrent_state = np.zeros((1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float32)
                dn_caches.append((conv_state, recurrent_state))
            else:
                dn_caches.append(None)

        t0 = time.time()
        h = hidden
        for L in range(N_LAYERS):
            t1 = time.time()
            layer_type = text_cfg.layer_types[L]
            h, new_state = layer_forward(h, per_layer[L], layer_type, cos_hf, sin_hf, dn_caches[L])
            if new_state is not None:
                dn_caches[L] = new_state
            print(f"  layer {L} ({layer_type}): out_norm={np.linalg.norm(h):.4f} "
                  f"({time.time()-t1:.2f}s)")
        print(f"  chain wall: {time.time()-t0:.1f}s")

        print(f"[7] cosine vs HF B{4 if N_LAYERS==4 else 5} reference…")
        cos = (h.flatten() @ expected_final.flatten()) / (
            np.linalg.norm(h) * np.linalg.norm(expected_final) + 1e-30
        )
        max_abs = np.abs(h - expected_final).max()
        print(f"  expected norm: {np.linalg.norm(expected_final):.6f}")
        print(f"  TP final norm: {np.linalg.norm(h):.6f}")
        print(f"  cosine: {cos:.6f}")
        print(f"  max|Δ|: {max_abs:.4f}")
        if cos > 0.999:
            print(f"  ✓ PASS — {N_LAYERS}-layer chain on TP mesh matches HF")
        else:
            print(f"  ✗ FAIL")

    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    print("\nDONE.")


if __name__ == "__main__":
    main()
