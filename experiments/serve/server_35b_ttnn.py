#!/usr/bin/env python3
"""B16 — Fully on-device server_35b: ttnn-on-mesh forward with NO host↔device
data movement in the per-token inner loop.

Companion to server_35b.py (numpy-hybrid fallback). Per user's roadmap:
  1. Fully on-device (this) — surfaces correctness bugs immediately
  2. Trace capture (B17) — eliminates dispatch overhead
  3. Custom ops + comm/compute overlap (B18+)

Architecture:
  - Mesh (1,4) opened once at bootstrap
  - All 40 layers' weights uploaded to mesh as ttnn tensors, SHARDED per
    the validated patterns from B10/B11/B12.8:
      DN:  V-head axis split (NV_PER_CHIP=8); replicated A_log/dt_bias/etc.
      attn: Q-head axis split (NQ_PER_CHIP=4); REPLICATED KV
      MoE: intermediate-dim axis split (MOE_INTER_CHIP=128)
  - Layernorm weights pre-multiplied by `1.0` and add `1.0` ahead of upload
    (model uses `output * (1 + weight)` convention; ttnn.rms_norm does
     `output * weight`, so we store `(1 + weight)` as the effective weight)
  - Per-token forward: ttnn ops only; hidden stays on device throughout
  - Argmax on device; only 1 token-id (8 bytes) flows back to host per step

Run via `experiments/serve/scripts/serve_35b_ttnn.sh start`.
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

import ttnn  # noqa: E402

SNAPSHOT_ROOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.6-35B-A3B" / "snapshots"
SOCK_PATH = PROJECT_ROOT / ".cache" / "server_35b_ttnn.sock"
LOG_PATH = PROJECT_ROOT / ".cache" / "server_35b_ttnn.log"

# Model constants
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

ROPE_THETA = 10_000_000.0  # Qwen3.6 rope_scaling.rope_theta (verified via inspector)

NCHIPS = 4
NV_PER_CHIP = NUM_V_HEADS // NCHIPS
NK_PER_CHIP = NUM_K_HEADS // NCHIPS
KEY_DIM_CHIP = NK_PER_CHIP * HEAD_K_DIM
VALUE_DIM_CHIP = NV_PER_CHIP * HEAD_V_DIM
CONV_DIM_CHIP = CONV_DIM // NCHIPS
NQ_PER_CHIP = NUM_Q_HEADS // NCHIPS
MOE_INTER_CHIP = MOE_INTER // NCHIPS

VOCAB = 248320
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MAX_KV = 4096  # max KV cache length; bump later for long context


# 27B-style compute kernel config — HiFi4 + fp32_dest_acc_en=True. Passed to
# every ttnn.matmul/ttnn.linear in the model forward to keep DEST register
# accumulation in fp32 even when inputs/weights are bf16. Mirrors 91f.py's
# discipline (every linear takes compute_kernel_config=hifi4) which is the
# established 27B mixed-precision pattern. 35B's server was missing this
# config on every matmul outside the SDPA call — the dominant cause of
# accumulated bf16 quant noise at late layers (L31/L39).
HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ── Tensor upload helpers ──────────────────────────────────────────────
def np_to_replicated(arr, mesh, dtype=ttnn.bfloat16):
    """Upload a numpy array to mesh, replicated on every chip."""
    return ttnn.from_torch(
        torch.from_numpy(arr.astype(np.float32)),
        dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def np_stacked_to_sharded(per_chip_list, mesh, dtype=ttnn.bfloat16):
    """Upload a list of 4 per-chip numpy arrays as a sharded ttnn tensor."""
    stacked = np.stack(per_chip_list, axis=0)
    return ttnn.from_torch(
        torch.from_numpy(stacked.astype(np.float32)),
        dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def load_t(key_to_shard, key):
    with safe_open(key_to_shard[key], framework="pt") as f:
        return f.get_tensor(key).float().numpy()


def build_key_to_shard():
    snap = next(SNAPSHOT_ROOT.glob("*"))
    out = {}
    for shard in sorted(snap.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                out[k] = shard
    return out


def shard_along(arr, axis, n=NCHIPS):
    """Split arr along given axis into n equal slices, return list."""
    size = arr.shape[axis] // n
    return [np.take(arr, range(i * size, (i + 1) * size), axis=axis) for i in range(n)]


# ── Per-block weight upload ────────────────────────────────────────────
def upload_dn_layer(sd, mesh):
    """Upload DN weights for one layer with V-head sharding."""
    out = {}
    in_proj_qkv = sd["linear_attn.in_proj_qkv.weight"]  # [8192, 2048] = K K V

    # Build per-chip in_proj_qkv: each chip gets its V-head slice of Q + K + V
    per_chip_qkv = []
    for chip in range(NCHIPS):
        q_slice = shard_along(in_proj_qkv[:KEY_DIM], 0)[chip]
        k_slice = shard_along(in_proj_qkv[KEY_DIM:2*KEY_DIM], 0)[chip]
        v_slice = shard_along(in_proj_qkv[2*KEY_DIM:], 0)[chip]
        per_chip_qkv.append(np.concatenate([q_slice, k_slice, v_slice], axis=0).T)  # [2048, 2048]
    out["in_proj_qkv"] = np_stacked_to_sharded(per_chip_qkv, mesh)  # [4, 2048, 2048]

    out["in_proj_z"] = np_stacked_to_sharded(
        [shard_along(sd["linear_attn.in_proj_z.weight"], 0)[c].T for c in range(NCHIPS)], mesh)  # [4, 2048, 1024]
    out["in_proj_a"] = np_stacked_to_sharded(
        [shard_along(sd["linear_attn.in_proj_a.weight"], 0)[c].T for c in range(NCHIPS)], mesh)  # [4, 2048, 8]
    out["in_proj_b"] = np_stacked_to_sharded(
        [shard_along(sd["linear_attn.in_proj_b.weight"], 0)[c].T for c in range(NCHIPS)], mesh)

    # conv1d_weight [8192, 1, 4] — shard along axis 0 with same K|K|V layout
    cw = sd["linear_attn.conv1d.weight"]
    per_chip_cw = []
    for chip in range(NCHIPS):
        k1 = shard_along(cw[:KEY_DIM], 0)[chip]
        k2 = shard_along(cw[KEY_DIM:2*KEY_DIM], 0)[chip]
        vv = shard_along(cw[2*KEY_DIM:], 0)[chip]
        per_chip_cw.append(np.concatenate([k1, k2, vv], axis=0).squeeze(1))  # [2048, 4]
    out["conv1d_weight"] = np_stacked_to_sharded(per_chip_cw, mesh)

    # A_log, dt_bias: per V-head, shard
    out["A_log"] = np_stacked_to_sharded(
        [shard_along(sd["linear_attn.A_log"], 0)[c] for c in range(NCHIPS)], mesh)  # [4, 8]
    out["dt_bias"] = np_stacked_to_sharded(
        [shard_along(sd["linear_attn.dt_bias"], 0)[c] for c in range(NCHIPS)], mesh)

    # norm_weight [128]: replicated (per-head_dim)
    out["norm_weight"] = np_to_replicated(sd["linear_attn.norm.weight"], mesh)

    # out_proj [2048, 4096]: column-sharded along input dim
    out["out_proj"] = np_stacked_to_sharded(
        [shard_along(sd["linear_attn.out_proj.weight"], 1)[c].T for c in range(NCHIPS)], mesh)  # [4, 1024, 2048]
    return out


def upload_attn_layer(sd, mesh):
    """Upload attention weights with Q-head sharding + KV-head-per-chip sharding.

    GQA mapping: chip i → KV head (i // (NCHIPS // NUM_KV_HEADS)).
    For NCHIPS=4, NUM_KV_HEADS=2: chip 0,1 → KV head 0; chip 2,3 → KV head 1.
    Per-chip k_proj/v_proj outputs ONE KV head's K (or V) vector each,
    matching the chip's 4 Q heads (which all attend to that one KV head).
    """
    out = {}
    q_proj = sd["self_attn.q_proj.weight"]  # [Q_HEADS*HEAD_DIM*2=8192, HIDDEN=2048]
    q_proj_r = q_proj.reshape(NUM_Q_HEADS, HEAD_DIM_ATTN * 2, HIDDEN)
    per_chip_q = []
    for chip in range(NCHIPS):
        slc = q_proj_r[chip*NQ_PER_CHIP:(chip+1)*NQ_PER_CHIP].reshape(
            NQ_PER_CHIP * HEAD_DIM_ATTN * 2, HIDDEN
        ).T  # [HIDDEN, NQ_PER_CHIP * HEAD_DIM * 2]
        per_chip_q.append(slc)
    out["q_proj"] = np_stacked_to_sharded(per_chip_q, mesh)

    # K, V projections: shard per chip→KV-head mapping.
    # k_proj weight [NUM_KV_HEADS*HEAD_DIM=512, HIDDEN=2048]
    k_proj = sd["self_attn.k_proj.weight"]
    v_proj = sd["self_attn.v_proj.weight"]
    per_chip_k = []
    per_chip_v = []
    chips_per_kv = NCHIPS // NUM_KV_HEADS  # 2
    for chip in range(NCHIPS):
        chip_kv = chip // chips_per_kv  # 0,0,1,1
        # Take this chip's KV head's rows from W
        k_slice = k_proj[chip_kv * HEAD_DIM_ATTN:(chip_kv + 1) * HEAD_DIM_ATTN, :].T  # [HIDDEN, HEAD_DIM]
        v_slice = v_proj[chip_kv * HEAD_DIM_ATTN:(chip_kv + 1) * HEAD_DIM_ATTN, :].T
        per_chip_k.append(k_slice)
        per_chip_v.append(v_slice)
    out["k_proj"] = np_stacked_to_sharded(per_chip_k, mesh)  # per-chip [HIDDEN, HEAD_DIM]
    out["v_proj"] = np_stacked_to_sharded(per_chip_v, mesh)
    out["q_norm"] = np_to_replicated(sd["self_attn.q_norm.weight"], mesh)
    out["k_norm"] = np_to_replicated(sd["self_attn.k_norm.weight"], mesh)

    # o_proj [HIDDEN=2048, NUM_Q_HEADS*HEAD_DIM=4096]: column-sharded along input dim.
    out["o_proj"] = np_stacked_to_sharded(
        [shard_along(sd["self_attn.o_proj.weight"], 1)[c].T for c in range(NCHIPS)], mesh)
    return out


def upload_moe_layer(sd, mesh):
    """Upload MoE with intermediate-dim sharding per plan §3.2 (C)."""
    out = {}
    # Router weight replicated
    out["router_weight"] = np_to_replicated(sd["mlp.gate.weight"].T, mesh)  # [2048, 256]

    # experts.gate_up_proj [256, 1024, 2048] = [E, gate||up, hidden]
    # Per-chip slice: each chip holds all 256 experts, but its 128-of-512 slice
    # of BOTH gate and up. Per-chip shape: [256_E, 2048_H, 256_I] (TRANSPOSED
    # to [hidden, intermediate] for matmul-friendly layout — h @ W → [1, 256]).
    # experts.gate_up_proj per-chip [256_E, HIDDEN=2048, 2*INTER_CHIP=256] TILE.
    # 3D shape preserves matmul-friendly dims so we can slice [0, e, 0..H, 0..2*I]
    # and use the result directly without flat→reshape round-trip.
    egu = sd["mlp.experts.gate_up_proj"]
    per_chip_egu = []
    for chip in range(NCHIPS):
        gate_slice = egu[:, chip*MOE_INTER_CHIP:(chip+1)*MOE_INTER_CHIP, :]
        up_slice = egu[:, MOE_INTER + chip*MOE_INTER_CHIP:MOE_INTER + (chip+1)*MOE_INTER_CHIP, :]
        stacked = np.concatenate([gate_slice, up_slice], axis=1)  # [256, 256, 2048]
        per_chip_egu.append(stacked.transpose(0, 2, 1))  # [256, 2048, 256] — [E, H, I]
    out["experts_gate_up"] = np_stacked_to_sharded(per_chip_egu, mesh)

    # experts.down_proj [256, 2048, 512]: shard along axis 2 (intermediate).
    # Per-chip [256_E, INTER_CHIP=128, HIDDEN=2048] TILE.
    ed = sd["mlp.experts.down_proj"]
    per_chip_ed = []
    for chip in range(NCHIPS):
        slab = ed[:, :, chip*MOE_INTER_CHIP:(chip+1)*MOE_INTER_CHIP]
        per_chip_ed.append(slab.transpose(0, 2, 1))  # [256, 128, 2048] — [E, I, H]
    out["experts_down"] = np_stacked_to_sharded(per_chip_ed, mesh)

    # shared_expert.gate_proj / up_proj [512, 2048]: row-sharded
    out["shared_gate"] = np_stacked_to_sharded(
        [shard_along(sd["mlp.shared_expert.gate_proj.weight"], 0)[c].T for c in range(NCHIPS)], mesh)
    out["shared_up"] = np_stacked_to_sharded(
        [shard_along(sd["mlp.shared_expert.up_proj.weight"], 0)[c].T for c in range(NCHIPS)], mesh)

    # shared_expert.down_proj [2048, 512]: column-sharded along input
    out["shared_down"] = np_stacked_to_sharded(
        [shard_along(sd["mlp.shared_expert.down_proj.weight"], 1)[c].T for c in range(NCHIPS)], mesh)  # [4, 128, 2048]

    # shared_expert_gate [1, 2048]: replicated
    out["shared_expert_gate"] = np_to_replicated(sd["mlp.shared_expert_gate.weight"].T, mesh)  # [2048, 1]
    return out


# ── Per-block ttnn forward ─────────────────────────────────────────────
# For now: hidden_tt arrives REPLICATED on every chip. Each block returns
# the same shape (replicated). Inside the block, partials are sharded then
# all_reduce'd. We'll evolve to keep state sharded later for perf.

def all_reduce_tt(x_tt, mesh):
    """Sum across mesh axis-1 (4 chips). Sum is the default reduce op."""
    return ttnn.all_reduce(x_tt, cluster_axis=1)


def dn_forward_ttnn(h_tt, w, mesh, dn_state, dn_sub_capture=None):
    """DN block fully on-device. h_tt [1, HIDDEN] replicated.

    Implements:
      mixed_qkv = h @ in_proj_qkv         # [1, CONV_DIM_CHIP] per chip
      conv1d update + silu                  # currently PLACEHOLDER (silu only)
      split q/k/v                           # per-chip [1, NK_PER_CHIP/NV_PER_CHIP, HEAD_DIM]
      beta = sigmoid(b)
      g    = -exp(A_log) * softplus(a + dt_bias)
      l2_norm q, k                          # per head_dim
      repeat 4 → 8 heads (per chip)
      recurrence: state*g; kv = sum(state*k_col, -2); delta = beta*(v-kv);
                  state += k_col * delta_row; out = sum(state*q_col, -2)
      RMSNormGated(out, z)                  # per-head rms_norm * silu(z)
      out_proj                              # column-sharded; all_reduce sums partials

    dn_state: (conv_state_tt [1, CONV_DIM_CHIP, KERNEL] per chip,
               recurrent_state_tt [1, NV_PER_CHIP, K_DIM, V_DIM] per chip)
    Returns (output_tt [1, HIDDEN] replicated, new_conv_state, new_recurrent_state).
    """
    conv_state_in, recurrent_state_in = dn_state

    # === Projections (sharded per-chip outputs) ===
    mixed_qkv = ttnn.matmul(h_tt, w["in_proj_qkv"], compute_kernel_config=HIFI4)  # [1, CONV_DIM_CHIP=2048] per chip
    z = ttnn.matmul(h_tt, w["in_proj_z"], compute_kernel_config=HIFI4)            # [1, V_DIM_CHIP=1024] per chip
    a = ttnn.matmul(h_tt, w["in_proj_a"], compute_kernel_config=HIFI4)            # [1, NV_PER_CHIP=8] per chip
    b = ttnn.matmul(h_tt, w["in_proj_b"], compute_kernel_config=HIFI4)            # [1, NV_PER_CHIP=8] per chip
    if dn_sub_capture is not None:
        # mixed_qkv per-chip [Q_CHIP | K_CHIP | V_CHIP]; reassemble to HF order
        dn_sub_capture["dn_in_proj_qkv"] = _reassemble_qkv_chip_to_hf(
            _ttnn_to_numpy_perchip(mixed_qkv, mesh),
            KEY_DIM_CHIP, VALUE_DIM_CHIP,
        )
        # z/a/b are pure V-head shards → straight concat across chips
        dn_sub_capture["dn_in_proj_z"] = _reassemble_heads_chip_to_hf(
            _ttnn_to_numpy_perchip(z, mesh))
        dn_sub_capture["dn_in_proj_a"] = _reassemble_heads_chip_to_hf(
            _ttnn_to_numpy_perchip(a, mesh))
        dn_sub_capture["dn_in_proj_b"] = _reassemble_heads_chip_to_hf(
            _ttnn_to_numpy_perchip(b, mesh))

    # === Conv1d update + silu — full causal_conv1d_update with state shift ===
    # GatedDeltaNet conv1d: depthwise conv over last `kernel_size=4` timesteps.
    # Algorithm (matches HF's causal_conv1d_update fallback):
    #   1. shift conv_state left by 1: new_state[..., :-1] = old_state[..., 1:]
    #   2. append current input as last slot: new_state[..., -1] = mixed_qkv
    #   3. conv_out = sum(new_state * w_conv, dim=-1)  (no bias for this layer)
    #   4. silu(conv_out)
    # conv_state shape per chip [1, CONV_DIM_CHIP, KERNEL=4]; w_conv same.
    # mesh-aware shape: conv_state may be rank 4 with leading mesh-shard dim
    cs_rank = len(list(conv_state_in.shape))
    cur = ttnn.reshape(mixed_qkv, [1, CONV_DIM_CHIP, 1])
    ttnn.deallocate(mixed_qkv)
    # Slice last KERNEL-1 positions from old state: state[..., 1:KERNEL]
    if cs_rank == 4:
        prior = ttnn.slice(conv_state_in, [0, 0, 0, 1], [1, 1, CONV_DIM_CHIP, CONV_KERNEL])
        prior = ttnn.reshape(prior, [1, CONV_DIM_CHIP, CONV_KERNEL - 1])
    else:
        prior = ttnn.slice(conv_state_in, [0, 0, 1], [1, CONV_DIM_CHIP, CONV_KERNEL])
    # Concat shifted state + current as new state (last slot = current)
    conv_state_new = ttnn.concat([prior, cur], dim=-1)
    ttnn.deallocate(prior); ttnn.deallocate(cur)
    # Conv: pointwise mul with w_conv (handle weight rank like conv_state)
    cw_rank_local = len(list(w["conv1d_weight"].shape))
    if cw_rank_local == 4:
        w_conv = ttnn.reshape(w["conv1d_weight"], [1, CONV_DIM_CHIP, CONV_KERNEL])
    else:
        w_conv = w["conv1d_weight"]
    state_w = ttnn.mul(conv_state_new, w_conv)
    if cw_rank_local == 4:
        ttnn.deallocate(w_conv)
    conv_out_3d = ttnn.sum(state_w, dim=-1, keepdim=True)  # [1, CONV_DIM_CHIP, 1]
    ttnn.deallocate(state_w)
    conv_out = ttnn.reshape(conv_out_3d, [1, CONV_DIM_CHIP])
    ttnn.deallocate(conv_out_3d)
    if dn_sub_capture is not None:
        # HF conv1d hook captures pre-silu output. Reassemble to HF layout.
        dn_sub_capture["dn_conv1d"] = _reassemble_qkv_chip_to_hf(
            _ttnn_to_numpy_perchip(conv_out, mesh),
            KEY_DIM_CHIP, VALUE_DIM_CHIP,
        )
    silu_out = ttnn.silu(conv_out)
    ttnn.deallocate(conv_out)

    # === Split q/k/v from silu_out ===
    # ttnn.matmul output on mesh has rank 3 (with leading batch/mesh padding):
    # per-chip logical shape [1, 1, CONV_DIM_CHIP]. Slice begins/ends must match.
    # Layout per chip: [Q_CHIP=512 | K_CHIP=512 | V_CHIP=1024]
    sr = len(list(silu_out.shape))  # detect rank
    if sr == 3:
        q_flat = ttnn.slice(silu_out, [0, 0, 0], [1, 1, KEY_DIM_CHIP])
        k_flat = ttnn.slice(silu_out, [0, 0, KEY_DIM_CHIP], [1, 1, 2 * KEY_DIM_CHIP])
        v_flat = ttnn.slice(silu_out, [0, 0, 2 * KEY_DIM_CHIP], [1, 1, CONV_DIM_CHIP])
    else:
        q_flat = ttnn.slice(silu_out, [0, 0], [1, KEY_DIM_CHIP])
        k_flat = ttnn.slice(silu_out, [0, KEY_DIM_CHIP], [1, 2 * KEY_DIM_CHIP])
        v_flat = ttnn.slice(silu_out, [0, 2 * KEY_DIM_CHIP], [1, CONV_DIM_CHIP])
    ttnn.deallocate(silu_out)

    # Reshape to per-head: q/k [1, NK_PER_CHIP=4, HEAD_K_DIM=128], v [1, NV_PER_CHIP=8, HEAD_V_DIM=128]
    q_h = ttnn.reshape(q_flat, [1, NK_PER_CHIP, HEAD_K_DIM])
    k_h = ttnn.reshape(k_flat, [1, NK_PER_CHIP, HEAD_K_DIM])
    v_h = ttnn.reshape(v_flat, [1, NV_PER_CHIP, HEAD_V_DIM])

    # L2-normalize Q/K per-head: x / sqrt(sum(x*x, dim=-1, keepdim=True) + eps)
    # Use ttnn.rsqrt to avoid a divide. eps matches HF GatedDeltaNet (1e-6).
    q_sq = ttnn.mul(q_h, q_h)
    q_sumsq = ttnn.sum(q_sq, dim=-1, keepdim=True)
    ttnn.deallocate(q_sq)
    q_inv = ttnn.rsqrt(ttnn.add(q_sumsq, EPS))
    ttnn.deallocate(q_sumsq)
    q_n = ttnn.mul(q_h, q_inv)
    ttnn.deallocate(q_h); ttnn.deallocate(q_inv)

    k_sq = ttnn.mul(k_h, k_h)
    k_sumsq = ttnn.sum(k_sq, dim=-1, keepdim=True)
    ttnn.deallocate(k_sq)
    k_inv = ttnn.rsqrt(ttnn.add(k_sumsq, EPS))
    ttnn.deallocate(k_sumsq)
    k_n = ttnn.mul(k_h, k_inv)
    ttnn.deallocate(k_h); ttnn.deallocate(k_inv)

    # HF Qwen3_5MoeGatedDeltaNet scales query by 1/sqrt(head_k_dim) before
    # the recurrence (standard attention scaling), see torch_chunk_gated_delta_rule
    # line ~264. Without it, our core_attn_out comes out sqrt(128)=11.31x larger
    # than HF's, breaking RMSNormGated downstream.
    q_scale = 1.0 / (HEAD_K_DIM ** 0.5)
    q_n_scaled = ttnn.multiply(q_n, q_scale)
    ttnn.deallocate(q_n)
    q_n = q_n_scaled

    q_h = q_n
    k_h = k_n

    # === beta + g ===
    beta = ttnn.sigmoid(b)  # [1, 8] per chip
    ttnn.deallocate(b)
    # g = -exp(A_log) * softplus(a + dt_bias)
    # A_log per chip [8], dt_bias per chip [8]
    # softplus = log(1 + exp(x)); ttnn has softplus
    a_plus_dt = ttnn.add(a, w["dt_bias"])  # broadcast OK if shapes align
    ttnn.deallocate(a)
    softplus_v = ttnn.softplus(a_plus_dt)
    ttnn.deallocate(a_plus_dt)
    neg_exp_alog = ttnn.neg(ttnn.exp(w["A_log"]))  # [8] per chip
    g_decay = ttnn.exp(ttnn.mul(softplus_v, neg_exp_alog))  # [1, 8] per chip, or [8]
    ttnn.deallocate(softplus_v); ttnn.deallocate(neg_exp_alog)

    # === Recurrence (state update + output query) ===
    # State per chip: [1, NV_PER_CHIP=8, HEAD_K_DIM=128, HEAD_V_DIM=128]
    # q, k per chip: [1, NK_PER_CHIP=4, HEAD_K_DIM=128] — need to broadcast/repeat to 8 heads
    # v per chip: [1, NV_PER_CHIP=8, HEAD_V_DIM=128]
    # g_decay per chip: [1, NV_PER_CHIP=8]
    # beta per chip: [1, NV_PER_CHIP=8]

    # Q/K head broadcast: V_HEADS / K_HEADS = 2× repeat_interleave.
    # ttnn lacks repeat_interleave; emulate via reshape→repeat→reshape so
    # heads come out [q0, q0, q1, q1, q2, q2, q3, q3] (NOT [q0, q1, q2, q3, q0, q1, ...]).
    GQA_REPEAT = NV_PER_CHIP // NK_PER_CHIP  # = 2
    q_4d = ttnn.reshape(q_h, [1, NK_PER_CHIP, 1, HEAD_K_DIM])
    k_4d = ttnn.reshape(k_h, [1, NK_PER_CHIP, 1, HEAD_K_DIM])
    ttnn.deallocate(q_h); ttnn.deallocate(k_h)
    q_rep_4d = ttnn.repeat(q_4d, ttnn.Shape([1, 1, GQA_REPEAT, 1]))
    k_rep_4d = ttnn.repeat(k_4d, ttnn.Shape([1, 1, GQA_REPEAT, 1]))
    ttnn.deallocate(q_4d); ttnn.deallocate(k_4d)
    q_rep = ttnn.reshape(q_rep_4d, [1, NV_PER_CHIP, HEAD_K_DIM])
    k_rep = ttnn.reshape(k_rep_4d, [1, NV_PER_CHIP, HEAD_K_DIM])
    ttnn.deallocate(q_rep_4d); ttnn.deallocate(k_rep_4d)

    # Reshape g_decay [1, 8] → [1, 8, 1, 1] for state broadcast
    g_b = ttnn.reshape(g_decay, [1, NV_PER_CHIP, 1, 1])
    ttnn.deallocate(g_decay)

    # state = state * g
    state = ttnn.mul(recurrent_state_in, g_b)
    ttnn.deallocate(g_b)

    # k_col: [1, 8, 128] → [1, 8, 128, 1]
    k_col = ttnn.reshape(k_rep, [1, NV_PER_CHIP, HEAD_K_DIM, 1])
    # kv_mem = sum(state * k_col, dim=-2)
    state_k = ttnn.mul(state, k_col)
    kv_mem = ttnn.sum(state_k, dim=-2)  # [1, 8, 1, 128] or [1, 8, 128]
    ttnn.deallocate(state_k)
    # kv_mem might be [1, 8, 1, 128]; reshape to match v_h [1, 8, 128]
    kv_mem_3d = ttnn.reshape(kv_mem, [1, NV_PER_CHIP, HEAD_V_DIM])
    ttnn.deallocate(kv_mem)

    # delta = (v - kv_mem) * beta
    v_minus_kv = ttnn.sub(v_h, kv_mem_3d)
    ttnn.deallocate(kv_mem_3d); ttnn.deallocate(v_h)
    beta_b = ttnn.reshape(beta, [1, NV_PER_CHIP, 1])
    ttnn.deallocate(beta)
    delta = ttnn.mul(v_minus_kv, beta_b)
    ttnn.deallocate(v_minus_kv); ttnn.deallocate(beta_b)

    # state += k_col * delta.unsqueeze(-2)
    delta_row = ttnn.reshape(delta, [1, NV_PER_CHIP, 1, HEAD_V_DIM])
    ttnn.deallocate(delta)
    k_delta = ttnn.mul(k_col, delta_row)
    ttnn.deallocate(k_col); ttnn.deallocate(delta_row)
    state_new = ttnn.add(state, k_delta)
    ttnn.deallocate(state); ttnn.deallocate(k_delta)

    # out = sum(state_new * q_col, dim=-2)
    q_col = ttnn.reshape(q_rep, [1, NV_PER_CHIP, HEAD_K_DIM, 1])
    ttnn.deallocate(q_rep)
    state_q = ttnn.mul(state_new, q_col)
    ttnn.deallocate(q_col)
    core_attn_out_4d = ttnn.sum(state_q, dim=-2)  # [1, 8, 1, 128]
    ttnn.deallocate(state_q)
    core_attn_out = ttnn.reshape(core_attn_out_4d, [1, NV_PER_CHIP, HEAD_V_DIM])
    ttnn.deallocate(core_attn_out_4d)

    # === RMSNormGated: (output / sqrt(mean(x^2)) * norm_weight) * silu(z) ===
    # per-head; norm_weight is [HEAD_V_DIM=128] replicated
    # Reshape core_attn_out [1, 8, 128] → [8, 128] for batched per-head norm
    core_2d = ttnn.reshape(core_attn_out, [NV_PER_CHIP, HEAD_V_DIM])
    z_2d = ttnn.reshape(z, [NV_PER_CHIP, HEAD_V_DIM])
    if dn_sub_capture is not None:
        # core_attn_out per-chip [NV_PER_CHIP, HEAD_V_DIM] → reassemble heads
        dn_sub_capture["dn_core_attn_out"] = _reassemble_heads_chip_to_hf(
            _ttnn_to_numpy_perchip(core_2d, mesh))
        dn_sub_capture["dn_norm_gate_z"] = _reassemble_heads_chip_to_hf(
            _ttnn_to_numpy_perchip(z_2d, mesh))
        dn_sub_capture["_debug_core_2d_shape"] = list(core_2d.shape)
        dn_sub_capture["_debug_z_2d_shape"] = list(z_2d.shape)
        dn_sub_capture["_debug_norm_weight_shape"] = list(w["norm_weight"].shape)
    ttnn.deallocate(z)
    # Bug isolation: ttnn.rms_norm(x, weight=w) output cos 0.9582 vs numpy oracle.
    # Try: call ttnn.rms_norm WITHOUT weight, then multiply explicitly.
    normed_raw = ttnn.rms_norm(core_2d, epsilon=EPS)
    ttnn.deallocate(core_2d)
    normed = ttnn.mul(normed_raw, w["norm_weight"])
    ttnn.deallocate(normed_raw)
    silu_z = ttnn.silu(z_2d)
    ttnn.deallocate(z_2d)
    if dn_sub_capture is not None:
        dn_sub_capture["dn_norm_rms_only"] = _reassemble_heads_chip_to_hf(
            _ttnn_to_numpy_perchip(normed, mesh))
        dn_sub_capture["dn_norm_silu_z"] = _reassemble_heads_chip_to_hf(
            _ttnn_to_numpy_perchip(silu_z, mesh))
    gated = ttnn.mul(normed, silu_z)
    ttnn.deallocate(normed); ttnn.deallocate(silu_z)

    # Reshape back to [1, V_DIM_CHIP=1024]
    gated_1d = ttnn.reshape(gated, [1, VALUE_DIM_CHIP])
    ttnn.deallocate(gated)
    if dn_sub_capture is not None:
        # HF norm output: same head layout as z (V-head sharded)
        dn_sub_capture["dn_norm"] = _reassemble_heads_chip_to_hf(
            _ttnn_to_numpy_perchip(gated_1d, mesh))

    # === out_proj column-sharded ===
    partial = ttnn.matmul(gated_1d, w["out_proj"], compute_kernel_config=HIFI4)  # [1, HIDDEN] per chip (partial)
    ttnn.deallocate(gated_1d)
    out = all_reduce_tt(partial, mesh)
    ttnn.deallocate(partial)
    if dn_sub_capture is not None:
        # HF out_proj hook fires on Linear OUTPUT — that's BEFORE any cross-chip
        # reduction in TP. Our post-all_reduce out is the equivalent of HF's
        # full out_proj output (which has no TP partitioning).
        dn_sub_capture["dn_out_proj"] = _ttnn_to_numpy_replicated(out, mesh).reshape(-1)

    # Return out + new recurrent state + new conv state (both evolve per token).
    return out, conv_state_new, state_new


def _apply_partial_rope(x, cos_tt, sin_tt, n_heads):
    """Apply Qwen3.6 partial RoPE to x [n_heads, HEAD_DIM_ATTN]:
       only first ROTARY_DIM dims rotated, rest passthrough.

       x_rot_embed = x_rot * cos + rotate_half(x_rot) * sin
       rotate_half([a, b]) = [-b, a]   (a, b each half of ROTARY_DIM)
    """
    x_rot = ttnn.slice(x, [0, 0], [n_heads, ROTARY_DIM])
    x_pass = ttnn.slice(x, [0, ROTARY_DIM], [n_heads, HEAD_DIM_ATTN])
    half = ROTARY_DIM // 2
    x1 = ttnn.slice(x_rot, [0, 0], [n_heads, half])
    x2 = ttnn.slice(x_rot, [0, half], [n_heads, ROTARY_DIM])
    neg_x2 = ttnn.neg(x2)
    ttnn.deallocate(x2)
    rotated = ttnn.concat([neg_x2, x1], dim=-1)
    ttnn.deallocate(neg_x2); ttnn.deallocate(x1)
    x_rot_cos = ttnn.mul(x_rot, cos_tt)  # broadcast cos [1, R] across [N, R]
    rotated_sin = ttnn.mul(rotated, sin_tt)
    ttnn.deallocate(x_rot); ttnn.deallocate(rotated)
    x_rot_embed = ttnn.add(x_rot_cos, rotated_sin)
    ttnn.deallocate(x_rot_cos); ttnn.deallocate(rotated_sin)
    x_embed = ttnn.concat([x_rot_embed, x_pass], dim=-1)
    ttnn.deallocate(x_rot_embed); ttnn.deallocate(x_pass)
    return x_embed


def attn_forward_ttnn_sdpa(h_tt, w, mesh, cos_tt, sin_tt, state, sub_capture=None):
    """Attention block via ttnn paged_scaled_dot_product_attention_decode + B3 config.

    Per-chip GQA layout: NQ_PER_CHIP=4 Q heads attend to 1 KV head (chip→KV-head
    mapping at upload). Paged KV cache shape [NUM_BLOCKS, 1, BLOCK_SIZE, HEAD_DIM]
    per chip; cur_pos selects the slot via state.cur_pos_buf (int32 [1] device).

    RoPE: keeps the broadcast-to-NQ_PER_CHIP workaround for K from the manual
    path (feedback_qwen36_attn_rope_single_row_ttnn_bug.md). After RoPE, all
    NQ_PER_CHIP K rows are identical (cos/sin are scalar per step), so we slice
    row 0 back to [1, HEAD_DIM] before paged_update_cache.

    Returns (out [1, HIDDEN] replicated, kv_cache_tuple). The kv_cache_tuple is
    the SAME (kc, vc) tensors that came in — paged_update_cache mutates them
    in place, so callers can ignore the returned value.
    """
    L = None  # layer index unknown here; state holds caches per layer, attached at call site
    # The caller (layer_forward_ttnn) passes the per-layer kv_cache via state's
    # kv_caches_tt[L]. Here we receive the kc/vc as a paired tuple via state.
    # By contract, attn_forward_ttnn (the dispatcher) sets state._current_kv_cache
    # to (kc, vc) before calling this function.
    kc, vc = state._current_kv_cache

    # Q/K/V projections (Q-head sharded; K, V per chip = 1 KV head)
    q_full = ttnn.matmul(h_tt, w["q_proj"], compute_kernel_config=HIFI4)
    k = ttnn.matmul(h_tt, w["k_proj"], compute_kernel_config=HIFI4)
    v = ttnn.matmul(h_tt, w["v_proj"], compute_kernel_config=HIFI4)
    if sub_capture is not None:
        # Per-chip q_full: [1, NQ_PER_CHIP * HEAD_DIM_ATTN * 2] = [1, 2048]
        # Reassemble across 4 chips along the Q-head axis (each chip holds 4 of
        # 16 Q heads, with gate-doubled). Full Q+gate shape: [1, NUM_Q_HEADS *
        # HEAD_DIM_ATTN * 2] = [1, 8192] to match HF q_proj output.
        per_chip = ttnn.to_torch(
            q_full, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()  # [NCHIPS, 1, NQ_PER_CHIP * HEAD_DIM_ATTN * 2]
        sub_capture["attn_q_proj_full"] = per_chip.reshape(
            NCHIPS, NQ_PER_CHIP, HEAD_DIM_ATTN * 2
        ).reshape(NUM_Q_HEADS, HEAD_DIM_ATTN * 2).reshape(-1)

    # Split Q + gate per head (HF chunk after view to [..., -1, head_dim*2])
    q_full_2d = ttnn.reshape(q_full, [NQ_PER_CHIP, HEAD_DIM_ATTN * 2])
    ttnn.deallocate(q_full)
    q_h = ttnn.slice(q_full_2d, [0, 0], [NQ_PER_CHIP, HEAD_DIM_ATTN])
    gate_part_per_head = ttnn.slice(q_full_2d, [0, HEAD_DIM_ATTN],
                                     [NQ_PER_CHIP, HEAD_DIM_ATTN * 2])
    ttnn.deallocate(q_full_2d)
    gate_part = ttnn.reshape(gate_part_per_head, [1, NQ_PER_CHIP * HEAD_DIM_ATTN])
    ttnn.deallocate(gate_part_per_head)

    # Q/K rms_norm per HEAD_DIM
    k_h = ttnn.reshape(k, [1, HEAD_DIM_ATTN])
    ttnn.deallocate(k)
    q_n = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    k_n = ttnn.rms_norm(k_h, weight=w["k_norm"], epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h)

    # RoPE: Q rotation works on [4, HEAD_DIM] directly. K's path depends on
    # state.attn_sdpa_no_broadcast — Phase 2 ablation flag for the
    # broadcast workaround.
    q_n_rope = _apply_partial_rope(q_n, cos_tt, sin_tt, NQ_PER_CHIP)
    ttnn.deallocate(q_n); q_n = q_n_rope
    if state.attn_sdpa_no_broadcast:
        # H2 ablation: apply RoPE directly on [1, HEAD_DIM] K. If the ttnn
        # [1, HEAD_DIM] slice/concat bug is still live this will silently
        # corrupt K — drift will collapse, telling us the workaround is
        # still necessary. If clean, broadcast workaround can be removed.
        k_n_single = _apply_partial_rope(k_n, cos_tt, sin_tt, 1)
        ttnn.deallocate(k_n)
    else:
        # Broadcast workaround (commit c5b0012, feedback_qwen36_attn_rope_*).
        # Broadcast K to NQ_PER_CHIP rows so _apply_partial_rope operates on
        # [4, HEAD_DIM] (Q-side path that is known to work). Then dedupe
        # row 0 — all rows are bit-identical post-RoPE because cos/sin are
        # scalar per step.
        k_n_b = ttnn.concat([k_n] * NQ_PER_CHIP, dim=0)
        ttnn.deallocate(k_n)
        k_n_rope = _apply_partial_rope(k_n_b, cos_tt, sin_tt, NQ_PER_CHIP)
        ttnn.deallocate(k_n_b)
        k_n_single = ttnn.slice(k_n_rope, [0, 0], [1, HEAD_DIM_ATTN])
        ttnn.deallocate(k_n_rope)
    v_h = ttnn.reshape(v, [1, HEAD_DIM_ATTN])
    ttnn.deallocate(v)

    # Paged update_cache: write K, V at cur_pos. Input contract is
    # [1, 1, NKV_PER_CHIP=1, HEAD_DIM] padded to TILE_HEIGHT=32 on dim -2,
    # HEIGHT_SHARDED L1.
    def _shard_for_paged_write(t_2d):
        t4d = ttnn.reshape(t_2d, [1, 1, 1, HEAD_DIM_ATTN])
        t_pad = ttnn.pad(t4d, [[0, 0], [0, 0], [0, state.sdpa_block_size - 1], [0, 0]],
                         value=0.0)
        return ttnn.to_memory_config(t_pad, state.paged_write_mem_cfg)
    k_sharded = _shard_for_paged_write(k_n_single)
    v_sharded = _shard_for_paged_write(v_h)
    ttnn.deallocate(k_n_single); ttnn.deallocate(v_h)
    ttnn.experimental.paged_update_cache(
        kc, k_sharded,
        update_idxs_tensor=state.cur_pos_buf,
        page_table=state.page_table_tt,
    )
    ttnn.experimental.paged_update_cache(
        vc, v_sharded,
        update_idxs_tensor=state.cur_pos_buf,
        page_table=state.page_table_tt,
    )
    ttnn.deallocate(k_sharded); ttnn.deallocate(v_sharded)

    # SDPA decode (B3 config + CoreCoord(4,4) program config). Q shape
    # [1, 1, NQ_PER_CHIP, HEAD_DIM] per chip.
    q_for_sdpa = ttnn.reshape(q_n, [1, 1, NQ_PER_CHIP, HEAD_DIM_ATTN])
    ttnn.deallocate(q_n)
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_for_sdpa, kc, vc,
        cur_pos_tensor=state.cur_pos_buf,
        page_table_tensor=state.page_table_tt,
        scale=1.0 / (HEAD_DIM_ATTN ** 0.5),
        program_config=state.paged_sdpa_progcfg,
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )  # [1, 1, NQ_PER_CHIP, HEAD_DIM] per chip
    ttnn.deallocate(q_for_sdpa)
    attn_flat = ttnn.reshape(attn_out, [1, NQ_PER_CHIP * HEAD_DIM_ATTN])
    ttnn.deallocate(attn_out)
    if sub_capture is not None:
        # Per-chip attn_flat: [1, NQ_PER_CHIP * HEAD_DIM] (pre-gate, post-SDPA).
        # Reassemble across 4 chips along Q-head axis to match HF o_proj_input
        # shape [seq, NUM_Q_HEADS * HEAD_DIM] = [seq, 4096].
        per_chip = ttnn.to_torch(
            attn_flat, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()  # [NCHIPS, 1, NQ_PER_CHIP * HEAD_DIM]
        sub_capture["attn_sdpa_out"] = per_chip.reshape(-1)  # [NUM_Q_HEADS * HEAD_DIM]

    # attn_output_gate: attn_out * sigmoid(gate)
    gate_sig = ttnn.sigmoid(gate_part)
    ttnn.deallocate(gate_part)
    gated = ttnn.mul(attn_flat, gate_sig)
    ttnn.deallocate(attn_flat); ttnn.deallocate(gate_sig)
    if sub_capture is not None:
        # Post-gate (= HF o_proj_input). Per-chip [1, NQ_PER_CHIP * HEAD_DIM];
        # reassemble across Q-head axis.
        per_chip = ttnn.to_torch(
            gated, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()
        sub_capture["attn_post_gate"] = per_chip.reshape(-1)

    # o_proj column-sharded + all_reduce (same as manual path)
    partial = ttnn.matmul(gated, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(gated)
    if sub_capture is not None:
        # Per-chip o_proj partial — each chip outputs a partial sum of the same
        # [1, HIDDEN] shape; full = sum across chips. Capture the sum to match
        # the all_reduce final result; then compare to HF o_proj output.
        per_chip = ttnn.to_torch(
            partial, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        ).float().numpy()  # [NCHIPS, 1, HIDDEN]
        sub_capture["attn_o_proj_summed"] = per_chip.sum(axis=0).reshape(-1)
    out = all_reduce_tt(partial, mesh)
    ttnn.deallocate(partial)
    if sub_capture is not None:
        sub_capture["attn_out_post_ar"] = _ttnn_to_numpy_replicated(out, mesh).reshape(-1)
    return out, (kc, vc)


def attn_forward_ttnn(h_tt, w, mesh, cos_tt, sin_tt, kv_cache=None, *, state=None,
                      sub_capture=None):
    """Dispatcher between SDPA and manual attention paths.

    When state is provided and state.attn_mode == "sdpa", routes to
    attn_forward_ttnn_sdpa (with paged KV cache + B3 config). Otherwise
    falls through to the historical manual matmul → softmax → matmul path
    used through B16/B17.

    sub_capture (optional dict): when set AND running SDPA path, the SDPA
    function populates it with reassembled-to-HF-layout numpy arrays for
    diagnostic comparison against HF forward hooks. Manual path ignores it.
    """
    if state is not None and state.attn_mode == "sdpa":
        # kv_cache here is the (kc, vc) tuple from state.kv_caches_tt[L].
        # paged_scaled_dot_product_attention_decode mutates the caches in place;
        # the returned tuple == the input, so the caller's state.kv_caches_tt[L]
        # assignment is a no-op (which is fine).
        state._current_kv_cache = kv_cache
        try:
            return attn_forward_ttnn_sdpa(h_tt, w, mesh, cos_tt, sin_tt, state,
                                          sub_capture=sub_capture)
        finally:
            state._current_kv_cache = None
    return attn_forward_ttnn_manual(h_tt, w, mesh, cos_tt, sin_tt, kv_cache=kv_cache)


def attn_forward_ttnn_manual(h_tt, w, mesh, cos_tt, sin_tt, kv_cache=None):
    """Attention block on-device, single-token decode (T=1).

    Q-head sharded (NQ_PER_CHIP=4 Q heads per chip). K, V each chip holds the
    ONE KV head its 4 Q heads attend to (chip→KV-head mapping at upload).

    kv_cache: None on pos 0, else (k_history_tt [n, HEAD_DIM], v_history_tt [n, HEAD_DIM])
      per chip. Naive concat growth — fine for short context, swap for in-place
      tt-metal kv_cache.update_cache_for_token_ post-correctness.

    RoPE not yet wired (identity placeholder; effective at pos 0 since cos=1, sin=0,
    and at pos≥1 history attention still works approximately because V values differ
    per position even without Q/K position discrimination).

    Returns (out [1, HIDDEN] replicated, new_kv_cache).
    """
    # Q projection (Q-head sharded); per-chip [1, NQ_PER_CHIP * HEAD_DIM * 2]
    q_full = ttnn.matmul(h_tt, w["q_proj"], compute_kernel_config=HIFI4)
    # K, V per chip [1, HEAD_DIM_ATTN] — single KV head per chip (mapping at upload).
    k = ttnn.matmul(h_tt, w["k_proj"], compute_kernel_config=HIFI4)
    v = ttnn.matmul(h_tt, w["v_proj"], compute_kernel_config=HIFI4)

    # Split Q + gate per-head (HF chunk after view to [..., -1, head_dim*2])
    q_full_2d = ttnn.reshape(q_full, [NQ_PER_CHIP, HEAD_DIM_ATTN * 2])
    ttnn.deallocate(q_full)
    q_h = ttnn.slice(q_full_2d, [0, 0], [NQ_PER_CHIP, HEAD_DIM_ATTN])
    gate_part_per_head = ttnn.slice(q_full_2d, [0, HEAD_DIM_ATTN],
                                     [NQ_PER_CHIP, HEAD_DIM_ATTN * 2])
    ttnn.deallocate(q_full_2d)
    gate_part = ttnn.reshape(gate_part_per_head, [1, NQ_PER_CHIP * HEAD_DIM_ATTN])
    ttnn.deallocate(gate_part_per_head)

    # Q/K rms_norm per HEAD_DIM
    k_h = ttnn.reshape(k, [1, HEAD_DIM_ATTN])
    ttnn.deallocate(k)
    q_n = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    k_n = ttnn.rms_norm(k_h, weight=w["k_norm"], epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h)

    # RoPE deferred (see commit ab711e7). NOOP_ROPE=True confirmed to match
    # the no-RoPE baseline pos 1 L39 cos exactly (0.9484); turning on
    # _apply_partial_rope (even with cos=1, sin=0) regresses to 0.81. Bug
    # is inside the slice/concat round-trip when integrated, even though
    # the helper passes isolated micro-probes. Flip ENABLE_ROPE to True
    # when the integration issue is fixed.
    _ = cos_tt; _ = sin_tt
    ENABLE_ROPE = True
    BROADCAST_KV = True  # broadcast K and V both to [NQ_PER_CHIP, HEAD_DIM]
    if ENABLE_ROPE:
        # Q: direct rotation, works in micro-probe and integration
        q_n_rope = _apply_partial_rope(q_n, cos_tt, sin_tt, NQ_PER_CHIP)
        ttnn.deallocate(q_n); q_n = q_n_rope
        # K: broadcast to NQ_PER_CHIP rows BEFORE rotation. Cache will also
        # store [NQ_PER_CHIP, HEAD_DIM] per step. Attention math holds (each
        # Q head attends to identical 4 K copies; softmax splits weight
        # equally; V also broadcast so weighted sum recovers correct result).
        if BROADCAST_KV:
            k_n_b = ttnn.concat([k_n] * NQ_PER_CHIP, dim=0)
            ttnn.deallocate(k_n)
            k_n_rope = _apply_partial_rope(k_n_b, cos_tt, sin_tt, NQ_PER_CHIP)
            ttnn.deallocate(k_n_b)
            k_n = k_n_rope

    # === KV cache evolution ===
    # k_n: [NQ_PER_CHIP, HEAD_DIM] when ENABLE_ROPE+BROADCAST_KV is on,
    #      else [1, HEAD_DIM]
    # v: from matmul, [1, HEAD_DIM]
    v_h = ttnn.reshape(v, [1, HEAD_DIM_ATTN])
    # When BROADCAST_KV is on, also broadcast V to match K's row count so
    # K/V cache shapes stay consistent. Math: 4 identical V copies + 4
    # identical K copies → softmax distributes evenly → weighted sum recovers
    # the single-head attention result.
    BROADCAST_V_LOCAL = (ENABLE_ROPE and BROADCAST_KV)
    if BROADCAST_V_LOCAL:
        v_h_broadcast = ttnn.concat([v_h] * NQ_PER_CHIP, dim=0)
        ttnn.deallocate(v_h); v_h = v_h_broadcast
    if kv_cache is None:
        # pos 0: cache IS current K, V (no concat needed). Use direct refs.
        k_hist = k_n
        v_hist = v_h
    else:
        # pos ≥ 1: concat current to existing history along position dim 0
        k_prev, v_prev = kv_cache
        k_hist = ttnn.concat([k_prev, k_n], dim=0)
        v_hist = ttnn.concat([v_prev, v_h], dim=0)
        ttnn.deallocate(k_prev); ttnn.deallocate(v_prev)
        ttnn.deallocate(k_n); ttnn.deallocate(v_h)
    ttnn.deallocate(v)  # safe after v_h consumed (or aliased as v_hist for pos 0)
    new_kv_cache = (k_hist, v_hist)

    # === Attention: softmax(Q @ K_hist^T / sqrt(d_k)) @ V_hist ===
    # Q [NQ_PER_CHIP=4, HEAD_DIM=256], K/V_hist [hist_len, 256]
    # At pos 0 with hist_len=1, softmax over single value is degenerate (always 1.0
    # mathematically but ttnn impl may differ). For now: compute QK^T via element-wise
    # mul + sum (avoids transpose for the hist_len=1 case), and use ttnn.softmax which
    # should handle len-1 correctly.
    if len(list(k_hist.shape)) <= 2 and k_hist.shape[-2] == 1:
        # pos 0 special-case: hist_len=1, output is just V broadcast per Q head.
        # softmax([scalar]) = 1.0 exactly, so attn_out = V_hist.
        # Broadcast V to NQ_PER_CHIP via concat (matches prior placeholder).
        attn_out = ttnn.concat([v_hist] * NQ_PER_CHIP, dim=0)
        ttnn.deallocate(q_n)
    else:
        k_hist_T = ttnn.transpose(k_hist, -2, -1)
        scores = ttnn.matmul(q_n, k_hist_T, compute_kernel_config=HIFI4)
        ttnn.deallocate(k_hist_T); ttnn.deallocate(q_n)
        scale = 1.0 / (HEAD_DIM_ATTN ** 0.5)
        scores_scaled = ttnn.multiply(scores, scale)
        ttnn.deallocate(scores)
        weights = ttnn.softmax(scores_scaled, dim=-1)
        ttnn.deallocate(scores_scaled)
        attn_out = ttnn.matmul(weights, v_hist, compute_kernel_config=HIFI4)
        ttnn.deallocate(weights)

    attn_flat = ttnn.reshape(attn_out, [1, NQ_PER_CHIP * HEAD_DIM_ATTN])
    ttnn.deallocate(attn_out)

    # attn_output_gate: attn_out * sigmoid(gate)
    gate_sig = ttnn.sigmoid(gate_part)
    ttnn.deallocate(gate_part)
    gated = ttnn.mul(attn_flat, gate_sig)
    ttnn.deallocate(attn_flat); ttnn.deallocate(gate_sig)

    # o_proj column-sharded + all_reduce
    partial = ttnn.matmul(gated, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(gated)
    out = all_reduce_tt(partial, mesh)
    ttnn.deallocate(partial)
    return out, new_kv_cache


def layer_forward_ttnn(h_tt, w, layer_type, mesh, cos_tt, sin_tt, dn_state, kv_cache,
                       sub_capture=None, *, state=None):
    """Full decoder layer forward on-device:
      residual = h
      h = input_layernorm(h)          # rms_norm with pre-(1+w) weight
      mixer = DN(h) OR attn(h)
      h = residual + mixer
      residual = h
      h = post_attention_layernorm(h)
      moe = MoE(h)
      h = residual + moe
      return h, new_dn_state, new_kv_cache

    sub_capture (debug only): dict to fill with intermediate hidden states
    {"in_norm", "mixer_out", "after_mixer", "post_attn_norm", "moe_out"} as
    numpy arrays (one chip's view).
    """
    residual_1 = h_tt
    h_norm_1 = ttnn.rms_norm(h_tt, weight=w["input_layernorm"], epsilon=EPS)
    if sub_capture is not None:
        sub_capture["in_norm"] = _ttnn_to_numpy_replicated(h_norm_1, mesh).reshape(-1)
    if layer_type == "linear_attention":
        dn_sc = sub_capture.setdefault("dn_sub", {}) if sub_capture is not None else None
        mixer_out, new_conv, new_rec = dn_forward_ttnn(
            h_norm_1, w, mesh, dn_state, dn_sub_capture=dn_sc,
        )
        new_dn = (new_conv, new_rec)
        new_kv = kv_cache
    else:
        attn_sc = sub_capture.setdefault("attn_sub", {}) if sub_capture is not None else None
        mixer_out, new_kv = attn_forward_ttnn(h_norm_1, w, mesh, cos_tt, sin_tt, kv_cache,
                                              state=state, sub_capture=attn_sc)
        new_dn = dn_state
    if sub_capture is not None:
        sub_capture["mixer_out"] = _ttnn_to_numpy_replicated(mixer_out, mesh).reshape(-1)
    ttnn.deallocate(h_norm_1)
    # Residual-add 1. Optionally upcast to fp32 to avoid bf16 quantization
    # noise at large magnitudes (see feedback_35b_a3b_drift_is_user_facing.md).
    use_fp32_add = state is not None and getattr(state, "residual_add_dtype", "bf16") == "fp32"
    if use_fp32_add:
        r1_fp32 = ttnn.typecast(residual_1, ttnn.float32)
        m_fp32 = ttnn.typecast(mixer_out, ttnn.float32)
        h_after_mixer_fp32 = ttnn.add(r1_fp32, m_fp32)
        ttnn.deallocate(r1_fp32); ttnn.deallocate(m_fp32)
        h_after_mixer = ttnn.typecast(h_after_mixer_fp32, ttnn.bfloat16)
        ttnn.deallocate(h_after_mixer_fp32)
    else:
        h_after_mixer = ttnn.add(residual_1, mixer_out)
    ttnn.deallocate(mixer_out)
    if sub_capture is not None:
        sub_capture["after_mixer"] = _ttnn_to_numpy_replicated(h_after_mixer, mesh).reshape(-1)

    residual_2 = h_after_mixer
    h_norm_2 = ttnn.rms_norm(h_after_mixer, weight=w["post_attention_layernorm"], epsilon=EPS)
    if sub_capture is not None:
        sub_capture["post_attn_norm"] = _ttnn_to_numpy_replicated(h_norm_2, mesh).reshape(-1)
    moe_sc = sub_capture.setdefault("moe_sub", {}) if sub_capture is not None else None
    moe_out = moe_forward_ttnn(h_norm_2, w, mesh, sub_capture=moe_sc)
    ttnn.deallocate(h_norm_2)
    if sub_capture is not None:
        sub_capture["moe_out"] = _ttnn_to_numpy_replicated(moe_out, mesh).reshape(-1)
    # Residual-add 2. Same fp32-upcast option as residual-add 1.
    if use_fp32_add:
        r2_fp32 = ttnn.typecast(residual_2, ttnn.float32)
        mo_fp32 = ttnn.typecast(moe_out, ttnn.float32)
        h_final_fp32 = ttnn.add(r2_fp32, mo_fp32)
        ttnn.deallocate(r2_fp32); ttnn.deallocate(mo_fp32)
        h_final = ttnn.typecast(h_final_fp32, ttnn.bfloat16)
        ttnn.deallocate(h_final_fp32)
    else:
        h_final = ttnn.add(residual_2, moe_out)
    ttnn.deallocate(residual_2); ttnn.deallocate(moe_out)
    return h_final, new_dn, new_kv


def moe_forward_ttnn(h_tt, w, mesh, sub_capture=None):
    """MoE block fully on-device. h_tt [1, 2048] replicated. Returns [1, 2048] replicated.

    sub_capture (optional dict): when set, populates with key MoE intermediates
    for drift-attribution comparison vs HF. Keys: top_idxs, top_weights,
    routed_full, shared_full, final.
    """
    # Router (replicated weight, replicated input → replicated output)
    logits = ttnn.matmul(h_tt, w["router_weight"], compute_kernel_config=HIFI4)  # [1, 256]
    probs = ttnn.softmax(logits, dim=-1)
    # topk
    top_vals, top_idxs = ttnn.topk(probs, k=TOP_K, dim=-1)
    # Renormalize: top_vals / sum(top_vals)
    sum_v = ttnn.sum(top_vals, dim=-1, keepdim=True)
    weights = ttnn.div(top_vals, sum_v)

    # NOTE: ttnn.embedding can gather rows from a weight table given a uint32 index
    # tensor. For per-expert weight gather: experts_gate_up has shape [256, 256, 2048]
    # per chip. We need each of the K=8 selected experts' [256, 2048] slab.
    # Approach: ttnn.embedding(top_idxs, experts_gate_up) → [1, K=8, 256, 2048]
    # Then per-expert matmul against h_tt.
    # FIRST PASS: Python loop over K to avoid complex gather; each iter does
    # ttnn.embedding(single_idx, table) → [1, 256, 2048] then matmul.

    # B17-B v3: on-device expert dispatch. Don't FLATTEN experts (caused L1
    # overflow). Instead: read top_idxs / weights via host readback (same as
    # original) so trace doesn't apply here, but Python loop body uses the
    # SAME slice pattern as before. This restores the original (working)
    # MoE path. Trace support deferred — see research/b17_trace_handoff.
    top_idxs_host = ttnn.to_torch(
        top_idxs, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    ).int().numpy()[0].reshape(-1)
    weights_host = ttnn.to_torch(
        weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    ).float().numpy()[0].reshape(-1)
    if sub_capture is not None:
        sub_capture["moe_top_idxs"] = top_idxs_host.astype(np.int32).copy()
        sub_capture["moe_top_weights"] = weights_host.astype(np.float32).copy()

    routed_partial = None
    for k_idx in range(TOP_K):
        e = int(top_idxs_host[k_idx])
        w_scalar = float(weights_host[k_idx])
        gate_up_e = ttnn.slice(
            w["experts_gate_up"],
            [0, e, 0, 0],
            [1, e + 1, HIDDEN, 2 * MOE_INTER_CHIP],
        )
        gate_up_e_w = ttnn.reshape(gate_up_e, [HIDDEN, 2 * MOE_INTER_CHIP])
        gate_up = ttnn.matmul(h_tt, gate_up_e_w, compute_kernel_config=HIFI4)
        ttnn.deallocate(gate_up_e); ttnn.deallocate(gate_up_e_w)
        gu_rank = len(list(gate_up.shape))
        if gu_rank == 3:
            gate_part = ttnn.slice(gate_up, [0, 0, 0], [1, 1, MOE_INTER_CHIP])
            up_part = ttnn.slice(gate_up, [0, 0, MOE_INTER_CHIP], [1, 1, 2 * MOE_INTER_CHIP])
        else:
            gate_part = ttnn.slice(gate_up, [0, 0], [1, MOE_INTER_CHIP])
            up_part = ttnn.slice(gate_up, [0, MOE_INTER_CHIP], [1, 2 * MOE_INTER_CHIP])
        ttnn.deallocate(gate_up)
        mid = ttnn.mul(ttnn.silu(gate_part), up_part)
        ttnn.deallocate(gate_part); ttnn.deallocate(up_part)
        down_e = ttnn.slice(
            w["experts_down"],
            [0, e, 0, 0],
            [1, e + 1, MOE_INTER_CHIP, HIDDEN],
        )
        down_e_w = ttnn.reshape(down_e, [MOE_INTER_CHIP, HIDDEN])
        expert_out = ttnn.matmul(mid, down_e_w, compute_kernel_config=HIFI4)
        ttnn.deallocate(down_e); ttnn.deallocate(down_e_w); ttnn.deallocate(mid)
        weighted = ttnn.multiply(expert_out, w_scalar)
        ttnn.deallocate(expert_out)
        if routed_partial is None:
            routed_partial = weighted
        else:
            new_routed = ttnn.add(routed_partial, weighted)
            ttnn.deallocate(routed_partial); ttnn.deallocate(weighted)
            routed_partial = new_routed

    # SHARED EXPERT (sharded intermediate dim)
    s_gate = ttnn.matmul(h_tt, w["shared_gate"], compute_kernel_config=HIFI4)  # [1, 128] per chip
    s_up = ttnn.matmul(h_tt, w["shared_up"], compute_kernel_config=HIFI4)      # [1, 128] per chip
    s_mid = ttnn.mul(ttnn.silu(s_gate), s_up)
    shared_partial = ttnn.matmul(s_mid, w["shared_down"], compute_kernel_config=HIFI4)  # [1, 2048] per chip (partial)
    shared_full = all_reduce_tt(shared_partial, mesh)      # [1, 2048] replicated

    # Scalar gate (replicated weight; output [1, 1] replicated)
    gate_logit = ttnn.matmul(h_tt, w["shared_expert_gate"], compute_kernel_config=HIFI4)  # [1, 1]
    gate_sig = ttnn.sigmoid(gate_logit)
    gated_shared = ttnn.mul(shared_full, gate_sig)  # broadcast

    # Routed sum + shared (routed is 0 for first cut)
    if routed_partial is None:
        if sub_capture is not None:
            sub_capture["moe_final"] = _ttnn_to_numpy_replicated(gated_shared, mesh).reshape(-1)
            sub_capture["moe_gated_shared"] = sub_capture["moe_final"].copy()
        return gated_shared
    routed_full = all_reduce_tt(routed_partial, mesh)
    if sub_capture is not None:
        sub_capture["moe_routed_full"] = _ttnn_to_numpy_replicated(routed_full, mesh).reshape(-1)
        sub_capture["moe_gated_shared"] = _ttnn_to_numpy_replicated(gated_shared, mesh).reshape(-1)
    final = ttnn.add(routed_full, gated_shared)
    if sub_capture is not None:
        sub_capture["moe_final"] = _ttnn_to_numpy_replicated(final, mesh).reshape(-1)
    return final


# ── Persistent state + bootstrap ───────────────────────────────────────
class State:
    def __init__(self):
        self.mesh = None
        self.tokenizer = None
        self.text_cfg = None
        self.layer_types = None
        self.embed_w_np = None
        self.embed_tt = None     # ROW_MAJOR table for ttnn.embedding
        self.final_norm_tt = None
        self.lm_head_tt = None
        self.per_layer_tt = None
        self.dn_caches_tt = None
        self.kv_caches_tt = None
        # B17 trace-capture input buffers (pre-allocated, written in-place
        # OUTSIDE the trace via update_input_buffers).
        self.tok_buf = None         # uint32 [1, 1] ROW_MAJOR — for ttnn.embedding(state.embed_tt)
        self.rot_idxs_buf = None    # uint32 [1, 1] ROW_MAJOR — for cos/sin table lookup
        self.cos_table_tt = None    # fp32 [MAX_KV, ROTARY_DIM] — pre-baked cos values for all positions
        self.sin_table_tt = None    # fp32 [MAX_KV, ROTARY_DIM] — pre-baked sin values for all positions
        # Attention path selection. "manual" is the historical Q@K^T → softmax → @V
        # path used through B16/B17. "sdpa" routes through ttnn paged_scaled_dot_product_attention_decode
        # with the 27B B3 compute_kernel_config (HiFi2, no fp32_dest_acc). See
        # research/35b_a3b_sdpa_swap_design.md + feedback_35b_a3b_attn_layer_drift.md.
        # Set BEFORE bootstrap; the cache allocation + paged plumbing depends on this flag.
        # Default flipped to "sdpa" 2026-05-22: 85-pos ladder showed pos-0 bit-perfect,
        # median cos_final 0.94 → 0.95, structural cleanup. Top-1 81% vs 80% is within
        # noise; the dominant 35B drift mechanism is NOT the SDPA bug (it's something
        # else — see feedback_35b_a3b_sdpa_swap_result.md). Override with "manual"
        # before bootstrap for A/B comparisons.
        self.attn_mode = "sdpa"
        # Phase 2 ablation flag: skip the K-broadcast workaround in the SDPA path
        # and apply RoPE directly on [1, HEAD_DIM] K. The broadcast was added in
        # commit c5b0012 to work around a ttnn [1, HEAD_DIM] slice/concat bug
        # (feedback_qwen36_attn_rope_single_row_ttnn_bug.md). Phase 1 drift
        # attribution identified the 4x redundant bf16 ops as the prime
        # per-step-noise suspect. Set True to test whether the ttnn bug is fixed
        # in the current build — if K is silently corrupted, drift will collapse;
        # if drift is unchanged or improves, broadcast workaround can be removed.
        self.attn_sdpa_no_broadcast = False
        # SDPA compute kernel config variant. "B3" = 27B-validated recipe
        # (HiFi2, no fp32_dest_acc) — currently default. "hifi4_fp32" =
        # HiFi4 + fp32_dest_acc_en=True for max precision. The 27B bug that
        # made this variant buggy may not apply at 35B-A3B's GQA shapes.
        # See feedback_35b_a3b_drift_is_user_facing.md.
        self.sdpa_compute_variant = "B3"
        # Residual-add dtype. "bf16" = current behavior (ttnn.add stays bf16).
        # "fp32" = upcast operands to fp32 before each residual ADD in
        # layer_forward_ttnn, then cast back to bf16. The bf16 ADD of large-
        # magnitude tensors at late layers (L39 has L2~117) loses ~0.014
        # cos to bf16 quantization on the sum. fp32 add removes that noise.
        # ttnn.add's Python binding doesn't expose compute_kernel_config,
        # so explicit casts are how we plumb fp32 accumulation here.
        self.residual_add_dtype = "bf16"
        # SDPA-mode plumbing — only populated when attn_mode == "sdpa".
        self.cur_pos_buf = None             # int32 [1] device — replicated; consumed by paged_update_cache + paged SDPA
        self.page_table_tt = None           # int32 [1, NUM_BLOCKS] — identity mapping for B=1
        self.paged_write_mem_cfg = None     # HEIGHT_SHARDED L1 mem_cfg for the paged_update_cache input tile
        self.paged_sdpa_progcfg = None      # SDPAProgramConfig (CoreCoord(4,4), exp_approx_mode=False)
        self.sdpa_compute_kernel_config = None  # B3: HiFi2 + math_approx=False + fp32_dest_acc=False + packer_l1=False

    def reset_caches_ttnn(self):
        n = self.text_cfg.num_hidden_layers
        dn = []
        kv = []
        for L in range(n):
            if self.layer_types[L] == "linear_attention":
                cs_np = np.zeros((NCHIPS, 1, CONV_DIM_CHIP, CONV_KERNEL), dtype=np.float32)
                rs_np = np.zeros((NCHIPS, 1, NV_PER_CHIP, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float32)
                cs_tt = ttnn.from_torch(
                    torch.from_numpy(cs_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=self.mesh, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
                )
                rs_tt = ttnn.from_torch(
                    torch.from_numpy(rs_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                    device=self.mesh, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
                )
                dn.append((cs_tt, rs_tt))
                kv.append(None)
            else:
                dn.append(None)
                if self.attn_mode == "sdpa":
                    # Pre-allocate paged K/V cache per attn layer.
                    # Match 27B server_tp.py:354-358 layout: logical 4D shape
                    # (NUM_BLOCKS, NCHIPS, BLOCK_SIZE, HEAD_DIM) sharded along dim=1
                    # gives per-chip rank-4 view (NUM_BLOCKS, 1, BLOCK_SIZE, HEAD_DIM).
                    # The "NCHIPS" axis here doubles as the kv-head-per-chip axis.
                    cache_shape = (self.sdpa_num_blocks, NCHIPS,
                                   self.sdpa_block_size, HEAD_DIM_ATTN)
                    kc_init = np.zeros(cache_shape, dtype=np.float32)
                    vc_init = np.zeros(cache_shape, dtype=np.float32)
                    # state.kv_cache_dtype controls KV storage precision.
                    # bf16 is default + 27B-validated. fp32 was hard-rejected by
                    # paged SDPA decode in 27B's ttnn build (feedback_fp32_kv_cache.md);
                    # may have been fixed or may still apply. Try via flag for the
                    # 35B drift-mitigation probe.
                    kv_dtype = getattr(self, "kv_cache_dtype", "bf16")
                    kv_tt_dtype = ttnn.float32 if kv_dtype == "fp32" else ttnn.bfloat16
                    kc = ttnn.from_torch(
                        torch.from_numpy(kc_init), dtype=kv_tt_dtype, layout=ttnn.TILE_LAYOUT,
                        device=self.mesh, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=1),
                    )
                    vc = ttnn.from_torch(
                        torch.from_numpy(vc_init), dtype=kv_tt_dtype, layout=ttnn.TILE_LAYOUT,
                        device=self.mesh, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=1),
                    )
                    kv.append((kc, vc))
                else:
                    kv.append(None)
        self.dn_caches_tt = dn
        self.kv_caches_tt = kv


def _ttnn_to_numpy_replicated(t, mesh):
    """Concat-mesh-to-tensor of a replicated tensor, take chip 0 view as numpy."""
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    ).float().numpy()[0]


def _ttnn_to_numpy_perchip(t, mesh):
    """Concat-mesh-to-tensor and return list of NCHIPS per-chip numpy arrays.

    Uses np.split along axis 0 so per-chip slabs of any dim-0 size work
    (e.g. core_attn_out with NV_PER_CHIP rows per chip, not just 1).
    """
    arr = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    ).float().numpy()
    return list(np.split(arr, NCHIPS, axis=0))


def _reassemble_qkv_chip_to_hf(per_chip_list, key_dim_chip, value_dim_chip):
    """Per-chip mixed_qkv layout [Q_chip | K_chip | V_chip] → HF layout
    [Q_full | K_full | V_full] as a flat numpy vector.
    """
    qs, ks, vs = [], [], []
    for chip_arr in per_chip_list:
        flat = chip_arr.reshape(-1)
        assert flat.shape[0] == 2 * key_dim_chip + value_dim_chip, \
            f"unexpected chip dim {flat.shape}"
        qs.append(flat[:key_dim_chip])
        ks.append(flat[key_dim_chip:2 * key_dim_chip])
        vs.append(flat[2 * key_dim_chip:])
    return np.concatenate(qs + ks + vs, axis=0)


def _reassemble_heads_chip_to_hf(per_chip_list):
    """Per-chip head-sharded tensor (e.g. z/a/b with chip = NV_PER_CHIP heads)
    → full HF layout via straight concat across the head axis.
    """
    return np.concatenate([c.reshape(-1) for c in per_chip_list], axis=0)


def update_input_buffers(state, token_id, cur_pos):
    """Write new token id + rot_idxs (+ cur_pos in sdpa mode) into pre-allocated
    buffers in-place.

    Must be called OUTSIDE trace capture. The trace then reads from these
    buffers (no host-side from_torch inside captured region).
    """
    tok_host = ttnn.from_torch(
        torch.tensor([[token_id]], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    ttnn.copy_host_to_device_tensor(tok_host, state.tok_buf)
    rot_host = ttnn.from_torch(
        torch.tensor([[cur_pos]], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    ttnn.copy_host_to_device_tensor(rot_host, state.rot_idxs_buf)
    if state.attn_mode == "sdpa":
        pos_host = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        ttnn.copy_host_to_device_tensor(pos_host, state.cur_pos_buf)


def _precompute_cos_sin_table(mesh, max_pos):
    """Precompute cos/sin tables for all positions [0, max_pos) of partial RoPE.

    Returns (cos_table_tt, sin_table_tt) each shape [max_pos, ROTARY_DIM] fp32.
    Used with ttnn.embedding(rot_idxs_buf, table) to look up the row for the
    current position inside the trace (no host writes per step).
    """
    inv_freq = 1.0 / (ROPE_THETA ** (
        np.arange(0, ROTARY_DIM, 2).astype(np.float64) / ROTARY_DIM))  # [R/2]
    pos = np.arange(0, max_pos, dtype=np.float64)
    angle = pos[:, None] * inv_freq[None, :]   # [max_pos, R/2]
    cos_half = np.cos(angle).astype(np.float32)
    sin_half = np.sin(angle).astype(np.float32)
    cos = np.concatenate([cos_half, cos_half], axis=-1)  # [max_pos, R]
    sin = np.concatenate([sin_half, sin_half], axis=-1)
    cos_tt = ttnn.from_torch(
        torch.from_numpy(cos),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sin_tt = ttnn.from_torch(
        torch.from_numpy(sin),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return cos_tt, sin_tt


def step_forward_ttnn(state, tok_id, pos, capture=None):
    """Eager-mode public API: updates input buffers + runs step_forward_inner.

    Caller-friendly: pass tok_id (int) and pos (int). For trace capture, call
    update_input_buffers + step_forward_inner directly (so update writes happen
    OUTSIDE the captured trace region).
    """
    update_input_buffers(state, tok_id, pos)
    return step_forward_inner(state, capture=capture)


def step_forward_inner(state, capture=None):
    """Trace-friendly forward step: reads ONLY from pre-allocated buffers
    (state.tok_buf, state.rot_idxs_buf) and per-layer weights. No host writes
    inside. Returns next_id (int — one 8-byte readback at end).
    """
    embed_out = ttnn.embedding(state.tok_buf, state.embed_tt)  # [1, 1, HIDDEN] per chip
    h_tt = ttnn.to_layout(embed_out, ttnn.TILE_LAYOUT)
    ttnn.deallocate(embed_out)
    if capture is not None:
        capture["embed"] = _ttnn_to_numpy_replicated(h_tt, state.mesh).reshape(-1)

    # 2. RoPE cos/sin: look up the row for the current position from the
    # pre-baked table via ttnn.embedding(rot_idxs_buf, cos_table_tt).
    # Avoids per-step host writes (trace-friendly).
    cos_row = ttnn.embedding(state.rot_idxs_buf, state.cos_table_tt)  # [1, 1, R]
    sin_row = ttnn.embedding(state.rot_idxs_buf, state.sin_table_tt)
    # Cast to TILE_LAYOUT for downstream ops (rms_norm, mul, etc.)
    cos_tt = ttnn.to_layout(cos_row, ttnn.TILE_LAYOUT)
    sin_tt = ttnn.to_layout(sin_row, ttnn.TILE_LAYOUT)
    ttnn.deallocate(cos_row); ttnn.deallocate(sin_row)
    # Reshape from [1, 1, R] → [1, R] to match _apply_partial_rope broadcast expectation
    cos_tt = ttnn.reshape(cos_tt, [1, ROTARY_DIM])
    sin_tt = ttnn.reshape(sin_tt, [1, ROTARY_DIM])

    # 3. 40-layer chain.
    n = state.text_cfg.num_hidden_layers
    sub_capture_layers = capture.get("sub_capture_layers", []) if capture is not None else []
    for L in range(n):
        lt = state.layer_types[L]
        sc = {} if (capture is not None and L in sub_capture_layers) else None
        h_new, new_dn, new_kv = layer_forward_ttnn(
            h_tt, state.per_layer_tt[L], lt, state.mesh,
            cos_tt, sin_tt, state.dn_caches_tt[L], state.kv_caches_tt[L],
            sub_capture=sc, state=state,
        )
        if sc is not None:
            capture[f"layer_{L}_sub"] = sc
        ttnn.deallocate(h_tt)
        h_tt = h_new
        if new_dn is not None:
            # Deallocate old caches before overwriting
            old_conv, old_rec = state.dn_caches_tt[L]
            new_conv, new_rec = new_dn
            if new_conv is not old_conv:
                ttnn.deallocate(old_conv)
            if new_rec is not old_rec:
                ttnn.deallocate(old_rec)
            state.dn_caches_tt[L] = new_dn
        if new_kv is not None:
            state.kv_caches_tt[L] = new_kv
        if capture is not None:
            capture[f"layer_{L}"] = _ttnn_to_numpy_replicated(h_tt, state.mesh).reshape(-1)

    ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

    # 4. Final norm + lm_head + argmax (all on device).
    h_norm = ttnn.rms_norm(h_tt, weight=state.final_norm_tt, epsilon=EPS)
    ttnn.deallocate(h_tt)
    if capture is not None:
        capture["final_norm"] = _ttnn_to_numpy_replicated(h_norm, state.mesh).reshape(-1)
    logits = ttnn.matmul(h_norm, state.lm_head_tt, compute_kernel_config=HIFI4)  # [1, VOCAB] per chip (replicated)
    ttnn.deallocate(h_norm)
    if capture is not None:
        capture["logits"] = _ttnn_to_numpy_replicated(logits, state.mesh).reshape(-1)

    # On-device argmax requires ROW_MAJOR input (multicore argmax constraint).
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(logits)
    argmax_tt = ttnn.argmax(logits_rm, dim=-1, keepdim=True, use_multicore=True)
    ttnn.deallocate(logits_rm)

    # 5. Single 8-byte readback of the argmax — ONLY host transfer per step.
    next_id_t = ttnn.to_torch(
        argmax_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
    )
    ttnn.deallocate(argmax_tt)
    next_id = int(next_id_t.flatten()[0].item())
    return next_id


def bootstrap(state, log):
    log("[bootstrap] open mesh + fabric…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    state.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    log(f"  mesh: {state.mesh}")

    log("[bootstrap] config + tokenizer…")
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    state.text_cfg = cfg.text_config
    state.text_cfg.dtype = torch.bfloat16
    state.layer_types = list(state.text_cfg.layer_types)
    state.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    log("[bootstrap] enumerate shards + load top-level weights to mesh…")
    key_to_shard = build_key_to_shard()
    # Upload embed table to mesh (replicated). bf16 because the ttnn.embedding
    # kernel hard-rejects fp32 weights ("Weights tensor must have BFLOAT16
    # dtype"). We cast the embedding OUTPUT to fp32 in step_forward_inner to
    # get fp32 propagation through the residual stream — matches 27B's 91l
    # recipe in spirit (fp32 residual stream avoids bf16 quantization noise
    # on the residual ADDs at late layers).
    # ttnn.embedding requires ROW_MAJOR layout for the table.
    embed_w_np = load_t(key_to_shard, "model.language_model.embed_tokens.weight")
    state.embed_tt = ttnn.from_torch(
        torch.from_numpy(embed_w_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    state.embed_w_np = embed_w_np  # keep host copy too as fallback

    final_norm_w = load_t(key_to_shard, "model.language_model.norm.weight")
    lm_head_w = load_t(key_to_shard, "lm_head.weight")
    # final_norm uses (1+w) convention per Qwen3_5MoeRMSNorm — pre-add at upload
    state.final_norm_tt = np_to_replicated(final_norm_w + 1.0, state.mesh)
    state.lm_head_tt = np_to_replicated(lm_head_w.T, state.mesh)  # [hidden, vocab] for matmul

    log("[bootstrap] uploading 40 layer weights to mesh (sharded per plan §3.2)…")
    t0 = time.time()
    state.per_layer_tt = []
    for L in range(state.text_cfg.num_hidden_layers):
        layer_sd = {k.replace(f"model.language_model.layers.{L}.", ""):
                    load_t(key_to_shard, k)
                    for k in key_to_shard
                    if k.startswith(f"model.language_model.layers.{L}.")}
        layer_tt = {}
        # 91f loader convention: input_layernorm + post_attention_layernorm use
        # `output * (1 + weight)`; q_norm, k_norm same convention. linear_attn.norm
        # is standard `output * weight` (RMSNormGated). Pre-add 1.0 at upload so
        # downstream rms_norm calls just multiply by the stored weight.
        layer_tt["input_layernorm"] = np_to_replicated(layer_sd["input_layernorm.weight"] + 1.0, state.mesh)
        layer_tt["post_attention_layernorm"] = np_to_replicated(layer_sd["post_attention_layernorm.weight"] + 1.0, state.mesh)
        if state.layer_types[L] == "linear_attention":
            layer_tt.update(upload_dn_layer(layer_sd, state.mesh))
        else:
            layer_tt.update(upload_attn_layer(layer_sd, state.mesh))
        layer_tt.update(upload_moe_layer(layer_sd, state.mesh))
        state.per_layer_tt.append(layer_tt)
        if (L + 1) % 10 == 0:
            log(f"  layer {L+1}/{state.text_cfg.num_hidden_layers} uploaded ({time.time()-t0:.1f}s)")
    log(f"  all weights uploaded in {time.time()-t0:.1f}s")

    log("[bootstrap] pre-allocate trace input buffers + cos/sin table…")
    # Pre-allocate tok_buf and rot_idxs_buf — written in-place via
    # update_input_buffers() outside trace, read inside via ttnn.embedding.
    state.tok_buf = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    state.rot_idxs_buf = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    state.cos_table_tt, state.sin_table_tt = _precompute_cos_sin_table(state.mesh, MAX_KV)
    log(f"  tok_buf, rot_idxs_buf, cos/sin tables ({MAX_KV} positions) ready.")

    if state.attn_mode == "sdpa":
        # Paged SDPA plumbing — mirrors 27B server_tp.py:468-525.
        # NUM_BLOCKS*BLOCK_SIZE = MAX_KV; BLOCK_SIZE must be a multiple of
        # TILE_HEIGHT=32. cur_pos_buf is a single int32 device tensor that
        # paged_update_cache + paged SDPA consume; written each step via
        # update_input_buffers OUTSIDE any captured trace.
        SDPA_BLOCK_SIZE = 32
        SDPA_NUM_BLOCKS = MAX_KV // SDPA_BLOCK_SIZE  # 128 at MAX_KV=4096
        SDPA_TILE_HEIGHT = 32
        state.sdpa_block_size = SDPA_BLOCK_SIZE
        state.sdpa_num_blocks = SDPA_NUM_BLOCKS
        state.cur_pos_buf = ttnn.from_torch(
            torch.zeros((1,), dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        page_table_np = np.arange(SDPA_NUM_BLOCKS, dtype=np.int32).reshape(1, SDPA_NUM_BLOCKS)
        state.page_table_tt = ttnn.from_torch(
            torch.from_numpy(page_table_np),
            device=state.mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        compute_grid = state.mesh.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(1, compute_grid, row_wise=True)
        shard_spec = ttnn.ShardSpec(
            shard_grid, [SDPA_TILE_HEIGHT, HEAD_DIM_ATTN],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        state.paged_write_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec,
        )
        # B3: HiFi2 + math_approx_mode=False + fp32_dest_acc_en=False + packer_l1_acc=False.
        # CoreCoord(4,4) keeps SDPA in the per-chip slab (default grabs ~110 cores
        # per head, which trips the tree-reduction error on (1,4) mesh).
        state.paged_sdpa_progcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
            q_chunk_size=0,
            k_chunk_size=0,
            exp_approx_mode=False,
        )
        if state.sdpa_compute_variant == "hifi4_fp32":
            state.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            variant_label = "hifi4_fp32 (HiFi4 + fp32_dest_acc + no packer_l1)"
        else:  # "B3" default
            state.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            variant_label = "B3 (HiFi2 + no fp32_dest_acc + no packer_l1)"
        log(f"  SDPA paged plumbing: NUM_BLOCKS={SDPA_NUM_BLOCKS}, "
            f"BLOCK_SIZE={SDPA_BLOCK_SIZE}, MAX_KV={MAX_KV}")
        log(f"  SDPA compute_kernel_config: {variant_label}")
    else:
        log("  attn_mode=manual; SDPA plumbing skipped")

    log("[bootstrap] ready.")


# ── Smoke test (run only this main if invoked directly for debugging) ──
def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = State()
    bootstrap(state, lambda m: print(m, flush=True))

    print("\nsmoke: MoE block on layer 0 with first prompt token's embed")
    prompt_ids = state.tokenizer.encode("The capital of France is")
    tok_id = prompt_ids[0]
    h_np = state.embed_w_np[tok_id].reshape(1, HIDDEN).astype(np.float32)
    h_tt = np_to_replicated(h_np, state.mesh)
    print(f"  h_tt shape={list(h_tt.shape)} dtype={h_tt.dtype}")

    # Run MoE on layer 0 (forward through just one block to validate plumbing)
    h_norm = ttnn.rms_norm(h_tt, weight=state.per_layer_tt[0]["post_attention_layernorm"], epsilon=EPS)
    out = moe_forward_ttnn(h_norm, state.per_layer_tt[0], state.mesh)
    print(f"  MoE out shape={list(out.shape)}")
    out_np = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
    ).float().numpy()[0]
    print(f"  out norm: {np.linalg.norm(out_np):.4f} (shape {out_np.shape})")
    print(f"  ✓ on-device MoE block (routed + shared) works on (1,4) mesh")

    # DN smoke (placeholder — projections + out_proj, no conv/recurrence yet)
    if state.layer_types[0] == "linear_attention":
        print("\n  DN smoke (layer 0)…")
        # Build zero state tensors (sharded per chip)
        conv_state_np = np.zeros((NCHIPS, 1, CONV_DIM_CHIP, CONV_KERNEL), dtype=np.float32)
        recurrent_state_np = np.zeros((NCHIPS, 1, NV_PER_CHIP, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float32)
        conv_state_tt = ttnn.from_torch(
            torch.from_numpy(conv_state_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0),
        )
        recurrent_state_tt = ttnn.from_torch(
            torch.from_numpy(recurrent_state_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0),
        )
        dn_out, _, _ = dn_forward_ttnn(h_tt, state.per_layer_tt[0], state.mesh,
                                        (conv_state_tt, recurrent_state_tt))
        dn_out_np = ttnn.to_torch(
            dn_out, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        ).float().numpy()[0]
        print(f"  DN out norm: {np.linalg.norm(dn_out_np):.4f} (shape {dn_out_np.shape})")
        print(f"  ✓ on-device DN with recurrence works")

    # Attention smoke on layer 3 (first full_attention layer)
    attn_layer_idx = None
    for i, lt in enumerate(state.layer_types):
        if lt == "full_attention":
            attn_layer_idx = i
            break
    if attn_layer_idx is not None and attn_layer_idx < len(state.per_layer_tt):
        print(f"\n  Attention smoke (layer {attn_layer_idx})…")
        # Dummy cos/sin tensors replicated
        cos_np = np.zeros((1, 1, ROTARY_DIM), dtype=np.float32)
        sin_np = np.zeros((1, 1, ROTARY_DIM), dtype=np.float32)
        cos_tt = np_to_replicated(cos_np, state.mesh)
        sin_tt = np_to_replicated(sin_np, state.mesh)
        attn_out, _ = attn_forward_ttnn(h_tt, state.per_layer_tt[attn_layer_idx],
                                         state.mesh, cos_tt, sin_tt)
        attn_out_np = ttnn.to_torch(
            attn_out, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        ).float().numpy()[0]
        print(f"  Attn out norm: {np.linalg.norm(attn_out_np):.4f} (shape {attn_out_np.shape})")
        print(f"  ✓ on-device attention plumbing works")

    # Full layer smoke (layer 0 = DN + MoE composed with residuals + layernorms)
    print("\n  Layer 0 composed smoke (DN + MoE + residuals + layernorms)…")
    conv_state_np2 = np.zeros((NCHIPS, 1, CONV_DIM_CHIP, CONV_KERNEL), dtype=np.float32)
    rec_state_np2 = np.zeros((NCHIPS, 1, NV_PER_CHIP, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float32)
    conv_state_tt2 = ttnn.from_torch(
        torch.from_numpy(conv_state_np2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0),
    )
    rec_state_tt2 = ttnn.from_torch(
        torch.from_numpy(rec_state_np2), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0),
    )
    cos_zero = np_to_replicated(np.zeros((1, 1, ROTARY_DIM), dtype=np.float32), state.mesh)
    sin_zero = np_to_replicated(np.zeros((1, 1, ROTARY_DIM), dtype=np.float32), state.mesh)
    layer_out, _, _ = layer_forward_ttnn(
        h_tt, state.per_layer_tt[0], state.layer_types[0], state.mesh,
        cos_zero, sin_zero, (conv_state_tt2, rec_state_tt2), None,
    )
    layer_out_np = ttnn.to_torch(
        layer_out, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
    ).float().numpy()[0]
    print(f"  Layer 0 out norm: {np.linalg.norm(layer_out_np):.4f}")
    print(f"  ✓ on-device full layer (DN + MoE + residuals + 2 layernorms) works")

    # Deallocate single-layer smoke leftovers so step_forward_ttnn starts clean
    ttnn.deallocate(h_tt)
    ttnn.deallocate(out); ttnn.deallocate(h_norm); ttnn.deallocate(layer_out)
    if state.layer_types[0] == "linear_attention":
        ttnn.deallocate(dn_out)
        ttnn.deallocate(conv_state_tt); ttnn.deallocate(recurrent_state_tt)
    ttnn.deallocate(conv_state_tt2); ttnn.deallocate(rec_state_tt2)
    ttnn.deallocate(cos_zero); ttnn.deallocate(sin_zero)
    if attn_layer_idx is not None and attn_layer_idx < len(state.per_layer_tt):
        ttnn.deallocate(attn_out); ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

    # ── FULL end-to-end on-device step_forward_ttnn smoke ──────────────
    print("\n  step_forward_ttnn end-to-end smoke (embed → 40 layers → lm_head → argmax)…")
    state.reset_caches_ttnn()
    print(f"  caches reset: {sum(1 for x in state.dn_caches_tt if x is not None)} DN, "
          f"{sum(1 for x in state.kv_caches_tt if x is None)} KV placeholders")
    t0 = time.time()
    next_id = step_forward_ttnn(state, tok_id, pos=0)
    t1 = time.time()
    tok_text = state.tokenizer.decode([next_id])
    print(f"  next_id={next_id} text={tok_text!r}  (took {(t1-t0)*1000:.1f} ms)")
    print(f"  ✓ FULL on-device step_forward_ttnn works end-to-end")

    ttnn.close_mesh_device(state.mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print("smoke done.")


if __name__ == "__main__":
    main()
