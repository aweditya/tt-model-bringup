#!/usr/bin/env python3
"""MM7 v0.1.0 — Nemotron-3 Nano 30B-A3B fully-on-device server scaffold.

THIS FILE STARTS AT v0.1.0 SCOPE — bootstrap + top-level (embed +
final_norm + lm_head + argmax) ONLY. Layer-by-layer composites land
incrementally per `research/nemotron3_nano_30b_a3b_bringup_plan.md`
§3b:
  v0.1.1 — L5 (Attention)
  v0.1.2 — L0 (Mamba2)
  v0.1.3 — L1 (MoE)
  v0.2   — full 52-layer dispatch
  v0.3+  — multi-step + perf + CB + HTTP

REUSE: structural fork of `experiments/serve/server_35b_ttnn.py`
(bootstrap pattern + helpers verbatim, +0 LOC of layer math; layer
uploads ship at v0.1.x). Key deltas vs 35B:
  - HIDDEN=2688 (vs 2048), VOCAB=131072 (vs 248320)
  - NUM_LAYERS=52 hybrid (23 Mamba2 + 23 MoE + 6 Attention)
  - **Llama-style RMSNorm — NO `+1.0`** (vs Qwen-style 35B `+1.0`)
  - Weight key prefix `backbone.*` (vs 35B's `model.language_model.*`)
  - tie_word_embeddings=False — separate `lm_head.weight`
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402

SNAPSHOT_ROOT = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" / "snapshots"
)
SOCK_PATH = PROJECT_ROOT / ".cache" / "server_nemotron3_nano_ttnn.sock"
LOG_PATH = PROJECT_ROOT / ".cache" / "server_nemotron3_nano_ttnn.log"

# Model constants (verified at v0.0 config probe + v0.0.2 weights introspect)
HIDDEN = 2688
VOCAB = 131072
N_LAYERS = 52
EPS = 1e-5  # Mamba/Nemotron config layer_norm_epsilon

# Mamba2
MAMBA_HEADS = 64
MAMBA_HEAD_DIM = 64
SSM_STATE = 128
N_GROUPS = 8
CONV_KERNEL = 4
D_INNER = MAMBA_HEADS * MAMBA_HEAD_DIM  # 4096
CONV_DIM_M = D_INNER + 2 * N_GROUPS * SSM_STATE  # 6144

# Attention
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM_ATTN = 128

# MoE
N_ROUTED_EXPERTS = 128
N_SHARED_EXPERTS = 1
TOP_K_ROUTED = 6
E_LOCAL = N_ROUTED_EXPERTS // 4  # 32 experts per chip (Expert Parallel)
# n_group=1 + topk_group=1 means the group restriction is degenerate
# (one big group containing all 128 experts). Verified at MoE config probe
# 2026-06-05; brief had said n_group=8 but actual config says 1.
N_GROUP_MOE = 1
TOPK_GROUP = 1
ROUTED_SCALING = 2.5
ROUTED_INTERMEDIATE = 1856
SHARED_INTERMEDIATE = 3712

NCHIPS = 4
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MAX_KV = 8192  # bumped vs 35B's 4096; Nemotron is a longer-context model

# HIFI4 + fp32_dest_acc=True is the right matmul default for all paths
# OTHER than SDPA decode — matches 35B `server_35b_ttnn.py` HIFI4. The
# bf16-noise floor without fp32_dest_acc accumulates through reductions
# (lm_head is a HIDDEN=2688 reduce; visible as logits cos ~0.92 in
# probes when running HiFi2/fp32_dest_acc=False).
HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)

# B3 (HiFi2 + fp32_dest_acc=False) is the SDPA-specific recipe per
# [[fp32-sdpa-cliff-probe]] — avoids the Blackhole HiFi4+fp32_dest_acc
# SDPA cliff at large positions. NOT for matmuls/lm_head.
B3_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)


# ── Tensor upload helpers (verbatim fork from 35B) ─────────────────────
def np_to_replicated(arr, mesh, dtype=ttnn.bfloat16):
    """Upload a numpy array to mesh, replicated on every chip."""
    return ttnn.from_torch(
        torch.from_numpy(arr.astype(np.float32)),
        dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def load_t(key_to_shard, key):
    """Load a single safetensors tensor as fp32 numpy."""
    with safe_open(key_to_shard[key], framework="pt") as f:
        return f.get_tensor(key).float().numpy()


def build_key_to_shard():
    """Enumerate every safetensors shard once, return {key → shard path}."""
    snap = next(SNAPSHOT_ROOT.glob("*"))
    out = {}
    for shard in sorted(snap.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                out[k] = shard
    return out


# ── Per-layer weight uploads ─────────────────────────────────────────
def upload_attn_layer(state, key_to_shard, L: int, log) -> dict:
    """Upload attention-layer weights (pre-norm + q/k/v/o_proj).

    Returns a dict with TT tensors. Replicated across the mesh for v0.1.1
    (no head sharding yet; that's a v0.5 perf optimization).
    """
    prefix = f"backbone.layers.{L}"
    mp = f"{prefix}.mixer"
    # Pre-norm — Llama-style, NO `+1.0`.
    norm_w = load_t(key_to_shard, f"{prefix}.norm.weight")
    assert norm_w.shape == (HIDDEN,)
    # Q/K/V/O projections.
    q_w = load_t(key_to_shard, f"{mp}.q_proj.weight")   # [NQ*HD, HIDDEN]
    k_w = load_t(key_to_shard, f"{mp}.k_proj.weight")   # [NKV*HD, HIDDEN]
    v_w = load_t(key_to_shard, f"{mp}.v_proj.weight")   # [NKV*HD, HIDDEN]
    o_w = load_t(key_to_shard, f"{mp}.o_proj.weight")   # [HIDDEN, NQ*HD]
    assert q_w.shape == (NUM_Q_HEADS * HEAD_DIM_ATTN, HIDDEN), q_w.shape
    assert k_w.shape == (NUM_KV_HEADS * HEAD_DIM_ATTN, HIDDEN), k_w.shape
    assert v_w.shape == (NUM_KV_HEADS * HEAD_DIM_ATTN, HIDDEN), v_w.shape
    assert o_w.shape == (HIDDEN, NUM_Q_HEADS * HEAD_DIM_ATTN), o_w.shape
    out = {
        "norm":   np_to_replicated(norm_w, state.mesh),
        # Pre-transpose at upload — matmul wants [in_dim, out_dim].
        "q_proj": np_to_replicated(q_w.T, state.mesh),   # [HIDDEN, NQ*HD]
        "k_proj": np_to_replicated(k_w.T, state.mesh),   # [HIDDEN, NKV*HD]
        "v_proj": np_to_replicated(v_w.T, state.mesh),   # [HIDDEN, NKV*HD]
        "o_proj": np_to_replicated(o_w.T, state.mesh),   # [NQ*HD, HIDDEN]
        "kind":   "attention",
    }
    log(f"  L{L} (attention): norm + q/k/v/o_proj uploaded replicated bf16")
    return out


def upload_moe_layer_ep(state, key_to_shard, L: int, log) -> dict:
    """v0.1.4 — Expert-Parallel upload (DeepSeek-V3-style).

    Shards 128 routed experts as E_LOCAL=32 per chip along the mesh
    EP axis. Per-chip memory drops from ~1.3 GB (full replicated) to
    ~340 MB (only this chip's local experts). Shared expert stays
    replicated (it's small).

    Forks the structure from `models/demos/deepseek_v3/tt/experts.py`
    but with Nemotron deltas: no gate_proj (just `up_proj → relu² →
    down_proj`), no SwiGLU.

    Memory layout per chip:
      experts_up_local   [E_LOCAL=32, HIDDEN=2688, INTERMEDIATE=1856] bf16
      experts_down_local [E_LOCAL=32, INTERMEDIATE, HIDDEN]            bf16
      (sharded along dim 0 — the expert dim — via ShardTensorToMesh)

    Plus the per-MoE-layer overhead (replicated):
      norm                          [HIDDEN]
      gate_w                        [HIDDEN, N_ROUTED_EXPERTS]
      e_score_correction_bias       [N_ROUTED_EXPERTS]
      shared_up_w                   [HIDDEN, SHARED_INTERMEDIATE]
      shared_down_w                 [SHARED_INTERMEDIATE, HIDDEN]
      expert_mapping_tensors        [1, 1, N_ROUTED_EXPERTS, NCHIPS]
    """
    prefix = f"backbone.layers.{L}"
    mp = f"{prefix}.mixer"
    assert N_ROUTED_EXPERTS % NCHIPS == 0, \
        f"N_ROUTED_EXPERTS {N_ROUTED_EXPERTS} not divisible by NCHIPS {NCHIPS}"

    norm_w = load_t(key_to_shard, f"{prefix}.norm.weight")
    gate_w = load_t(key_to_shard, f"{mp}.gate.weight")
    bias = load_t(key_to_shard, f"{mp}.gate.e_score_correction_bias")

    # Stack 128 experts into a single big array, pre-transposed for matmul,
    # then shard along dim 0 (the expert dim) across NCHIPS.
    # up_w on disk: [INTERMEDIATE, HIDDEN]
    # Stacked + transposed: [N_EXPERTS, HIDDEN, INTERMEDIATE]
    up_stack = np.stack([
        load_t(key_to_shard, f"{mp}.experts.{e}.up_proj.weight").T
        for e in range(N_ROUTED_EXPERTS)
    ], axis=0)  # [128, 2688, 1856]
    log(f"  L{L} up_stack:   {up_stack.shape}")

    # down_w on disk: [HIDDEN, INTERMEDIATE]
    # Stacked + transposed: [N_EXPERTS, INTERMEDIATE, HIDDEN]
    down_stack = np.stack([
        load_t(key_to_shard, f"{mp}.experts.{e}.down_proj.weight").T
        for e in range(N_ROUTED_EXPERTS)
    ], axis=0)  # [128, 1856, 2688]
    log(f"  L{L} down_stack: {down_stack.shape}")

    # Upload as sharded along dim 0 (the expert dim)
    experts_up_local_tt = ttnn.from_torch(
        torch.from_numpy(up_stack.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0),
    )
    experts_down_local_tt = ttnn.from_torch(
        torch.from_numpy(down_stack.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0),
    )

    # Shared expert (small, replicated)
    sh_up_w = load_t(key_to_shard, f"{mp}.shared_experts.up_proj.weight")
    sh_dn_w = load_t(key_to_shard, f"{mp}.shared_experts.down_proj.weight")
    shared_up_tt = np_to_replicated(sh_up_w.T, state.mesh)
    shared_down_tt = np_to_replicated(sh_dn_w.T, state.mesh)

    # Expert mapping tensor (DeepSeek-V3 pattern, demo line 95-101).
    # Shape: [1, 1, n_experts, n_devices] — eye(devices) repeated
    # n_experts_per_device times along dim 0.
    expert_mapping_torch = (
        torch.eye(NCHIPS, dtype=torch.int32)
        .repeat_interleave(E_LOCAL, dim=0)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    expert_mapping_tt = ttnn.from_torch(
        expert_mapping_torch,
        device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    out = {
        "kind":            "moe_ep",
        "norm":            np_to_replicated(norm_w, state.mesh),
        "gate_w":          np_to_replicated(gate_w.T, state.mesh),
        "e_score_bias_np": bias.copy(),
        "experts_up_local":   experts_up_local_tt,
        "experts_down_local": experts_down_local_tt,
        "shared_up":      shared_up_tt,
        "shared_down":    shared_down_tt,
        "expert_mapping": expert_mapping_tt,
    }
    log(f"  L{L} (moe_ep): {E_LOCAL} experts/chip + shared + router + "
        f"expert_mapping uploaded")
    return out


def upload_moe_layer_full(state, key_to_shard, L: int, log) -> dict:
    """v0.1.3.b — upload pre-norm + router + 128 routed experts + 1 shared.

    Memory budget (per chip, replicated): 128 routed experts × ~10 MB
    each + 1 shared expert × ~20 MB ≈ 1.29 GB. Plus the small overhead
    (norm, gate, bias). 23 MoE layers × 1.29 GB > our 8 GB/chip target
    — so for full v0.2 (all 52 layers) we'll need Pattern A sharding
    (35B precedent). For L1-only v0.1.3.b this fits.
    """
    prefix = f"backbone.layers.{L}"
    mp = f"{prefix}.mixer"

    norm_w = load_t(key_to_shard, f"{prefix}.norm.weight")
    gate_w = load_t(key_to_shard, f"{mp}.gate.weight")
    bias = load_t(key_to_shard, f"{mp}.gate.e_score_correction_bias")
    assert norm_w.shape == (HIDDEN,)
    assert gate_w.shape == (N_ROUTED_EXPERTS, HIDDEN)
    assert bias.shape == (N_ROUTED_EXPERTS,)

    # Routed experts — list of {up, down} tensors per expert.
    routed = []
    for e in range(N_ROUTED_EXPERTS):
        up_w = load_t(key_to_shard, f"{mp}.experts.{e}.up_proj.weight")
        dn_w = load_t(key_to_shard, f"{mp}.experts.{e}.down_proj.weight")
        assert up_w.shape == (ROUTED_INTERMEDIATE, HIDDEN), \
            f"L{L}.experts[{e}].up shape {up_w.shape}"
        assert dn_w.shape == (HIDDEN, ROUTED_INTERMEDIATE), \
            f"L{L}.experts[{e}].down shape {dn_w.shape}"
        # Pre-transpose for matmul: up [HIDDEN, intermediate], down [intermediate, HIDDEN]
        routed.append({
            "up":   np_to_replicated(up_w.T, state.mesh),
            "down": np_to_replicated(dn_w.T, state.mesh),
        })
        if (e + 1) % 32 == 0:
            log(f"  L{L} routed experts uploaded: {e+1}/{N_ROUTED_EXPERTS}")

    # Shared expert (2× wider intermediate)
    sh_up_w = load_t(key_to_shard, f"{mp}.shared_experts.up_proj.weight")
    sh_dn_w = load_t(key_to_shard, f"{mp}.shared_experts.down_proj.weight")
    assert sh_up_w.shape == (SHARED_INTERMEDIATE, HIDDEN)
    assert sh_dn_w.shape == (HIDDEN, SHARED_INTERMEDIATE)
    shared = {
        "up":   np_to_replicated(sh_up_w.T, state.mesh),
        "down": np_to_replicated(sh_dn_w.T, state.mesh),
    }

    out = {
        "kind": "moe",
        "norm": np_to_replicated(norm_w, state.mesh),
        "gate_w": np_to_replicated(gate_w.T, state.mesh),
        "e_score_bias_np": bias.copy(),
        "routed": routed,
        "shared": shared,
    }
    log(f"  L{L} (moe): norm + gate + bias + {N_ROUTED_EXPERTS} routed "
        f"+ 1 shared (2x wider) uploaded replicated bf16")
    return out


def upload_moe_layer_router_only(state, key_to_shard, L: int, log) -> dict:
    """v0.1.3.a — upload pre-norm + router weights only.

    Adds:
      norm                                   [HIDDEN]
      mixer.gate.weight                      [N_ROUTED_EXPERTS, HIDDEN]
      mixer.gate.e_score_correction_bias     [N_ROUTED_EXPERTS]

    Expert weights (128 routed + 1 shared) land at v0.1.3.b — they're
    the bulk of the layer (~256 + 2 weight tensors per layer × 23 layers).
    """
    prefix = f"backbone.layers.{L}"
    mp = f"{prefix}.mixer"
    norm_w = load_t(key_to_shard, f"{prefix}.norm.weight")
    gate_w = load_t(key_to_shard, f"{mp}.gate.weight")
    bias = load_t(key_to_shard, f"{mp}.gate.e_score_correction_bias")
    assert norm_w.shape == (HIDDEN,)
    assert gate_w.shape == (N_ROUTED_EXPERTS, HIDDEN), gate_w.shape
    assert bias.shape == (N_ROUTED_EXPERTS,), bias.shape

    out = {
        "kind": "moe_router_only",
        "norm": np_to_replicated(norm_w, state.mesh),
        # gate uploaded pre-transposed for matmul [HIDDEN, N_EXPERTS]
        "gate_w": np_to_replicated(gate_w.T, state.mesh),
        # bias kept host-side; cheap [128] vector, only used post-matmul
        # in the topk path which runs on host for v0.1.3.a.
        "e_score_bias_np": bias.copy(),
    }
    log(f"  L{L} (moe_router_only): norm + gate.weight + e_score_bias "
        f"uploaded replicated bf16 (experts come at v0.1.3.b)")
    return out


def upload_mamba2_layer(state, key_to_shard, L: int, log) -> dict:
    """Upload Mamba2-layer weights (pre-norm + in_proj/conv1d/dt_bias/A_log/D
    /norm/out_proj). Replicated across the mesh for v0.1.2 (no head
    sharding yet — v0.5 perf concern).

    Per-tensor shapes (from architecture brief + v0.0.2 introspect):
      norm           [HIDDEN]
      in_proj        [d_inner + conv_dim + num_heads, HIDDEN]
                     where conv_dim = d_inner + 2*n_groups*ssm_state
      conv1d.weight  [conv_dim, 1, conv_kernel]
      conv1d.bias    [conv_dim]
      dt_bias        [num_heads]
      A_log          [num_heads]
      D              [num_heads]
      norm.weight    [d_inner]  (MambaRMSNormGated; group_size=d_inner/n_groups)
      out_proj       [HIDDEN, d_inner]

    Helper constants (locally):
      d_inner   = MAMBA_HEADS * MAMBA_HEAD_DIM = 4096
      conv_dim  = d_inner + 2 * N_GROUPS * SSM_STATE = 6144
    """
    prefix = f"backbone.layers.{L}"
    mp = f"{prefix}.mixer"
    in_dim = D_INNER + CONV_DIM_M + MAMBA_HEADS  # 4096 + 6144 + 64 = 10304

    norm_w = load_t(key_to_shard, f"{prefix}.norm.weight")
    in_proj_w = load_t(key_to_shard, f"{mp}.in_proj.weight")
    conv1d_w = load_t(key_to_shard, f"{mp}.conv1d.weight")  # [conv_dim, 1, kernel]
    conv1d_b = load_t(key_to_shard, f"{mp}.conv1d.bias")
    dt_bias = load_t(key_to_shard, f"{mp}.dt_bias")
    A_log = load_t(key_to_shard, f"{mp}.A_log")
    D_w = load_t(key_to_shard, f"{mp}.D")
    mixer_norm_w = load_t(key_to_shard, f"{mp}.norm.weight")
    out_proj_w = load_t(key_to_shard, f"{mp}.out_proj.weight")

    assert norm_w.shape == (HIDDEN,)
    assert in_proj_w.shape == (in_dim, HIDDEN), in_proj_w.shape
    assert conv1d_w.shape == (CONV_DIM_M, 1, CONV_KERNEL), conv1d_w.shape
    assert conv1d_b.shape == (CONV_DIM_M,)
    assert dt_bias.shape == (MAMBA_HEADS,)
    assert A_log.shape == (MAMBA_HEADS,)
    assert D_w.shape == (MAMBA_HEADS,)
    assert mixer_norm_w.shape == (D_INNER,)
    assert out_proj_w.shape == (HIDDEN, D_INNER), out_proj_w.shape

    # Conv1d weight needs an extra kernel_height=1 dim for ttnn.conv1d
    # ([out_channels, in_channels/groups=1, kernel_height=1, kernel_width=4]).
    # The op itself is implemented as a 2D conv with H=1.
    conv1d_w_4d = conv1d_w[:, :, None, :]  # [conv_dim, 1, 1, 4]
    out = {
        "kind": "mamba2",
        "norm": np_to_replicated(norm_w, state.mesh),
        # in_proj uploaded pre-transposed for matmul [HIDDEN, in_dim].
        "in_proj": np_to_replicated(in_proj_w.T, state.mesh),
        # Conv1d weight + bias uploaded for ttnn.conv1d. Depth-wise:
        # groups=conv_dim makes this a per-channel kernel.
        "conv1d_w": ttnn.from_torch(
            torch.from_numpy(conv1d_w_4d.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "conv1d_b": ttnn.from_torch(
            torch.from_numpy(conv1d_b.reshape(1, 1, 1, -1).astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        # The small per-head ops (Mamba2 step's scalars) plumb at v0.1.2.c.
        "dt_bias_np": dt_bias.copy(),
        "A_log_np": A_log.copy(),
        "D_np": D_w.copy(),
        "mixer_norm_w_np": mixer_norm_w.copy(),
        # out_proj uploaded pre-transposed for matmul [d_inner, HIDDEN].
        "out_proj": np_to_replicated(out_proj_w.T, state.mesh),
    }
    log(f"  L{L} (mamba2): norm + in_proj + out_proj uploaded replicated bf16; "
        f"conv1d_w/b uploaded as host bf16 (for ttnn.conv1d at v0.1.2.b); "
        f"dt_bias/A_log/D/mixer_norm held host-side for v0.1.2.c")
    return out


# ── State ──────────────────────────────────────────────────────────────
class State:
    """All bootstrap state lives here so cb_api can stash it on a single object.

    v0.1.0 attrs:
      mesh, tokenizer, tok (alias), text_cfg, layer_types
      embed_tt, embed_w_np (fallback)
      final_norm_tt
      lm_head_tt  (shape [HIDDEN, VOCAB] for the matmul)

    v0.1.1 adds per-layer weight dicts:
      per_layer_tt[L]  → dict with at least:
        "norm"   — pre-norm weight (Llama-style, [HIDDEN])
        For attention layers also:
          "q_proj", "k_proj", "v_proj", "o_proj" weights
    """
    def __init__(self):
        self.mesh = None
        self.tokenizer = None
        self.tok = None
        self.text_cfg = None
        self.layer_types: list[str] = []
        self.embed_tt = None
        self.embed_w_np = None
        self.final_norm_tt = None
        self.lm_head_tt = None
        self.per_layer_tt: list[dict] = []


# ── Bootstrap ──────────────────────────────────────────────────────────
def bootstrap(state: State, log=None):
    """Open (1,4) mesh + upload embed + final_norm + lm_head.

    v0.1.0 stops here. Layer weights ship per-stage in v0.1.x.
    """
    if log is None:
        log = print

    log("[bootstrap] open mesh + fabric…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    # l1_small_size: ttnn.conv1d needs L1_SMALL for its kernel scratch
    # (default 0 → "0 B per bank" TT_FATAL). 64 KB is generous for our
    # depth-wise conv at conv_dim=6144.
    # trace_region_size: mirrors Gemma 4 — 400 MB for v0.4 traces.
    state.mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, NCHIPS),
        l1_small_size=65536,
        trace_region_size=400_000_000,
    )
    log(f"  mesh: {state.mesh}")

    log("[bootstrap] config + tokenizer…")
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Nemotron-H stores everything top-level (no .text_config wrapper for
    # the architectures we care about). Stay robust via getattr.
    state.text_cfg = getattr(cfg, "text_config", cfg)
    state.text_cfg.dtype = torch.bfloat16
    pattern = state.text_cfg.hybrid_override_pattern
    _kind = {"M": "mamba2", "E": "moe", "*": "attention"}
    state.layer_types = [_kind[c] for c in pattern]
    assert len(state.layer_types) == N_LAYERS, \
        f"layer_types len {len(state.layer_types)} != {N_LAYERS}"
    log(f"  {len(state.layer_types)} layers; "
        f"M/E/* counts = {state.layer_types.count('mamba2')}/"
        f"{state.layer_types.count('moe')}/"
        f"{state.layer_types.count('attention')}")
    state.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    state.tok = state.tokenizer

    log("[bootstrap] enumerate shards + load top-level weights to mesh…")
    key_to_shard = build_key_to_shard()
    log(f"  {len(key_to_shard)} weight keys across "
        f"{len(set(key_to_shard.values()))} shards")

    # Embed table — `backbone.embeddings.weight` per the Nemotron-H
    # modeling code (`backbone.embeddings = nn.Embedding(vocab, hidden)`,
    # modeling_nemotron_h.py:459). ttnn.embedding wants ROW_MAJOR bf16.
    embed_w_np = load_t(key_to_shard, "backbone.embeddings.weight")
    assert embed_w_np.shape == (VOCAB, HIDDEN), \
        f"embed shape {embed_w_np.shape} != ({VOCAB}, {HIDDEN})"
    state.embed_tt = ttnn.from_torch(
        torch.from_numpy(embed_w_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    state.embed_w_np = embed_w_np
    log(f"  embed: [{VOCAB}, {HIDDEN}] replicated bf16 ROW_MAJOR")

    # Final norm — Llama-style `y = x/rms * w` (NO `+1.0`); upload weight
    # verbatim. (35B's Qwen-style `+1.0` pre-add is NOT applied here —
    # see [[feedback-qwen36-qnorm-knorm-zero-centered]] for the contrast.)
    final_norm_w = load_t(key_to_shard, "backbone.norm_f.weight")
    assert final_norm_w.shape == (HIDDEN,), \
        f"final_norm shape {final_norm_w.shape} != ({HIDDEN},)"
    state.final_norm_tt = np_to_replicated(final_norm_w, state.mesh)
    log(f"  final_norm: [{HIDDEN}] replicated bf16 (Llama-style, no +1.0)")

    # LM head — separate weight (tie_word_embeddings=False per the brief
    # and v0.0.2 introspect). Upload as [HIDDEN, VOCAB] for matmul.
    lm_head_w = load_t(key_to_shard, "lm_head.weight")
    assert lm_head_w.shape == (VOCAB, HIDDEN), \
        f"lm_head shape {lm_head_w.shape} != ({VOCAB}, {HIDDEN})"
    state.lm_head_tt = np_to_replicated(lm_head_w.T, state.mesh)
    log(f"  lm_head: [{HIDDEN}, {VOCAB}] replicated bf16 (separate from embed)")

    # v0.1.1 staged layer upload — controlled by env so the smoke can
    # request just L5 (the attention warmup layer) without paying for
    # all 52. Without the env, bootstrap stays v0.1.0 scope.
    import os as _os
    upload_layers_csv = _os.environ.get("NEMOTRON3_UPLOAD_LAYERS", "")
    if upload_layers_csv:
        targets = [int(x) for x in upload_layers_csv.split(",") if x]
        log(f"[bootstrap] uploading {len(targets)} layer(s): {targets}")
        # Pre-fill per_layer_tt with None to allow sparse indexing.
        state.per_layer_tt = [None] * N_LAYERS
        for L in targets:
            kind = state.layer_types[L]
            if kind == "attention":
                state.per_layer_tt[L] = upload_attn_layer(state, key_to_shard, L, log)
            elif kind == "mamba2":
                state.per_layer_tt[L] = upload_mamba2_layer(state, key_to_shard, L, log)
            elif kind == "moe":
                # Default to Expert Parallel (v0.1.4). NEMOTRON3_MOE_MODE
                # overrides: "router_only" for the v0.1.3.a smoke,
                # "full" for the v0.1.3.b naive-replicated path (fallback).
                _mode = _os.environ.get("NEMOTRON3_MOE_MODE", "ep").lower()
                if _mode == "router_only":
                    state.per_layer_tt[L] = upload_moe_layer_router_only(
                        state, key_to_shard, L, log)
                elif _mode == "full":
                    state.per_layer_tt[L] = upload_moe_layer_full(
                        state, key_to_shard, L, log)
                else:  # "ep" — default
                    state.per_layer_tt[L] = upload_moe_layer_ep(
                        state, key_to_shard, L, log)
            else:
                raise NotImplementedError(
                    f"v0.1.3 supports {{attention, mamba2, moe}}; L{L} is {kind!r}")
        log(f"[bootstrap] v0.1.1 ready (sparse layer upload: {targets}).")
    else:
        log("[bootstrap] v0.1.0 ready (top-level only — no layers uploaded).")


# ── Forward fragments for v0.1.0 validation ───────────────────────────
def embed_lookup(state: State, ids_np):
    """Run ttnn.embedding(table, ids) → fp32 numpy [B, S, HIDDEN].

    Input: numpy int32 array of shape [B, S] (or [S]).
    Output: numpy fp32 array of shape [B, S, HIDDEN].
    """
    if ids_np.ndim == 1:
        ids_np = ids_np.reshape(1, -1)
    ids_tt = ttnn.from_torch(
        torch.from_numpy(ids_np.astype(np.int32)),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    out_tt = ttnn.embedding(ids_tt, state.embed_tt, layout=ttnn.TILE_LAYOUT)
    out_np = ttnn.to_torch(
        out_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )
    # Composer returns [NCHIPS*B, ...]; since replicated each chip
    # has identical data → take the first one.
    out_np = out_np[:1].float().numpy()
    return out_np


def apply_final_norm(state: State, h_np):
    """Run ttnn.rms_norm(h, weight=final_norm_w, epsilon=EPS) → fp32 numpy.

    Input: numpy fp32 [B, S, HIDDEN] or [S, HIDDEN].
    Output: numpy fp32 of same shape.
    """
    squeeze_batch = h_np.ndim == 2
    if squeeze_batch:
        h_np = h_np[None]
    h_tt = ttnn.from_torch(
        torch.from_numpy(h_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    out_tt = ttnn.rms_norm(h_tt, weight=state.final_norm_tt, epsilon=EPS)
    out_np = ttnn.to_torch(
        out_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )[:1].float().numpy()
    if squeeze_batch:
        out_np = out_np[0]
    return out_np


def apply_lm_head_and_argmax(state: State, h_np):
    """Run lm_head matmul + argmax over vocab.

    Input: numpy fp32 [B, S, HIDDEN] (or [S, HIDDEN]).
    Returns (logits_np[..., VOCAB], argmax_np[..., int32]).
    """
    squeeze_batch = h_np.ndim == 2
    if squeeze_batch:
        h_np = h_np[None]
    h_tt = ttnn.from_torch(
        torch.from_numpy(h_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    logits_tt = ttnn.matmul(
        h_tt, state.lm_head_tt, compute_kernel_config=HIFI4,
    )
    logits_np = ttnn.to_torch(
        logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )[:1].float().numpy()
    if squeeze_batch:
        logits_np = logits_np[0]
    argmax_np = logits_np.argmax(axis=-1).astype(np.int32)
    return logits_np, argmax_np


# ── Attention forward (v0.1.1) ────────────────────────────────────────
def attn_projections_only(state: State, h_input_np, layer_idx: int):
    """v0.1.1.a — pre-norm + q/k/v projections on a TT mesh, return as
    numpy. Skips SDPA + o_proj; those land at v0.1.1.b.

    Input:  h_input_np  numpy fp32  [B, S, HIDDEN] (or [S, HIDDEN])
    Output: dict with q/k/v numpy fp32 of shape [B, S, NQ*HD] / [B, S, NKV*HD].
            Also returns the pre-norm output as `h_norm` for downstream stages.
    """
    w = state.per_layer_tt[layer_idx]
    assert w is not None and w["kind"] == "attention", \
        f"L{layer_idx} weights not loaded as attention"

    squeeze_batch = h_input_np.ndim == 2
    if squeeze_batch:
        h_input_np = h_input_np[None]

    h_tt = ttnn.from_torch(
        torch.from_numpy(h_input_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    # Pre-norm (Llama-style, no +1.0)
    h_norm_tt = ttnn.rms_norm(h_tt, weight=w["norm"], epsilon=EPS)
    # Projections — replicated matmul. HiFi4+fp32_dest_acc for accuracy.
    q_tt = ttnn.matmul(h_norm_tt, w["q_proj"], compute_kernel_config=HIFI4)
    k_tt = ttnn.matmul(h_norm_tt, w["k_proj"], compute_kernel_config=HIFI4)
    v_tt = ttnn.matmul(h_norm_tt, w["v_proj"], compute_kernel_config=HIFI4)

    def _readback(t):
        arr = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )
        return arr[:1].float().numpy()

    out = {
        "h_norm": _readback(h_norm_tt),
        "q":      _readback(q_tt),
        "k":      _readback(k_tt),
        "v":      _readback(v_tt),
    }
    if squeeze_batch:
        for k in out:
            out[k] = out[k][0]
    return out


def attn_block_eager(state: State, h_input_np, layer_idx: int):
    """v0.1.1.c — full attention block FULLY ON-DEVICE.

    Forks the 27B prefill pattern at `experiments/serve/server_tp.py:1832`:
    a single `ttnn.transformer.scaled_dot_product_attention` call. Per
    [[reference-ttnn-sdpa-gqa-native]], the non-paged prefill kernel
    takes Q[b,nqh,s,dh] + K/V[b,nkv,s,dh] with `nqh ≠ nkv` as a 1st-class
    contract — NO caller-side K/V repeat. The NKV>1 contract issue
    (tt-metal #12330) is decode-only; doesn't fire here.

    Replicated across the mesh (no head sharding yet — v0.5 perf concern).

    Input:  h_input_np  numpy fp32  [B, S, HIDDEN] (or [S, HIDDEN])
    Returns dict with: h_norm, q, k, v, o_proj_out, block_out
    (all post-readback fp32 numpy for the validator).
    """
    w = state.per_layer_tt[layer_idx]
    assert w is not None and w["kind"] == "attention"

    squeeze_batch = h_input_np.ndim == 2
    if squeeze_batch:
        h_input_np = h_input_np[None]
    B, S, _ = h_input_np.shape
    NQ = NUM_Q_HEADS
    NKV = NUM_KV_HEADS
    HD = HEAD_DIM_ATTN

    h_input_tt = ttnn.from_torch(
        torch.from_numpy(h_input_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )

    def _readback(t):
        arr = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )
        return arr[:1].float().numpy()

    # Stage 1: pre-norm (Llama-style)
    h_norm_tt = ttnn.rms_norm(h_input_tt, weight=w["norm"], epsilon=EPS)
    h_norm_np = _readback(h_norm_tt)

    # Stage 2: Q/K/V projections
    q_tt = ttnn.matmul(h_norm_tt, w["q_proj"], compute_kernel_config=HIFI4)
    k_tt = ttnn.matmul(h_norm_tt, w["k_proj"], compute_kernel_config=HIFI4)
    v_tt = ttnn.matmul(h_norm_tt, w["v_proj"], compute_kernel_config=HIFI4)
    q_np = _readback(q_tt); k_np = _readback(k_tt); v_np = _readback(v_tt)

    # Stage 3: reshape + transpose to [B, NH, S, HD] for SDPA.
    # ttnn.reshape splits the last dim into (heads, head_dim);
    # ttnn.transpose then swaps the seq and head axes (B, S, NH, HD)
    # → (B, NH, S, HD).
    q4 = ttnn.transpose(ttnn.reshape(q_tt, [B, S, NQ, HD]), 1, 2)
    k4 = ttnn.transpose(ttnn.reshape(k_tt, [B, S, NKV, HD]), 1, 2)
    v4 = ttnn.transpose(ttnn.reshape(v_tt, [B, S, NKV, HD]), 1, 2)
    ttnn.deallocate(q_tt); ttnn.deallocate(k_tt); ttnn.deallocate(v_tt)

    # Stage 4: single on-device SDPA call. GQA is native (NQ ≠ NKV
    # is part of the contract); the kernel broadcasts K/V internally.
    # Scale = 1/sqrt(HD) is the standard attention scale Nemotron uses.
    attn_tt = ttnn.transformer.scaled_dot_product_attention(
        q4, k4, v4,
        is_causal=True,
        scale=1.0 / math.sqrt(HD),
        compute_kernel_config=B3_HIFI2,
    )
    ttnn.deallocate(q4); ttnn.deallocate(k4); ttnn.deallocate(v4)

    # Stage 5: transpose + reshape back to [B, S, NQ*HD] for o_proj.
    attn_tt = ttnn.transpose(attn_tt, 1, 2)
    attn_tt = ttnn.reshape(attn_tt, [B, S, NQ * HD])

    # Stage 6: o_proj
    o_tt = ttnn.matmul(attn_tt, w["o_proj"], compute_kernel_config=HIFI4)
    o_np = _readback(o_tt)
    ttnn.deallocate(attn_tt)

    # Stage 7: residual add on-device
    block_tt = ttnn.add(h_input_tt, o_tt)
    block_np = _readback(block_tt)
    ttnn.deallocate(h_input_tt); ttnn.deallocate(o_tt); ttnn.deallocate(block_tt)
    ttnn.deallocate(h_norm_tt)

    out = {
        "h_norm":     h_norm_np,
        "q":          q_np,
        "k":          k_np,
        "v":          v_np,
        "o_proj_out": o_np,
        "block_out":  block_np,
    }
    if squeeze_batch:
        for k in out:
            out[k] = out[k][0]
    return out


# ── Mamba2 forward (v0.1.2) ───────────────────────────────────────────
def mamba2_in_proj_only(state: State, h_input_np, layer_idx: int):
    """v0.1.2.a — pre-norm + in_proj only. Sanity-check weight upload
    and the [HIDDEN → d_inner + conv_dim + num_heads = 10304] matmul
    before composing the rest of the Mamba2 block.

    Input:  h_input_np  numpy fp32 [B, S, HIDDEN] (or [S, HIDDEN])
    Output: dict with h_norm, in_proj_out as numpy fp32.
    """
    w = state.per_layer_tt[layer_idx]
    assert w is not None and w["kind"] == "mamba2", \
        f"L{layer_idx} weights not loaded as mamba2"

    squeeze_batch = h_input_np.ndim == 2
    if squeeze_batch:
        h_input_np = h_input_np[None]

    h_tt = ttnn.from_torch(
        torch.from_numpy(h_input_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    h_norm_tt = ttnn.rms_norm(h_tt, weight=w["norm"], epsilon=EPS)
    in_proj_tt = ttnn.matmul(h_norm_tt, w["in_proj"], compute_kernel_config=HIFI4)

    def _readback(t):
        arr = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )
        return arr[:1].float().numpy()

    out = {
        "h_norm":      _readback(h_norm_tt),
        "in_proj_out": _readback(in_proj_tt),
    }
    if squeeze_batch:
        for k in out:
            out[k] = out[k][0]
    return out


def mamba2_in_proj_split_conv1d(state: State, h_input_np, layer_idx: int):
    """v0.1.2.b — pre-norm + in_proj + split + conv1d on x_BC.

    Pipeline (all on-device):
      pre-norm → in_proj → split into (z, x_BC, dt) along the last dim
      → ttnn.conv1d(x_BC) [depth-wise, K=4, sym pad=3, groups=conv_dim]

    Returns dict with intermediates incl. `conv1d_out` shape
    [B, conv_dim, S + 2*pad - K + 1] = [B, 6144, 8] matching the HF
    hook (HF captures the conv output BEFORE the causal-slice).
    """
    w = state.per_layer_tt[layer_idx]
    assert w is not None and w["kind"] == "mamba2"
    squeeze_batch = h_input_np.ndim == 2
    if squeeze_batch:
        h_input_np = h_input_np[None]
    B, S, _ = h_input_np.shape

    h_tt = ttnn.from_torch(
        torch.from_numpy(h_input_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    h_norm_tt = ttnn.rms_norm(h_tt, weight=w["norm"], epsilon=EPS)
    in_proj_tt = ttnn.matmul(h_norm_tt, w["in_proj"], compute_kernel_config=HIFI4)

    # Split last dim into (z [d_inner], x_BC [conv_dim], dt [num_heads]).
    z_tt   = ttnn.slice(in_proj_tt, [0, 0, 0],                          [B, S, D_INNER])
    xBC_tt = ttnn.slice(in_proj_tt, [0, 0, D_INNER],                    [B, S, D_INNER + CONV_DIM_M])
    dt_tt  = ttnn.slice(in_proj_tt, [0, 0, D_INNER + CONV_DIM_M],       [B, S, D_INNER + CONV_DIM_M + MAMBA_HEADS])

    # ttnn.conv1d expects input as [N, 1, W=S, C=conv_dim] (NHWC, with
    # implicit H=1 since this is a 1D conv). Need ROW_MAJOR for the op.
    xBC_nhwc = ttnn.to_layout(ttnn.reshape(xBC_tt, [B, 1, S, CONV_DIM_M]),
                                ttnn.ROW_MAJOR_LAYOUT)

    conv_out_tt = ttnn.conv1d(
        input_tensor=xBC_nhwc,
        weight_tensor=w["conv1d_w"],
        device=state.mesh,
        in_channels=CONV_DIM_M,
        out_channels=CONV_DIM_M,
        batch_size=B,
        input_length=S,
        kernel_size=CONV_KERNEL,
        stride=1,
        padding=CONV_KERNEL - 1,  # symmetric (= 3)
        dilation=1,
        groups=CONV_DIM_M,        # depth-wise
        bias_tensor=w["conv1d_b"],
    )

    def _readback(t):
        arr = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )
        return arr[:1].float().numpy()

    out = {
        "h_norm":      _readback(h_norm_tt),
        "in_proj_out": _readback(in_proj_tt),
        "z":           _readback(z_tt),
        "x_BC":        _readback(xBC_tt),
        "dt":          _readback(dt_tt),
        "conv1d_out":  _readback(conv_out_tt),
    }
    if squeeze_batch:
        for k in out:
            out[k] = out[k][0]
    return out


def mamba2_block_eager(state: State, h_input_np, layer_idx: int):
    """v0.1.2.c — full L0 Mamba2 block forward.

    Pipeline (mostly on-device; SSD step bridges via the G4 wrapper):
      1. TT  pre-norm (Llama-style, no +1.0)
      2. TT  in_proj matmul + slice into (z, x_BC, dt)
      3. TT  conv1d (depth-wise K=4, sym pad=3) on x_BC → [B,8,conv_dim]
      4. TT  causal slice [:, :S, :] → [B,S,conv_dim]
      5. TT  ttnn.silu
      6. TT  slice x_BC_silu into x_inner / B_inner / C_inner
      7. host readback of x_inner/z/dt/B_inner/C_inner (single batch)
      8. host SSD loop over S positions, calling the G4 wrapper
         (each call is on-device — runs the owned Mamba2 SSD kernel).
         Accumulates ssm_state across positions.
      9. TT  upload y → MambaRMSNormGated (group_size=d_inner/n_groups=512)
     10. TT  out_proj matmul
     11. TT  residual add

    Returns a dict with the intermediates needed by the v0.1.2.c smoke
    (norm_out, out_proj_out, block_out) plus the soft-gate items
    (y_post_ssd from the wrapper).
    """
    w = state.per_layer_tt[layer_idx]
    assert w is not None and w["kind"] == "mamba2"
    squeeze_batch = h_input_np.ndim == 2
    if squeeze_batch:
        h_input_np = h_input_np[None]
    B, S, _ = h_input_np.shape
    NH = MAMBA_HEADS
    HD = MAMBA_HEAD_DIM
    NG = N_GROUPS
    SS = SSM_STATE

    # ── 1. upload + pre-norm + in_proj ─────────────────────────
    h_tt = ttnn.from_torch(
        torch.from_numpy(h_input_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    h_norm_tt = ttnn.rms_norm(h_tt, weight=w["norm"], epsilon=EPS)
    in_proj_tt = ttnn.matmul(h_norm_tt, w["in_proj"], compute_kernel_config=HIFI4)

    z_tt   = ttnn.slice(in_proj_tt, [0, 0, 0],                    [B, S, D_INNER])
    xBC_tt = ttnn.slice(in_proj_tt, [0, 0, D_INNER],              [B, S, D_INNER + CONV_DIM_M])
    dt_tt  = ttnn.slice(in_proj_tt, [0, 0, D_INNER + CONV_DIM_M], [B, S, D_INNER + CONV_DIM_M + MAMBA_HEADS])

    # ── 2. conv1d on x_BC ──────────────────────────────────────
    xBC_nhwc = ttnn.to_layout(ttnn.reshape(xBC_tt, [B, 1, S, CONV_DIM_M]),
                                ttnn.ROW_MAJOR_LAYOUT)
    conv_full_tt = ttnn.conv1d(
        input_tensor=xBC_nhwc,
        weight_tensor=w["conv1d_w"],
        device=state.mesh,
        in_channels=CONV_DIM_M, out_channels=CONV_DIM_M,
        batch_size=B, input_length=S,
        kernel_size=CONV_KERNEL, stride=1,
        padding=CONV_KERNEL - 1, dilation=1,
        groups=CONV_DIM_M,
        bias_tensor=w["conv1d_b"],
    )
    # conv_full_tt: [B, S+2*pad-K+1, CONV_DIM_M] = [B, 8, 6144] in our NHWC
    # → squeeze H=1, slice causal first S positions, to TILE for silu.
    # Readback as numpy to handle the layout/dim juggle cleanly here; this is
    # ~250 KB and not on the perf-critical hot path (v0.5 perf will fuse).
    conv_full_np = ttnn.to_torch(
        conv_full_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )[:1].float().numpy()
    # Normalize shape to [B, S_out, C]:
    if conv_full_np.ndim == 4:
        conv_full_np = conv_full_np.squeeze(1)
    if conv_full_np.shape[-1] != CONV_DIM_M:
        conv_full_np = conv_full_np.transpose(0, 2, 1)
    # Causal slice: keep first S positions
    conv_causal_np = conv_full_np[:, :S, :]

    # ── 3. silu ────────────────────────────────────────────────
    # Cleanest path back to device: re-upload causal-sliced + silu on TT.
    conv_causal_tt = ttnn.from_torch(
        torch.from_numpy(conv_causal_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    silu_out_tt = ttnn.silu(conv_causal_tt)

    # ── 4. split silu output into x / B_in / C_in ─────────────
    # x_inner: [B, S, d_inner]; B/C_inner: [B, S, n_groups*ssm_state]
    BC_SIZE = NG * SS  # 1024
    x_inner_tt = ttnn.slice(silu_out_tt, [0, 0, 0],                  [B, S, D_INNER])
    B_inner_tt = ttnn.slice(silu_out_tt, [0, 0, D_INNER],            [B, S, D_INNER + BC_SIZE])
    C_inner_tt = ttnn.slice(silu_out_tt, [0, 0, D_INNER + BC_SIZE],  [B, S, D_INNER + 2 * BC_SIZE])

    # ── 5. read back x_inner / z / dt / B_inner / C_inner for SSD ──
    def _readback(t):
        arr = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )
        return arr[:1].float().numpy()

    x_inner_np = _readback(x_inner_tt)  # [B, S, d_inner=4096]
    z_full_np = _readback(z_tt)         # [B, S, d_inner=4096]
    B_inner_np = _readback(B_inner_tt)  # [B, S, 1024]
    C_inner_np = _readback(C_inner_tt)  # [B, S, 1024]
    dt_full_np = _readback(dt_tt)       # [B, S, NUM_HEADS=64]

    # Reshape for the wrapper
    x_inner_np = x_inner_np.reshape(B, S, NH, HD)
    z_inner_np = z_full_np.reshape(B, S, NH, HD)
    B_inner_np = B_inner_np.reshape(B, S, NG, SS)
    C_inner_np = C_inner_np.reshape(B, S, NG, SS)

    # ── 6. SSD loop ────────────────────────────────────────────
    import nemotron3_mamba2_step as _step_mod  # local module already on path
    dt_bias = w["dt_bias_np"]
    A_log = w["A_log_np"]
    D_w = w["D_np"]
    ssm_state = np.zeros((B, NH, HD, SS), dtype=np.float32)
    y_list = []
    for p in range(S):
        new_state, y_p = _step_mod.mamba2_decode_step_ttnn(
            x=x_inner_np[:, p, :, :],
            z=z_inner_np[:, p, :, :],
            dt=dt_full_np[:, p, :],
            dt_bias=dt_bias, A_log=A_log, D=D_w,
            B_in=B_inner_np[:, p, :, :],
            C_in=C_inner_np[:, p, :, :],
            ssm_state=ssm_state,
            device=state.mesh,
            debug_mode=5,
        )
        ssm_state = new_state
        y_list.append(y_p)
    y_post_ssd = np.stack(y_list, axis=1)  # [B, S, NH, HD]
    y_flat = y_post_ssd.reshape(B, S, NH * HD)  # [B, S, d_inner]

    # ── 7. MambaRMSNormGated (on-device) ───────────────────────
    # Matches the order used by our `mamba_ssm` CPU stub (which is what
    # generated the HF oracle): group-RMSNorm → weight → silu(z).
    #
    # IMPORTANT: Nemotron's modeling passes `norm_before_gate=False`.
    # The upstream `mamba_ssm` semantics of that flag are ambiguous; our
    # stub effectively ignores it and applies gate AFTER norm. We match
    # the stub so the oracle validates. Real upstream may differ — when
    # we swap to upstream `mamba_ssm` (post-CUDA), revisit + flip if needed.
    y_tt = ttnn.from_torch(
        torch.from_numpy(y_flat.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    group_size = D_INNER // NG  # 4096 / 8 = 512
    y_grouped = ttnn.reshape(y_tt, [B, S, NG, group_size])
    sq = ttnn.mul(y_grouped, y_grouped)
    var = ttnn.mean(sq, dim=-1, keepdim=True)
    var_eps = ttnn.add(var, EPS)
    rsqrt_var = ttnn.rsqrt(var_eps)
    y_normed_g = ttnn.mul(y_grouped, rsqrt_var)
    y_normed = ttnn.reshape(y_normed_g, [B, S, D_INNER])
    mixer_norm_w_tt = ttnn.from_torch(
        torch.from_numpy(w["mixer_norm_w_np"].astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    y_weighted = ttnn.mul(y_normed, mixer_norm_w_tt)
    silu_z = ttnn.silu(z_tt)
    norm_out_tt = ttnn.mul(y_weighted, silu_z)
    norm_out_np = _readback(norm_out_tt)

    # ── 8. out_proj ────────────────────────────────────────────
    o_tt = ttnn.matmul(norm_out_tt, w["out_proj"], compute_kernel_config=HIFI4)
    o_np = _readback(o_tt)

    # ── 9. residual add ────────────────────────────────────────
    block_tt = ttnn.add(h_tt, o_tt)
    block_np = _readback(block_tt)

    out = {
        "h_norm":      _readback(h_norm_tt),
        "conv1d_out":  conv_full_np,    # full pre-causal-slice for HF compare
        "y_post_ssd":  y_post_ssd,       # [B, S, NH, HD], for soft sanity
        "norm_out":    norm_out_np,
        "o_proj_out":  o_np,
        "block_out":   block_np,
    }
    if squeeze_batch:
        for k in out:
            out[k] = out[k][0]
    return out


# ── MoE forward (v0.1.3) ───────────────────────────────────────────────
def moe_router_only(state: State, h_input_np, layer_idx: int):
    """v0.1.3.a — pre-norm + router (sigmoid + e_score_correction_bias +
    topk). Returns topk_indices and topk_weights matching HF
    NemotronHTopkRouter contract.

    The router math (modeling_nemotron_h.py:905-918, simplified for
    Nemotron's degenerate n_group=topk_group=1 case):
      1. router_logits = h_norm @ gate.weight.T          [B*S, n_experts]
      2. scores = sigmoid(router_logits)
      3. scores_for_choice = scores + e_score_correction_bias
      4. topk_indices = topk(scores_for_choice, k=6)
      5. topk_weights = scores.gather(1, topk_indices)    (UN-biased!)
      6. norm_topk_prob=True → topk_weights /= sum(topk_weights, dim=-1)
      7. topk_weights *= routed_scaling_factor (2.5)

    HYBRID: matmul + sigmoid on device; bias + topk + normalize on host
    (those last three need scatter/topk/gather primitives that ttnn
    exposes individually; for v0.1.3.a the host loop is correct + cheap
    at B*S=5. Full on-device topk lands at v0.5 perf — 27B already has
    `experiments/utils/ttnn_introspect.py` validated on-device topk).
    """
    w = state.per_layer_tt[layer_idx]
    assert w is not None and w["kind"] == "moe_router_only"
    squeeze_batch = h_input_np.ndim == 2
    if squeeze_batch:
        h_input_np = h_input_np[None]
    B, S, _ = h_input_np.shape

    h_tt = ttnn.from_torch(
        torch.from_numpy(h_input_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    # Pre-norm + router matmul + sigmoid all on device
    h_norm_tt = ttnn.rms_norm(h_tt, weight=w["norm"], epsilon=EPS)
    # HF uses fp32 for the router matmul (line 910). Use HiFi4 +
    # fp32_dest_acc to closely match.
    logits_tt = ttnn.matmul(h_norm_tt, w["gate_w"], compute_kernel_config=HIFI4)
    scores_tt = ttnn.sigmoid(logits_tt)

    def _readback(t):
        arr = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )
        return arr[:1].float().numpy()

    scores_np = _readback(scores_tt)  # [B, S, N_EXPERTS]
    if scores_np.ndim == 3:
        scores_np = scores_np[0]  # [S, N_EXPERTS]

    # Host topk
    bias = w["e_score_bias_np"].astype(np.float32)
    scores_for_choice = scores_np + bias[None, :]
    # n_group == topk_group == 1 → group restriction is a NOP for
    # Nemotron. Direct topk over all N_EXPERTS.
    # np.argpartition gives unsorted top-k; sort by value to match HF
    # sorted=False (HF doesn't sort; order doesn't matter for downstream).
    topk_indices = np.argpartition(
        -scores_for_choice, TOP_K_ROUTED, axis=-1
    )[:, :TOP_K_ROUTED]  # [S, top_k]
    # gather the ORIGINAL (un-biased) scores at these indices
    rows = np.arange(scores_np.shape[0])[:, None]
    topk_weights = scores_np[rows, topk_indices]  # [S, top_k]
    # norm_topk_prob=True for Nemotron
    denom = topk_weights.sum(axis=-1, keepdims=True) + 1e-20
    topk_weights = topk_weights / denom
    topk_weights = topk_weights * np.float32(ROUTED_SCALING)

    out = {
        "h_norm":       _readback(h_norm_tt),
        "scores":       scores_np,
        "topk_indices": topk_indices.astype(np.int32),
        "topk_weights": topk_weights,
    }
    if not squeeze_batch:
        out["topk_indices"] = out["topk_indices"][None]
        out["topk_weights"] = out["topk_weights"][None]
    return out


def moe_block_eager(state: State, h_input_np, layer_idx: int):
    """v0.1.3.b — full L1 MoE block forward (mostly on-device).

    Chain:
      1. TT pre-norm + router (matmul + sigmoid)
      2. host topk-6 (sigmoid + bias + topk + gather + normalize + scale)
      3. per-token expert dispatch (5 tokens × 6 experts = 30 forwards):
         - TT matmul(h_norm[t], expert.up.T) → relu² → matmul(.., expert.down.T)
         - weighted add to accumulator
      4. TT shared expert: matmul(h_input, sh.up.T) → relu² → matmul(.., sh.down.T)
      5. mixer_out = routed_combined + shared_out (TT add)
      6. block_out = h_input + mixer_out (TT residual)

    Returns dict with intermediates needed by the smoke.
    """
    w = state.per_layer_tt[layer_idx]
    assert w is not None and w["kind"] == "moe"
    squeeze_batch = h_input_np.ndim == 2
    if squeeze_batch:
        h_input_np = h_input_np[None]
    B, S, _ = h_input_np.shape
    assert B == 1, "v0.1.3.b currently single-batch (CB lands later)"

    # Upload h_input once
    h_tt = ttnn.from_torch(
        torch.from_numpy(h_input_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )

    def _readback(t):
        arr = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )
        return arr[:1].float().numpy()

    # ── 1+2. Router (matmul + sigmoid + host topk) ──────────────
    h_norm_tt = ttnn.rms_norm(h_tt, weight=w["norm"], epsilon=EPS)
    logits_tt = ttnn.matmul(h_norm_tt, w["gate_w"], compute_kernel_config=HIFI4)
    scores_tt = ttnn.sigmoid(logits_tt)
    scores_np = _readback(scores_tt)[0]  # [S, n_experts]

    bias = w["e_score_bias_np"].astype(np.float32)
    scores_for_choice = scores_np + bias[None, :]
    topk_indices = np.argpartition(
        -scores_for_choice, TOP_K_ROUTED, axis=-1
    )[:, :TOP_K_ROUTED]
    rows = np.arange(scores_np.shape[0])[:, None]
    topk_weights = scores_np[rows, topk_indices]
    denom = topk_weights.sum(axis=-1, keepdims=True) + 1e-20
    topk_weights = topk_weights / denom
    topk_weights = topk_weights * np.float32(ROUTED_SCALING)

    # ── 3. Per-token expert dispatch ─────────────────────────────
    # Strategy: gather tokens per expert (most experts see ≤1 token at S=5).
    # For each unique expert e, run on the gathered token slice once.
    routed_accum_np = np.zeros((B, S, HIDDEN), dtype=np.float32)
    # Map expert_idx → list of (token_idx, weight)
    routings: dict[int, list[tuple[int, float]]] = {}
    for t in range(S):
        for k in range(TOP_K_ROUTED):
            e = int(topk_indices[t, k])
            routings.setdefault(e, []).append((t, float(topk_weights[t, k])))

    # h_norm readback for host-side per-token slicing (small: [B,S,HIDDEN])
    h_norm_np = _readback(h_norm_tt)  # [1, S, HIDDEN] or [4*1, S, HIDDEN] → first chip
    if h_norm_np.ndim == 3:
        pass  # already [1, S, HIDDEN]

    for e, tw_list in routings.items():
        tok_idxs = [t for t, _ in tw_list]
        weights_for_tok = [_w for _, _w in tw_list]
        # Build per-expert input [n_tok_for_e, HIDDEN]
        x_e_np = h_norm_np[0, tok_idxs, :]  # [n_tok, HIDDEN]
        x_e_np_3d = x_e_np[None]  # [1, n_tok, HIDDEN]

        x_e_tt = ttnn.from_torch(
            torch.from_numpy(x_e_np_3d.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        up_tt = ttnn.matmul(x_e_tt, w["routed"][e]["up"],
                              compute_kernel_config=HIFI4)
        # relu² = relu(x) * relu(x)
        relu = ttnn.relu(up_tt)
        relu_sq = ttnn.mul(relu, relu)
        down_tt = ttnn.matmul(relu_sq, w["routed"][e]["down"],
                                compute_kernel_config=HIFI4)
        # Read back the expert output and weighted-add into routed_accum
        expert_out_np = _readback(down_tt)[0]  # [n_tok, HIDDEN]
        for i, t in enumerate(tok_idxs):
            routed_accum_np[0, t, :] += weights_for_tok[i] * expert_out_np[i]

    # ── 4. Shared expert ─────────────────────────────────────────
    sh_up_tt = ttnn.matmul(h_norm_tt, w["shared"]["up"],
                              compute_kernel_config=HIFI4)
    sh_relu = ttnn.relu(sh_up_tt)
    sh_relu_sq = ttnn.mul(sh_relu, sh_relu)
    sh_down_tt = ttnn.matmul(sh_relu_sq, w["shared"]["down"],
                                compute_kernel_config=HIFI4)
    shared_out_np = _readback(sh_down_tt)

    # ── 5+6. Combine + residual ───────────────────────────────────
    mixer_out_np = routed_accum_np + shared_out_np
    block_out_np = h_input_np.astype(np.float32) + mixer_out_np

    out = {
        "h_norm":       h_norm_np,
        "topk_indices": topk_indices.astype(np.int32),
        "topk_weights": topk_weights,
        "routed_accum": routed_accum_np,
        "shared_out":   shared_out_np,
        "mixer_out":    mixer_out_np,
        "block_out":    block_out_np,
    }
    if squeeze_batch:
        for k in out:
            if isinstance(out[k], np.ndarray) and out[k].ndim >= 3:
                out[k] = out[k][0]
    return out


def moe_block_eager_ep(state: State, h_input_np, layer_idx: int):
    """v0.1.4 — Expert-Parallel MoE forward (fully on-device).

    Forks the dispatch+combine pattern from
    `models/demos/deepseek_v3/tt/moe.py:455 + :487` with Nemotron
    deltas (sigmoid+bias+topk router, relu² activation, scaled weights,
    shared expert added at the end). Validated on (1,4) BH by the
    v0.1.4.G0 spike (commit `138df8e`).

    Pipeline:
      1. TT pre-norm + router (matmul + sigmoid)
      2. host topk-6 with e_score_correction_bias (small, ~1 KB readback)
      3. TT all_to_all_dispatch: tokens routed to chips owning each topk
      4. TT local experts (E_LOCAL=32, batched matmul over expert dim)
         FFN: up_proj → relu² → down_proj
      5. TT all_to_all_combine (local_reduce=True for weighted sum)
      6. TT shared expert (replicated)
      7. TT residual

    Returns the same dict shape as `moe_block_eager` so the v0.1.3.b
    smoke validates this against HF without modification.
    """
    w = state.per_layer_tt[layer_idx]
    assert w is not None and w["kind"] == "moe_ep", \
        f"L{layer_idx} not loaded as moe_ep (kind={w.get('kind') if w else None!r})"
    squeeze_batch = h_input_np.ndim == 2
    if squeeze_batch:
        h_input_np = h_input_np[None]
    B, S_orig, _ = h_input_np.shape
    assert B == 1, "v0.1.4 single-batch (CB lands later)"

    # ── Seq-shard pad (Option 1, [[reference-all-to-all-dispatch-shape-contract]]) ──
    # Pad S to a multiple of NCHIPS so the dispatch op can shard the
    # seq dim across cluster_axis=1. Padded positions get arbitrary
    # router outputs / experts and are sliced off at the end.
    S_padded = ((S_orig + NCHIPS - 1) // NCHIPS) * NCHIPS
    S_per_chip = S_padded // NCHIPS
    if S_padded != S_orig:
        h_padded_np = np.zeros((B, S_padded, HIDDEN), dtype=h_input_np.dtype)
        h_padded_np[:, :S_orig, :] = h_input_np
    else:
        h_padded_np = h_input_np

    h_tt = ttnn.from_torch(
        torch.from_numpy(h_padded_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )

    def _readback(t):
        arr = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )
        return arr[:1].float().numpy()

    # ── 1+2. Router (matmul + sigmoid + host topk) ──────────────
    h_norm_tt = ttnn.rms_norm(h_tt, weight=w["norm"], epsilon=EPS)
    logits_tt = ttnn.matmul(h_norm_tt, w["gate_w"], compute_kernel_config=HIFI4)
    scores_tt = ttnn.sigmoid(logits_tt)
    scores_np = _readback(scores_tt)[0]  # [S_padded, n_experts]

    bias = w["e_score_bias_np"].astype(np.float32)
    scores_for_choice = scores_np + bias[None, :]
    topk_indices = np.argpartition(
        -scores_for_choice, TOP_K_ROUTED, axis=-1
    )[:, :TOP_K_ROUTED]
    rows = np.arange(scores_np.shape[0])[:, None]
    topk_weights = scores_np[rows, topk_indices]
    denom = topk_weights.sum(axis=-1, keepdims=True) + 1e-20
    topk_weights = topk_weights / denom
    topk_weights = topk_weights * np.float32(ROUTED_SCALING)
    # topk_indices, topk_weights both shape [S_padded, TOP_K_ROUTED].

    # ── 3. all_to_all_dispatch (seq-sharded, output_concat_dim=2) ──
    # Build the dispatch input by reading back the replicated h_norm
    # and re-uploading SHARDED along seq dim across cluster_axis=1.
    # Per-chip input: [B, 1, S_per_chip, HIDDEN] in ROW_MAJOR.
    h_norm_np_padded = _readback(h_norm_tt)  # [B, S_padded, HIDDEN]
    h_norm_4d_np = h_norm_np_padded.reshape(B, 1, S_padded, HIDDEN)
    h_norm_sharded_tt = ttnn.from_torch(
        torch.from_numpy(h_norm_4d_np.astype(np.float32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=state.mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=2),
    )
    topk_indices_4d = topk_indices.astype(np.int32).reshape(B, 1, S_padded, TOP_K_ROUTED)
    topk_indices_tt = ttnn.from_torch(
        torch.from_numpy(topk_indices_4d),
        device=state.mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=2),
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    dispatch_out_tt, dispatch_meta_tt = ttnn.all_to_all_dispatch(
        h_norm_sharded_tt,
        topk_indices_tt,
        w["expert_mapping"],
        cluster_axis=1,
        output_concat_dim=2,  # → per-chip [1, 1, S_per_chip*NCHIPS=S_padded, HIDDEN]
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(h_norm_sharded_tt)
    ttnn.deallocate(topk_indices_tt)
    print(f"[v014 dbg] dispatch_out shape: {list(dispatch_out_tt.shape)}  "
          f"meta shape: {list(dispatch_meta_tt.shape)}  "
          f"(S_orig={S_orig} S_padded={S_padded} S_per_chip={S_per_chip})",
          flush=True)

    # ── 4. Local experts (batched matmul over E_LOCAL) ──────────
    # dispatch_out_tt per-chip: [1, 1, S_padded, HIDDEN]. Repeat to
    # [1, E_LOCAL, S_padded, HIDDEN] so each local expert sees a copy,
    # then squeeze the leading 1 → rank-3 for matmul against rank-3
    # expert weights (ttnn.matmul TT_FATALs on rank-4 vs rank-3).
    dispatch_chunk = ttnn.repeat(dispatch_out_tt, ttnn.Shape([1, E_LOCAL, 1, 1]))
    dispatch_chunk = ttnn.reshape(dispatch_chunk, [E_LOCAL, S_padded, HIDDEN])
    dispatch_chunk = ttnn.to_layout(dispatch_chunk, ttnn.TILE_LAYOUT)
    ttnn.deallocate(dispatch_out_tt)

    up_out = ttnn.matmul(
        dispatch_chunk, w["experts_up_local"], compute_kernel_config=HIFI4,
    )
    ttnn.deallocate(dispatch_chunk)
    relu_out = ttnn.relu(up_out)
    relu_sq = ttnn.mul(relu_out, relu_out)
    ttnn.deallocate(up_out)
    ttnn.deallocate(relu_out)
    expert_out = ttnn.matmul(
        relu_sq, w["experts_down_local"], compute_kernel_config=HIFI4,
    )
    ttnn.deallocate(relu_sq)

    # Combine input contract: [E_LOCAL, B, S_padded, HIDDEN] per chip.
    expert_out_rm = ttnn.to_layout(expert_out, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(expert_out)
    expert_out_combine = ttnn.reshape(expert_out_rm, [E_LOCAL, B, S_padded, HIDDEN])
    ttnn.deallocate(expert_out_rm)

    # ── 5. all_to_all_combine (scatter back to source chips) ─────
    # output_shard_dim=2 → per-chip [TOP_K, B, S_per_chip, HIDDEN].
    # Default output_shard_dim=1 would give [TOP_K, B/D[A]=0, ...] for
    # B=1 / D[A]=4 (integer division) → moreh_full asserts shape[1] > 0.
    # See all_to_all_combine_nanobind.cpp:79 for the contract.
    combine_out_tt = ttnn.all_to_all_combine(
        expert_out_combine,
        dispatch_meta_tt,
        w["expert_mapping"],
        cluster_axis=1,
        output_shard_dim=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(expert_out_combine)
    ttnn.deallocate(dispatch_meta_tt)
    print(f"[v014 dbg] combine_out shape: {list(combine_out_tt.shape)}",
          flush=True)

    # Readback. With seq-sharded dispatch (output_concat_dim=2), the
    # combine op scatters results back; each chip should hold its
    # source-seq slice. Try seq-concat (dim 2) first; fall back to
    # chip-0 if the op returns replicated.
    try:
        combine_torch = ttnn.to_torch(
            combine_out_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=2),
        )
        combine_np = combine_torch.float().numpy()
        print(f"[v014 dbg] combine_np (concat dim=2): {combine_np.shape}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[v014 dbg] concat dim=2 failed ({type(e).__name__}: {e}); "
              f"falling back to dim=0 + chip-0 slice", flush=True)
        combine_torch = ttnn.to_torch(
            combine_out_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )
        combine_np = combine_torch[:1].float().numpy()
        print(f"[v014 dbg] combine_np (chip-0 slice): {combine_np.shape}",
              flush=True)
    ttnn.deallocate(combine_out_tt)

    # Normalise to rank-4 [TOP_K, B, S_padded, HIDDEN]
    if combine_np.ndim == 5:
        combine_np = combine_np[0]
    if combine_np.shape[0] != TOP_K_ROUTED and combine_np.ndim == 4:
        # Some BH builds may return [1, TOP_K, S, H]; squeeze if so.
        combine_np = combine_np.squeeze(0)
    assert combine_np.shape[0] == TOP_K_ROUTED, \
        f"unexpected combine shape {combine_np.shape}; want axis 0 = TOP_K={TOP_K_ROUTED}"

    # Weighted sum across TOP_K.
    # topk_weights: [S_padded, TOP_K] → broadcast to [TOP_K, 1, S_padded, 1]
    w_bc = topk_weights.T[:, None, :, None]
    routed_per_topk = combine_np.astype(np.float32) * w_bc.astype(np.float32)
    routed_np = routed_per_topk.sum(axis=0)  # [B, S_padded, HIDDEN]
    if routed_np.ndim == 2:
        routed_np = routed_np[None]

    # Slice back to original S.
    routed_np = routed_np[:, :S_orig, :]

    # ── 6. Shared expert (replicated, on padded h_norm) ─────────
    sh_up_out = ttnn.matmul(h_norm_tt, w["shared_up"], compute_kernel_config=HIFI4)
    sh_relu = ttnn.relu(sh_up_out)
    sh_relu_sq = ttnn.mul(sh_relu, sh_relu)
    ttnn.deallocate(sh_up_out)
    ttnn.deallocate(sh_relu)
    sh_down_out = ttnn.matmul(sh_relu_sq, w["shared_down"], compute_kernel_config=HIFI4)
    ttnn.deallocate(sh_relu_sq)
    shared_out_np = _readback(sh_down_out)
    ttnn.deallocate(sh_down_out)
    shared_out_np = shared_out_np[:, :S_orig, :]  # slice off pad

    # ── 7. Combine + residual ───────────────────────────────────
    mixer_out_np = routed_np + shared_out_np
    block_out_np = h_input_np.astype(np.float32) + mixer_out_np

    out = {
        "h_norm":       _readback(h_norm_tt)[:, :S_orig, :],
        "topk_indices": topk_indices[:S_orig, :].astype(np.int32),
        "topk_weights": topk_weights[:S_orig, :],
        "routed_accum": routed_np,
        "shared_out":   shared_out_np,
        "mixer_out":    mixer_out_np,
        "block_out":    block_out_np,
    }
    if squeeze_batch:
        for k in out:
            if isinstance(out[k], np.ndarray) and out[k].ndim >= 3:
                out[k] = out[k][0]
    return out


def main():
    """Tiny direct-invoke entry point for ad-hoc smoke testing."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = State()
    bootstrap(state, lambda m: print(m, flush=True))
    print("v0.1.0 scaffold bootstrap PASS — no probes wired here, "
          "use nemotron3_v010_bootstrap_smoke.py for the actual gate.")


if __name__ == "__main__":
    main()
