#!/usr/bin/env python3
"""Gemma 4 12B unified — text-only TT-Metal server on (1,4) Blackhole mesh.

Fork base: `experiments/serve/server_35b_ttnn.py` per REUSE MANDATE
(`research/gemma4_12b_bringup_plan.md` §"REUSE MANDATE"). Diff from 35B:

- DENSE (no MoE, no DeltaNet). Single `mlp(gate, up, down)` per layer.
- Sliding+global hybrid (5 sliding / 1 global × 8 = 48 layers). v0.3 will
  add `sliding_window_size=1024` kwarg to paged SDPA decode (Step 0.1 PASS).
- Four norms per decoder layer (Gemma 2 pattern §1.5):
  input_layernorm, post_attention_layernorm, pre_feedforward_layernorm,
  post_feedforward_layernorm. Each post-norm comes BEFORE its residual add.
- Llama-style RMSNorm: `y = x/rms * w` (NOT 27B/35B's `* (1+w)`). DO NOT
  add 1.0 at upload time. [[qwen36-qnorm-knorm-zero-centered]] is for Qwen3.6.
- Embed scale: multiply embed lookup by `sqrt(HIDDEN) = sqrt(3840) ≈ 61.97`.
- Final logit softcap: `logits = 30·tanh(logits/30)` before sampling.
- GELU_tanh activation: Step 0.2 found `ttnn.gelu(fast_and_approximate_mode=False)`
  matches `gelu_pytorch_tanh` at cos=0.99999803; the fused-activation path
  (`ttnn.mul(.., [UnaryOpType.GELU])`) uses the APPROXIMATE kernel — do
  NOT mirror 35B's SwiGLU fused pattern.
- Tied embeddings (`tie_word_embeddings=True`): lm_head ≡ embed_table.T.
- `attention_k_eq_v` on global layers only — sliding layers project K,V
  independently even when the flag is set.
- Dual head_dim: 256 (sliding) vs 512 (global). v0.1 handles sliding only.

## Staging

- **v0.1.0** (THIS file initially): bootstrap (open mesh, upload embed
  + all 48 layer weights), single-step forward at pos 0 that produces
  the rms_norm output (input_layernorm output for L0). Smallest
  deterministic chunk to prove bootstrap correctness.
- v0.1.1: add q_proj/k_proj/v_proj + q_norm/k_norm; compare vs HF
  per-sub-step.
- v0.1.2: add attention (manual, pos 0 collapses softmax to 1).
- v0.1.3: add MLP + residuals; full L0 forward; matches HF[1, 0, :].
- v0.2: extend forward to all 48 layers (sliding + global dispatch).
- v0.3: KV cache + paged SDPA with `sliding_window_size` kwarg.
- v0.4: trace capture.

Run via cosine probe at `experiments/cb/isolate/gm4_v01_L0_cos.py`.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

# Transformers in qb1's main .venv is too old to recognize `gemma4_unified`
# (model released 2026-06-03). We read `config.json` from the snapshot
# directly so the SERVER doesn't depend on the bleeding-edge transformers
# install. Tokenizer is deferred to v0.2 (needed only for chat). See
# `scripts/setup_venv_gemma4.sh` for the HF-reference / oracle venv.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402

# ── Model constants (plan §1.1, verified from config.json) ──────────────
MODEL_ID = "google/gemma-4-12B"
HIDDEN = 3840
NUM_LAYERS = 48
NUM_Q_HEADS = 16
NUM_KV_HEADS_SLIDING = 8
NUM_KV_HEADS_GLOBAL = 1
HEAD_DIM_SLIDING = 256
HEAD_DIM_GLOBAL = 512
INTERMEDIATE = 15360
VOCAB = 262144
ROPE_THETA_SLIDING = 10000.0
ROPE_THETA_GLOBAL = 1000000.0
PARTIAL_ROTARY_GLOBAL = 0.25
SLIDING_WINDOW = 1024
EMBED_SCALE = math.sqrt(HIDDEN)  # ≈ 61.97
FINAL_LOGIT_SOFTCAP = 30.0
EPS = 1e-6

NCHIPS = 4
NQ_PER_CHIP = NUM_Q_HEADS // NCHIPS  # 4
NKV_PER_CHIP_SLIDING = NUM_KV_HEADS_SLIDING // NCHIPS  # 2
GQA_GROUP_SLIDING = NUM_Q_HEADS // NUM_KV_HEADS_SLIDING  # 2 (Q heads per KV head)
HIDDEN_PER_CHIP = HIDDEN // NCHIPS  # 960
INTERMEDIATE_PER_CHIP = INTERMEDIATE // NCHIPS  # 3840

MAX_KV = 4096  # v0.1 doesn't use KV; bump for v0.3 paged SDPA

# 91f recipe: HiFi4 + fp32_dest_acc on every matmul. Same lesson holds for
# Gemma 4 — bf16 chain drift accumulates without it [[bf16-chain-drift-at-B-gt-1]].
HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ── Upload helpers (reused from 35B per REUSE MANDATE) ─────────────────
def np_to_replicated(arr, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch.from_numpy(arr.astype(np.float32)),
        dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def np_stacked_to_sharded(per_chip_list, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Stack a list of NCHIPS numpy arrays as the leading axis; shard along it.

    Uses `ShardTensorToMesh(dim=0)` (1D sharder) — matches 35B's
    `np_stacked_to_sharded` (`server_35b_ttnn.py:96`). The 2D variant
    `ShardTensor2dMesh` keeps the leading 4-dim on each chip's tensor
    which breaks downstream matmul (`a=1 vs b=4` shape mismatch).
    """
    stacked = np.stack(per_chip_list, axis=0).astype(np.float32)
    return ttnn.from_torch(
        torch.from_numpy(stacked),
        dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def shard_along(arr, axis, n=NCHIPS):
    """Split a numpy array into n equal shards along an axis."""
    return [s.copy() for s in np.split(arr, n, axis=axis)]


def load_t(key_to_shard, key):
    path = key_to_shard[key]
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key).float().numpy()


def build_key_to_shard():
    """Build {key -> shard_path} index for google/gemma-4-12B safetensors."""
    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-4-12B" / "snapshots"
    if not snapshot_root.exists():
        raise FileNotFoundError(f"no HF snapshot at {snapshot_root}. "
                                f"Run hf_reference_gemma4_12b.py once to fetch.")
    snap = next(snapshot_root.iterdir())
    index = snap / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text())
        return {k: str(snap / v) for k, v in idx["weight_map"].items()}
    # single-file fallback (smaller models)
    sf = next(snap.glob("*.safetensors"))
    with safe_open(sf, framework="pt") as f:
        return {k: str(sf) for k in f.keys()}


# ── Per-layer attention weight upload ──────────────────────────────────
def upload_attn_layer_sliding(layer_sd, mesh):
    """Upload Q/K/V/o projections + q_norm/k_norm for a SLIDING layer.

    Sharding:
    - q_proj: [HIDDEN, NUM_Q_HEADS * HEAD_DIM] = [3840, 16*256=4096]
              → shard along OUTPUT axis (NUM_Q_HEADS sharded): each chip
                holds [3840, NQ_PER_CHIP*HEAD_DIM=1024]
    - k_proj, v_proj: [HIDDEN, NUM_KV_HEADS_SLIDING * HEAD_DIM] = [3840, 2048]
              → shard along OUTPUT axis (NUM_KV_HEADS sharded): each chip
                holds [3840, NKV_PER_CHIP_SLIDING*HEAD_DIM=512]
    - o_proj: [NUM_Q_HEADS * HEAD_DIM, HIDDEN] = [4096, 3840]
              → shard along INPUT axis (matches Q-head sharding for the
                column-sharded matmul + all_reduce pattern)
    - q_norm, k_norm: [HEAD_DIM=256] replicated (per-head RMSNorm)

    HF weights are stored as [in, out] per nn.Linear; we transpose to
    [out, in] for ttnn matmul column-major shape conventions.
    """
    w = {}
    # HF Linear stores `weight` as [out, in]; for ttnn.matmul(activation, W)
    # we want W as [in, out]. Transpose at upload.
    q_w = layer_sd["self_attn.q_proj.weight"].T  # [HIDDEN, NUM_Q * HEAD_DIM]
    k_w = layer_sd["self_attn.k_proj.weight"].T  # [HIDDEN, NUM_KV * HEAD_DIM]
    v_w = layer_sd["self_attn.v_proj.weight"].T  # [HIDDEN, NUM_KV * HEAD_DIM]
    o_w = layer_sd["self_attn.o_proj.weight"].T  # [NUM_Q * HEAD_DIM, HIDDEN]

    w["q_proj"] = np_stacked_to_sharded(shard_along(q_w, axis=1), mesh)
    w["k_proj"] = np_stacked_to_sharded(shard_along(k_w, axis=1), mesh)
    w["v_proj"] = np_stacked_to_sharded(shard_along(v_w, axis=1), mesh)
    w["o_proj"] = np_stacked_to_sharded(shard_along(o_w, axis=0), mesh)
    # Gemma 4 Llama-style RMSNorm: `w` directly. NO +1.0.
    w["q_norm"] = np_to_replicated(layer_sd["self_attn.q_norm.weight"], mesh)
    w["k_norm"] = np_to_replicated(layer_sd["self_attn.k_norm.weight"], mesh)
    return w


def upload_attn_layer_global(layer_sd, mesh):
    """Upload Q/K/V/o for a GLOBAL layer (head_dim=512, NKV=1, p-RoPE).

    Plan §6.8: with NUM_KV_HEADS_GLOBAL=1 on (1,4) mesh, KV cannot be
    sharded — replicate it across chips. v0.3 will resolve.

    `attention_k_eq_v=True` on global layers means v_proj is None and V
    aliases K post-norm. v0.2 will skip v_proj upload for global layers.
    """
    w = {}
    q_w = layer_sd["self_attn.q_proj.weight"].T
    k_w = layer_sd["self_attn.k_proj.weight"].T
    o_w = layer_sd["self_attn.o_proj.weight"].T
    w["q_proj"] = np_stacked_to_sharded(shard_along(q_w, axis=1), mesh)
    w["k_proj"] = np_to_replicated(k_w, mesh)  # NKV=1, replicate
    w["o_proj"] = np_stacked_to_sharded(shard_along(o_w, axis=0), mesh)
    w["q_norm"] = np_to_replicated(layer_sd["self_attn.q_norm.weight"], mesh)
    w["k_norm"] = np_to_replicated(layer_sd["self_attn.k_norm.weight"], mesh)
    if "self_attn.v_proj.weight" in layer_sd:
        v_w = layer_sd["self_attn.v_proj.weight"].T
        w["v_proj"] = np_to_replicated(v_w, mesh)
    return w


def upload_mlp_layer(layer_sd, mesh):
    """Upload MLP (gate, up, down) projections for a dense layer.

    Standard SwiGLU shape, GELU(gate) × up activation:
    - gate_proj, up_proj: [HIDDEN, INTERMEDIATE] = [3840, 15360]
              → shard along OUTPUT axis (each chip has [3840, 3840])
    - down_proj: [INTERMEDIATE, HIDDEN] = [15360, 3840]
              → shard along INPUT axis (matches column-sharded pattern)
    """
    w = {}
    gate_w = layer_sd["mlp.gate_proj.weight"].T  # [HIDDEN, INTERMEDIATE]
    up_w   = layer_sd["mlp.up_proj.weight"].T
    down_w = layer_sd["mlp.down_proj.weight"].T  # [INTERMEDIATE, HIDDEN]
    w["gate_proj"] = np_stacked_to_sharded(shard_along(gate_w, axis=1), mesh)
    w["up_proj"]   = np_stacked_to_sharded(shard_along(up_w,   axis=1), mesh)
    w["down_proj"] = np_stacked_to_sharded(shard_along(down_w, axis=0), mesh)
    return w


def all_reduce_tt(x_tt, mesh):
    """All-reduce sum across the (1, NCHIPS) mesh; matches `server_35b_ttnn.py:367`.

    The simple `ttnn.all_reduce(cluster_axis=1)` is the right call on
    qb1's current ttnn build. `ttnn.experimental.all_reduce_async`'s
    signature was changed to require explicit barrier/scatter/gather
    semaphores; mistakenly copying that broke v0.1.2 first try.
    """
    return ttnn.all_reduce(x_tt, cluster_axis=1)


# ── State ──────────────────────────────────────────────────────────────
class State:
    """Per-process server state. v0.1.0 carries minimal fields; expands
    in v0.1.1..v0.3 with KV cache, cos/sin tables, page table, etc.
    """
    def __init__(self):
        self.mesh = None
        self.text_cfg = None
        self.layer_types = None
        self.tokenizer = None
        self.tok = None  # alias for cb_api convention
        self.embed_tt = None
        self.embed_w_np = None  # keep host copy for fallback / sanity
        self.final_norm_tt = None
        # No separate lm_head_tt — tied embeddings; we'll materialize as
        # transposed embed at v0.2 when we add lm_head.
        self.per_layer_tt = []  # [{...}, ...] per-layer weight dicts


# ── Bootstrap ──────────────────────────────────────────────────────────
def bootstrap(state, log=None):
    if log is None:
        log = print

    log("[bootstrap] open mesh + fabric…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    state.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
    log(f"  mesh: {state.mesh}")

    log("[bootstrap] config (from snapshot JSON; tokenizer deferred to v0.2)…")
    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-4-12B" / "snapshots"
    snap = next(snapshot_root.iterdir())
    cfg_json = json.loads((snap / "config.json").read_text())
    text_cfg_json = cfg_json["text_config"]
    state.text_cfg = text_cfg_json  # dict, not pydantic
    state.layer_types = list(text_cfg_json["layer_types"])
    log(f"  {len(state.layer_types)} layers; "
        f"{sum(1 for t in state.layer_types if t == 'sliding_attention')} sliding / "
        f"{sum(1 for t in state.layer_types if t == 'full_attention')} global")

    log("[bootstrap] enumerate shards + load top-level weights to mesh…")
    key_to_shard = build_key_to_shard()
    log(f"  {len(key_to_shard)} keys total")

    # Gemma 4's v_norm is `RMSNorm(head_dim, with_scale=False)` — pure
    # x/rms(x), no learnable weight. ttnn.rms_norm requires a weight; we
    # pass an all-ones tensor of size HEAD_DIM. One copy per head_dim.
    # Loaded once and reused across all 48 layers' v_norm calls.
    state.ones_head_dim_sliding = np_to_replicated(
        np.ones(HEAD_DIM_SLIDING, dtype=np.float32), state.mesh)
    state.ones_head_dim_global = np_to_replicated(
        np.ones(HEAD_DIM_GLOBAL, dtype=np.float32), state.mesh)

    # Embed: replicated, ROW_MAJOR (ttnn.embedding requires it).
    # Tied embeddings: same table will serve as lm_head.T at v0.2.
    embed_w_np = load_t(key_to_shard, "model.language_model.embed_tokens.weight")
    state.embed_tt = ttnn.from_torch(
        torch.from_numpy(embed_w_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    state.embed_w_np = embed_w_np
    log(f"  embed: {embed_w_np.shape}")

    # Final norm — Gemma 4 Llama RMSNorm convention, NO +1.0.
    final_norm_w = load_t(key_to_shard, "model.language_model.norm.weight")
    state.final_norm_tt = np_to_replicated(final_norm_w, state.mesh)
    log(f"  final_norm: {final_norm_w.shape}")

    # Tied lm_head (plan §1.8): lm_head ≡ embed_table.T. Upload separately
    # in [HIDDEN, VOCAB] layout for the final matmul. Memory: ~2 GB/chip
    # bf16 — fits easily in P150's 31.8 GB. For v0.2 we keep it replicated;
    # vocab-sharded variant can come later (see 27B's
    # `server_tp.py:1680-1687` pattern).
    state.lm_head_tt = np_to_replicated(embed_w_np.T, state.mesh)
    log(f"  lm_head (tied, replicated): {embed_w_np.T.shape}")

    # v0.3: KV cache + paged SDPA infrastructure (forked from 35B
    # `server_35b_ttnn.py:1700-1820` setup). Two head_dim variants —
    # sliding (256) and global (512); allocate caches and pos buffers
    # for each per layer.
    state.MAX_KV = MAX_KV
    state.sdpa_block_size = 32  # tile-height; matches 35B
    # cur_pos_buf: int32 [1] device-resident, written before each step
    # via update_input_buffers (v0.3.1+). For v0.3.0 we hard-set pos=0.
    state.cur_pos_buf = ttnn.from_torch(
        torch.zeros((1,), dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    # rot_idxs_buf: int32 [1] for RoPE cos/sin table indexing per step.
    state.rot_idxs_buf = ttnn.from_torch(
        torch.zeros((1,), dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    # page_table: identity mapping [1 slot, N blocks]; one block per
    # 32-token chunk up to MAX_KV. Replicated across chips.
    num_blocks = MAX_KV // state.sdpa_block_size
    page_table_np = np.arange(num_blocks, dtype=np.int32).reshape(1, num_blocks)
    state.page_table_tt = ttnn.from_torch(
        torch.from_numpy(page_table_np),
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    state.num_blocks = num_blocks
    log(f"  paged SDPA: MAX_KV={MAX_KV}, block_size={state.sdpa_block_size}, "
        f"num_blocks={num_blocks}")

    # KV caches per layer. Forked from `server_35b_ttnn.py:1455-1483` —
    # sliding layers have NKV=8 (2/chip) sharded along the chip axis; global
    # layers have NKV=1 replicated (plan §6.8 — wastes 4× memory but is the
    # simplest setup; revisit if memory pressures appear).
    state.kv_caches_tt = []
    for L in range(NUM_LAYERS):
        if state.layer_types[L] == "sliding_attention":
            # Logical [num_blocks, NCHIPS * NKV_PER_CHIP=8, block_size, head_dim]
            # sharded along dim=1 → per-chip [num_blocks, NKV_PER_CHIP=2, block, head_dim].
            cs = (state.num_blocks, NCHIPS * NKV_PER_CHIP_SLIDING,
                  state.sdpa_block_size, HEAD_DIM_SLIDING)
            init = np.zeros(cs, dtype=np.float32)
            kc = ttnn.from_torch(
                torch.from_numpy(init), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
            )
            vc = ttnn.from_torch(
                torch.from_numpy(init), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
            )
        else:  # full_attention (global)
            cs = (state.num_blocks, NUM_KV_HEADS_GLOBAL,
                  state.sdpa_block_size, HEAD_DIM_GLOBAL)
            init = np.zeros(cs, dtype=np.float32)
            kc = ttnn.from_torch(
                torch.from_numpy(init), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=state.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            )
            vc = ttnn.from_torch(
                torch.from_numpy(init), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=state.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            )
        state.kv_caches_tt.append((kc, vc))
    log(f"  KV caches: {NUM_LAYERS} layers allocated")

    # SDPA program/memory/compute configs (35B-style B3 recipe — HiFi2
    # +fp32_dest_acc_en=False per `[[fp32-sdpa-cliff-probe]]`; HiFi4+fp32
    # has a Blackhole bug at large positions).
    compute_grid = state.mesh.compute_with_storage_grid_size()
    shard_grid_sliding = ttnn.num_cores_to_corerangeset(
        NKV_PER_CHIP_SLIDING, compute_grid, row_wise=True)
    state.paged_write_mem_cfg_sliding = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid_sliding,
                       [state.sdpa_block_size, HEAD_DIM_SLIDING],
                       ttnn.ShardOrientation.ROW_MAJOR),
    )
    shard_grid_global = ttnn.num_cores_to_corerangeset(
        NUM_KV_HEADS_GLOBAL, compute_grid, row_wise=True)
    state.paged_write_mem_cfg_global = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid_global,
                       [state.sdpa_block_size, HEAD_DIM_GLOBAL],
                       ttnn.ShardOrientation.ROW_MAJOR),
    )
    state.paged_sdpa_progcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
        q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False,
    )
    state.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=False,
    )
    log("  SDPA program/memory/compute configs ready")

    # RoPE tables (plan §1.3) — TWO, one per layer type:
    # Sliding: rope_type=default, theta=10000, full head_dim=256 rotated.
    # Global:  rope_type=proportional (p-RoPE), theta=1e6, rotate first
    #          int(0.25 * 512 // 2) * 2 = 128 dims, leave rest at zero freq.
    positions = np.arange(MAX_KV, dtype=np.float64)
    inv_freq_sliding = 1.0 / (ROPE_THETA_SLIDING ** (
        np.arange(0, HEAD_DIM_SLIDING, 2, dtype=np.float64) / HEAD_DIM_SLIDING))
    ang_sliding = np.outer(positions, inv_freq_sliding)  # [MAX_KV, HEAD_DIM_SLIDING/2]
    cos_sliding = np.concatenate([np.cos(ang_sliding), np.cos(ang_sliding)], axis=1).astype(np.float32)
    sin_sliding = np.concatenate([np.sin(ang_sliding), np.sin(ang_sliding)], axis=1).astype(np.float32)
    state.cos_sliding_tt = np_to_replicated(cos_sliding, state.mesh,
                                            layout=ttnn.ROW_MAJOR_LAYOUT)
    state.sin_sliding_tt = np_to_replicated(sin_sliding, state.mesh,
                                            layout=ttnn.ROW_MAJOR_LAYOUT)

    rope_angles_global = int(PARTIAL_ROTARY_GLOBAL * HEAD_DIM_GLOBAL // 2)  # = 64
    inv_freq_rot = 1.0 / (ROPE_THETA_GLOBAL ** (
        np.arange(0, 2 * rope_angles_global, 2, dtype=np.float64) / HEAD_DIM_GLOBAL))
    inv_freq_global = np.concatenate(
        [inv_freq_rot, np.zeros(HEAD_DIM_GLOBAL // 2 - rope_angles_global, dtype=np.float64)])
    ang_global = np.outer(positions, inv_freq_global)
    cos_global = np.concatenate([np.cos(ang_global), np.cos(ang_global)], axis=1).astype(np.float32)
    sin_global = np.concatenate([np.sin(ang_global), np.sin(ang_global)], axis=1).astype(np.float32)
    state.cos_global_tt = np_to_replicated(cos_global, state.mesh,
                                           layout=ttnn.ROW_MAJOR_LAYOUT)
    state.sin_global_tt = np_to_replicated(sin_global, state.mesh,
                                           layout=ttnn.ROW_MAJOR_LAYOUT)
    log(f"  RoPE tables: sliding [{cos_sliding.shape}] + global [{cos_global.shape}] (MAX_KV={MAX_KV})")

    log(f"[bootstrap] uploading {NUM_LAYERS} layer weights to mesh…")
    t0 = time.time()
    state.per_layer_tt = []
    for L in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{L}."
        layer_sd = {k.replace(prefix, ""): load_t(key_to_shard, k)
                    for k in key_to_shard if k.startswith(prefix)}
        layer_tt = {}
        # All four norms: Gemma 4 uses Llama `w` (NO +1.0).
        layer_tt["input_layernorm"] = np_to_replicated(layer_sd["input_layernorm.weight"], state.mesh)
        layer_tt["post_attention_layernorm"] = np_to_replicated(layer_sd["post_attention_layernorm.weight"], state.mesh)
        layer_tt["pre_feedforward_layernorm"] = np_to_replicated(layer_sd["pre_feedforward_layernorm.weight"], state.mesh)
        layer_tt["post_feedforward_layernorm"] = np_to_replicated(layer_sd["post_feedforward_layernorm.weight"], state.mesh)

        if state.layer_types[L] == "sliding_attention":
            layer_tt.update(upload_attn_layer_sliding(layer_sd, state.mesh))
        else:
            layer_tt.update(upload_attn_layer_global(layer_sd, state.mesh))
        layer_tt.update(upload_mlp_layer(layer_sd, state.mesh))
        # Gemma 4 NEW: per-layer learned scalar applied at the END of the
        # decoder layer (after both residual adds): `h *= layer_scalar`.
        # See `Gemma4UnifiedTextDecoderLayer.forward:540` in HF. 27B/35B
        # don't have this. Without it the residual stream propagates with
        # wrong magnitudes; L0 cos~1 but mad ~18× too high, then L1+ collapse.
        layer_tt["layer_scalar"] = float(np.asarray(layer_sd["layer_scalar"]).reshape(-1)[0])
        state.per_layer_tt.append(layer_tt)
        if (L + 1) % 10 == 0:
            log(f"  layer {L+1}/{NUM_LAYERS} uploaded ({time.time()-t0:.1f}s)")
    log(f"  all layer weights uploaded in {time.time()-t0:.1f}s")
    log("[bootstrap] ready (v0.1.0 — embed + L0 input_layernorm only).")


# ── v0.1.0 + v0.1.1 forward: embed → scale → L0 input_layernorm →
#                              Q/K/V proj → q_norm/k_norm ─────────────
def step_forward_v01(state, tok_id, capture):
    """v0.1.1 forward at pos 0 (L0 only, sliding attention path).

    Stages (each adds a capture entry):
      v0.1.0:  embed_scaled, in_norm
      v0.1.1:  q_proj_out, k_proj_out, v_proj_out, q_norm_out, k_norm_out

    All capture arrays are returned in [HEAD, HEAD_DIM] order matching HF's
    `attn_L0_<sub>` shape (post-reshape from flat [head*dim]).
    """
    # ── v0.1.0: embed lookup + sqrt(HIDDEN) scale + input_layernorm ──
    tok_tt = ttnn.from_torch(
        torch.tensor([[int(tok_id)]], dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    embed = ttnn.embedding(tok_tt, state.embed_tt)
    ttnn.deallocate(tok_tt)
    h = ttnn.to_layout(embed, ttnn.TILE_LAYOUT)
    ttnn.deallocate(embed)
    h_scaled = ttnn.multiply(h, EMBED_SCALE)
    ttnn.deallocate(h)
    capture["embed_scaled"] = _readback_replicated(h_scaled, state.mesh)

    w0 = state.per_layer_tt[0]
    in_norm = ttnn.rms_norm(h_scaled, weight=w0["input_layernorm"], epsilon=EPS)
    ttnn.deallocate(h_scaled)
    capture["in_norm"] = _readback_replicated(in_norm, state.mesh)

    # ── v0.1.1: Q/K/V projections + per-head q_norm/k_norm ──
    # Q proj: replicated [1, HIDDEN] @ sharded [HIDDEN, NQ_PER_CHIP * head_dim]
    # → per-chip [1, NQ_PER_CHIP * head_dim].
    q = ttnn.matmul(in_norm, w0["q_proj"], compute_kernel_config=HIFI4)
    k = ttnn.matmul(in_norm, w0["k_proj"], compute_kernel_config=HIFI4)
    v = ttnn.matmul(in_norm, w0["v_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(in_norm)

    # Capture POST-PROJECTION shapes, reassembled to [HEAD, HEAD_DIM].
    capture["q_proj_out"] = _readback_sharded_head(q, state.mesh, NQ_PER_CHIP, HEAD_DIM_SLIDING)
    capture["k_proj_out"] = _readback_sharded_head(k, state.mesh, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING)
    capture["v_proj_out"] = _readback_sharded_head(v, state.mesh, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING)

    # Per-head RMSNorm: q_norm operates on the last (head_dim) axis. We
    # reshape per chip to [NQ_PER_CHIP, HEAD_DIM] so rms_norm normalizes
    # along HEAD_DIM. Weight is replicated [HEAD_DIM] across chips.
    q_h = ttnn.reshape(q, [NQ_PER_CHIP, HEAD_DIM_SLIDING])
    ttnn.deallocate(q)
    k_h = ttnn.reshape(k, [NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING])
    ttnn.deallocate(k)
    q_n = ttnn.rms_norm(q_h, weight=w0["q_norm"], epsilon=EPS)
    k_n = ttnn.rms_norm(k_h, weight=w0["k_norm"], epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h)

    capture["q_norm_out"] = _readback_sharded_head(q_n, state.mesh, NQ_PER_CHIP, HEAD_DIM_SLIDING)
    capture["k_norm_out"] = _readback_sharded_head(k_n, state.mesh, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING)
    ttnn.deallocate(q_n); ttnn.deallocate(k_n)

    # ── v0.1.2: V-norm + attention output at pos 0 + o_proj ──
    # Gemma 4 NEW vs 27B/35B: V goes through `v_norm = RMSNorm(head_dim,
    # with_scale=False)` — pure x/rms(x), no learnable weight. This was
    # missing in v0.1.2 first try; cost was rms(mixer_out) 3.7× too high
    # and cos=0.95. Verified with `experiments/cb/isolate/gm4_v012_oproj_sanity.py`
    # numpy reproducer (matched the TT result exactly → bug was in our
    # mental model, not in TT).
    # Reference: `transformers/models/gemma4_unified/modeling_gemma4_unified.py:391-401`.
    #
    # At pos 0 with seq_len=1, softmax(QK^T/sqrt(d)) = 1 (single position
    # attends to itself only with causal mask). RoPE at pos 0 is identity.
    # So attn_out[q] = V_normed[q // GQA_GROUP]. Expand via
    # `ttnn.repeat_interleave(..., dim=0)` — matches HF `repeat_kv` order
    # [kv0, kv0, kv1, kv1, ...].
    v_h = ttnn.reshape(v, [NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING])
    ttnn.deallocate(v)
    # v_norm with `with_scale=False`: ttnn.rms_norm requires a weight, so
    # we pass an all-ones tensor preloaded in bootstrap.
    v_normed = ttnn.rms_norm(v_h, weight=state.ones_head_dim_sliding, epsilon=EPS)
    ttnn.deallocate(v_h)
    attn_per_head = ttnn.repeat_interleave(v_normed, GQA_GROUP_SLIDING, dim=0)  # [NQ_PER_CHIP, head_dim]
    ttnn.deallocate(v_normed)

    # Flatten to [1, NQ_PER_CHIP * HEAD_DIM] for o_proj.
    attn_flat = ttnn.reshape(attn_per_head, [1, NQ_PER_CHIP * HEAD_DIM_SLIDING])
    ttnn.deallocate(attn_per_head)

    # o_proj per chip: [1, NQ_PER_CHIP * HEAD_DIM] @ [NQ_PER_CHIP * HEAD_DIM, HIDDEN]
    # = [1, HIDDEN] partial per chip. All-reduce sum gives the replicated
    # full mixer_out.
    partial = ttnn.matmul(attn_flat, w0["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    mixer_out = all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)

    capture["mixer_out"] = _readback_replicated(mixer_out, state.mesh)
    # KEEP mixer_out alive for the residual_1 step below — do NOT deallocate.

    # ── v0.1.3: post_attention_layernorm → residual → pre_ff_norm →
    #          MLP → post_ff_norm → residual → L0 output ──
    #
    # Gemma 4's "4-norm" decoder layer (plan §1.5):
    #   residual = h
    #   h = input_layernorm(h)
    #   h, _ = self_attn(h)            # this is mixer_out
    #   h = post_attention_layernorm(h)
    #   h = residual + h
    #   residual = h
    #   h = pre_feedforward_layernorm(h)
    #   h = mlp(h)
    #   h = post_feedforward_layernorm(h)
    #   h = residual + h
    #
    # NOTE: post-attn and post-ff norms are applied to the SUB-BLOCK output
    # BEFORE the residual add — they normalize the contribution, not the
    # combined h. Gemma 4 Llama RMSNorm: weight=`w` directly (NO +1.0).

    # We need `residual_1` (= the input to the attention block at L0) for
    # the first residual add. That's `h_scaled` (post embed-scale). But we
    # already deallocated h_scaled — recompute embed → scale to recover it.
    # An alternative would be to keep h_scaled alive across the entire
    # block; that's the standard pattern and saves a recompute. Doing
    # the keep-alive version now.
    # (We will refactor at v0.2 once the multi-layer loop forces a clean
    # in-place residual stream.)
    #
    # IMPLEMENTATION CHOICE: we re-derive `h_scaled` cheaply by running
    # the embed → scale again. For a single token it's tiny.
    tok_tt2 = ttnn.from_torch(
        torch.tensor([[int(tok_id)]], dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    embed2 = ttnn.embedding(tok_tt2, state.embed_tt)
    ttnn.deallocate(tok_tt2)
    residual_1 = ttnn.multiply(ttnn.to_layout(embed2, ttnn.TILE_LAYOUT), EMBED_SCALE)
    ttnn.deallocate(embed2)

    # post_attention_layernorm on mixer_out (the attn output).
    post_attn_norm = ttnn.rms_norm(mixer_out, weight=w0["post_attention_layernorm"], epsilon=EPS)
    ttnn.deallocate(mixer_out)
    capture["post_attn_norm"] = _readback_replicated(post_attn_norm, state.mesh)

    # First residual add.
    h_after_attn = ttnn.add(residual_1, post_attn_norm)
    ttnn.deallocate(residual_1); ttnn.deallocate(post_attn_norm)

    # pre_feedforward_layernorm.
    pre_ff_norm = ttnn.rms_norm(h_after_attn, weight=w0["pre_feedforward_layernorm"], epsilon=EPS)
    capture["pre_ff_norm"] = _readback_replicated(pre_ff_norm, state.mesh)

    # MLP: down(gelu(gate_proj(x)) * up_proj(x)).
    # Per Step 0.2 (commit 4395b28), GELU_tanh matches
    # ttnn.gelu(fast_and_approximate_mode=False). The fused-activation
    # path (ttnn.mul with [UnaryOpType.GELU]) uses the APPROXIMATE kernel
    # — DO NOT mirror 35B's SwiGLU fused pattern. Use two separate ops.
    gate = ttnn.matmul(pre_ff_norm, w0["gate_proj"], compute_kernel_config=HIFI4)
    up   = ttnn.matmul(pre_ff_norm, w0["up_proj"],   compute_kernel_config=HIFI4)
    ttnn.deallocate(pre_ff_norm)
    gelu_gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
    ttnn.deallocate(gate)
    mid = ttnn.mul(gelu_gate, up)
    ttnn.deallocate(gelu_gate); ttnn.deallocate(up)
    # down_proj column-sharded; partial per chip, all_reduce sums.
    mlp_partial = ttnn.matmul(mid, w0["down_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(mid)
    mlp_out = all_reduce_tt(mlp_partial, state.mesh)
    ttnn.deallocate(mlp_partial)
    capture["mlp_out"] = _readback_replicated(mlp_out, state.mesh)

    # post_feedforward_layernorm.
    post_ff_norm = ttnn.rms_norm(mlp_out, weight=w0["post_feedforward_layernorm"], epsilon=EPS)
    ttnn.deallocate(mlp_out)
    capture["post_ff_norm"] = _readback_replicated(post_ff_norm, state.mesh)

    # Second residual add → L0 output. THEN multiply by layer_scalar
    # per Gemma4UnifiedTextDecoderLayer.forward:540 (a learned per-layer
    # scalar buffer). Missing this was the v0.2 L1+ collapse bug — cos
    # invariant under scalar mult masked an 18× magnitude error at L0.
    h_pre = ttnn.add(h_after_attn, post_ff_norm)
    ttnn.deallocate(h_after_attn); ttnn.deallocate(post_ff_norm)
    h_l0 = ttnn.multiply(h_pre, w0["layer_scalar"])
    ttnn.deallocate(h_pre)
    capture["l0_out"] = _readback_replicated(h_l0, state.mesh)
    ttnn.deallocate(h_l0)


def _readback_replicated(t_tt, mesh):
    """Read back a replicated tensor; return as a flat fp32 numpy array."""
    arr = ttnn.to_torch(t_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    if arr.ndim >= 1 and arr.shape[0] == NCHIPS:
        arr = arr[0]
    return arr.float().reshape(-1).numpy()


def _readback_sharded_head(t_tt, mesh, per_chip_heads, head_dim):
    """Read back a tensor sharded over the Q-head (or KV-head) axis.

    The shard layout for q_proj/k_proj/v_proj is "each chip holds
    `per_chip_heads * head_dim` columns of the output". The flat per-chip
    output is [1, per_chip_heads * head_dim] for the projections, and
    [per_chip_heads, head_dim] for the post-RMSNorm tensors. Either way,
    after `ConcatMeshToTensor(dim=0)`, the chip-leading dim is NCHIPS and
    the total head count = NCHIPS * per_chip_heads.

    Returns [NCHIPS * per_chip_heads, head_dim] in HF [NUM_HEADS, HEAD_DIM]
    layout (matches `attn_L0_q_norm` view shape `[B, S, NQ, head_dim]`).
    """
    arr = ttnn.to_torch(t_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    arr = arr.float().reshape(NCHIPS, per_chip_heads, head_dim)
    return arr.reshape(NCHIPS * per_chip_heads, head_dim).numpy()


# ── v0.2: full forward through all 48 layers + final_norm + lm_head ──
#
# step_forward_v02 reuses two per-layer helpers (sliding vs global). At pos 0
# with seq_len=1, softmax(QK^T) = 1 trivially, so attn_out per Q head is just
# V (after v_norm + GQA mapping). The Q/K projections are skipped — they
# would be needed for pos > 0 (v0.3). RoPE at pos 0 is identity (cos=1,
# sin=0); also skipped.

def _layer_pos0_sliding(state, h_norm, w):
    """Sliding attention at pos 0 (validated bit-id to HF at v0.1.2)."""
    v = ttnn.matmul(h_norm, w["v_proj"], compute_kernel_config=HIFI4)
    v_h = ttnn.reshape(v, [NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING])
    ttnn.deallocate(v)
    v_n = ttnn.rms_norm(v_h, weight=state.ones_head_dim_sliding, epsilon=EPS)
    ttnn.deallocate(v_h)
    attn = ttnn.repeat_interleave(v_n, GQA_GROUP_SLIDING, dim=0)
    ttnn.deallocate(v_n)
    attn_flat = ttnn.reshape(attn, [1, NQ_PER_CHIP * HEAD_DIM_SLIDING])
    ttnn.deallocate(attn)
    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    out = all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


def _layer_pos0_global(state, h_norm, w):
    """Full (global) attention at pos 0. NKV=1 (K replicated across chips),
    head_dim=512, p-RoPE, attention_k_eq_v=True (V aliases K_raw pre-norm
    per HF code lines 391-401). At pos 0 with seq=1, attn_out per Q head
    is v_norm(K_raw), same for all heads since there's one KV head.
    """
    k = ttnn.matmul(h_norm, w["k_proj"], compute_kernel_config=HIFI4)
    # V is aliased from K_raw (pre-k_norm, pre-RoPE) per HF.
    v_h = ttnn.reshape(k, [NUM_KV_HEADS_GLOBAL, HEAD_DIM_GLOBAL])
    ttnn.deallocate(k)
    v_n = ttnn.rms_norm(v_h, weight=state.ones_head_dim_global, epsilon=EPS)
    ttnn.deallocate(v_h)
    # Repeat to all Q heads on this chip (each chip handles NQ_PER_CHIP heads).
    attn = ttnn.repeat_interleave(v_n, NQ_PER_CHIP, dim=0)  # [NQ_PER_CHIP, 512]
    ttnn.deallocate(v_n)
    attn_flat = ttnn.reshape(attn, [1, NQ_PER_CHIP * HEAD_DIM_GLOBAL])
    ttnn.deallocate(attn)
    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    out = all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


def _layer_forward_pos0(state, h_in, layer_idx):
    """One full Gemma 4 decoder layer at pos 0. Returns h_out [1, HIDDEN].

    Defensive `ttnn.clone(h_in)` for the residual_1 add: per-layer-cos
    debug showed L0 PASS, L1 hard-FAIL. Hypothesis: rms_norm + downstream
    ops may be aliasing `h_in` between the input_layernorm read and the
    residual add. Clone before rms_norm to break any aliasing.
    """
    w = state.per_layer_tt[layer_idx]
    lt = state.layer_types[layer_idx]
    residual_1 = ttnn.clone(h_in)
    h_norm = ttnn.rms_norm(h_in, weight=w["input_layernorm"], epsilon=EPS)
    if lt == "sliding_attention":
        mixer = _layer_pos0_sliding(state, h_norm, w)
    else:
        mixer = _layer_pos0_global(state, h_norm, w)
    ttnn.deallocate(h_norm)
    post_attn = ttnn.rms_norm(mixer, weight=w["post_attention_layernorm"], epsilon=EPS)
    ttnn.deallocate(mixer)
    h_after_attn = ttnn.add(residual_1, post_attn)
    ttnn.deallocate(residual_1); ttnn.deallocate(post_attn)
    pre_ff = ttnn.rms_norm(h_after_attn, weight=w["pre_feedforward_layernorm"], epsilon=EPS)
    gate = ttnn.matmul(pre_ff, w["gate_proj"], compute_kernel_config=HIFI4)
    up = ttnn.matmul(pre_ff, w["up_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(pre_ff)
    gelu_gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
    ttnn.deallocate(gate)
    mid = ttnn.mul(gelu_gate, up)
    ttnn.deallocate(gelu_gate); ttnn.deallocate(up)
    mlp_partial = ttnn.matmul(mid, w["down_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(mid)
    mlp_out = all_reduce_tt(mlp_partial, state.mesh)
    ttnn.deallocate(mlp_partial)
    post_ff = ttnn.rms_norm(mlp_out, weight=w["post_feedforward_layernorm"], epsilon=EPS)
    ttnn.deallocate(mlp_out)
    h_residual_2 = ttnn.add(h_after_attn, post_ff)
    ttnn.deallocate(h_after_attn); ttnn.deallocate(post_ff)
    # Gemma 4 final per-layer scalar multiplication (HF decoder layer
    # forward:540). Without this, L0 has cos~1 but mad ~18× too big at
    # `layer_scalar=0.054`, and L1+ collapses to near-zero cos.
    h_out = ttnn.multiply(h_residual_2, w["layer_scalar"])
    ttnn.deallocate(h_residual_2)
    return h_out


# ── v0.3.0: paged-SDPA forward at pos 0 (matches v0.2 result via the cache) ──
#
# Forks 35B's `attn_forward_ttnn_sdpa` paged path (server_35b_ttnn.py:842-917)
# and adapts for Gemma 4's:
# - v_norm (with_scale=False) on V after view-to-heads
# - NKV_PER_CHIP_SLIDING=2 (vs 35B's 1) — same shape contract; just a thicker
#   dim 1 in the cache
# - sliding_window_size=1024 kwarg on the SDPA decode call (verified Step 0.1)
# - global path: NKV=1 replicated, head_dim=512, attention_k_eq_v aliases V=K
#
# RoPE at pos 0 is identity (cos[0]=1, sin[0]=0). v0.3.0 SKIPS RoPE for that
# reason; v0.3.1 applies it at pos > 0.


def _shard_for_paged_write(t_2d, state, n_kv_heads, head_dim, mem_cfg):
    """Reshape + pad a [n_kv_heads, head_dim] per-chip tensor for paged_update_cache.

    Output: HEIGHT_SHARDED L1 [1, n_kv_heads, BLOCK_SIZE, head_dim] per chip.
    Matches 35B's `_shard_for_paged_write` helper at server_35b_ttnn.py:845-849
    extended for n_kv_heads > 1.
    """
    t4d = ttnn.reshape(t_2d, [1, n_kv_heads, 1, head_dim])
    t_pad = ttnn.pad(t4d, [[0, 0], [0, 0], [0, state.sdpa_block_size - 1], [0, 0]],
                     value=0.0)
    return ttnn.to_memory_config(t_pad, mem_cfg)


def _layer_pos0_sliding_paged(state, h_norm, w, layer_idx):
    """Sliding-attention forward at pos 0 via paged_update_cache + paged SDPA.

    Matches v0.2's `_layer_pos0_sliding` result (which proved correct: cos=0.99999
    vs HF at v0.1.2) but routes through the real cache pipeline so v0.3.1 can
    extend to pos > 0 by simply advancing cur_pos.
    """
    kc, vc = state.kv_caches_tt[layer_idx]

    # Q/K/V projections (Q sharded over heads, K/V sharded over KV heads).
    q = ttnn.matmul(h_norm, w["q_proj"], compute_kernel_config=HIFI4)
    k = ttnn.matmul(h_norm, w["k_proj"], compute_kernel_config=HIFI4)
    v = ttnn.matmul(h_norm, w["v_proj"], compute_kernel_config=HIFI4)

    q_h = ttnn.reshape(q, [NQ_PER_CHIP, HEAD_DIM_SLIDING])
    ttnn.deallocate(q)
    k_h = ttnn.reshape(k, [NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING])
    ttnn.deallocate(k)
    v_h = ttnn.reshape(v, [NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING])
    ttnn.deallocate(v)

    # q_norm, k_norm (with_scale=True; learnable weights per head).
    q_n = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    k_n = ttnn.rms_norm(k_h, weight=w["k_norm"], epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h)

    # v_norm (with_scale=False; identity-weight rms norm).
    v_n = ttnn.rms_norm(v_h, weight=state.ones_head_dim_sliding, epsilon=EPS)
    ttnn.deallocate(v_h)

    # RoPE at pos 0 is identity (cos=1, sin=0). v0.3.1 applies it for pos > 0.

    # Write K, V to paged cache at cur_pos=0.
    k_sharded = _shard_for_paged_write(k_n, state, NKV_PER_CHIP_SLIDING,
                                        HEAD_DIM_SLIDING, state.paged_write_mem_cfg_sliding)
    v_sharded = _shard_for_paged_write(v_n, state, NKV_PER_CHIP_SLIDING,
                                        HEAD_DIM_SLIDING, state.paged_write_mem_cfg_sliding)
    ttnn.deallocate(k_n); ttnn.deallocate(v_n)
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

    # Paged SDPA decode. Q shape [1, 1, NQ_PER_CHIP, head_dim] per chip.
    # sliding_window_size=1024 on sliding layers (Step 0.1 verified).
    q_for_sdpa = ttnn.reshape(q_n, [1, 1, NQ_PER_CHIP, HEAD_DIM_SLIDING])
    ttnn.deallocate(q_n)
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_for_sdpa, kc, vc,
        cur_pos_tensor=state.cur_pos_buf,
        page_table_tensor=state.page_table_tt,
        scale=1.0 / (HEAD_DIM_SLIDING ** 0.5),
        program_config=state.paged_sdpa_progcfg,
        compute_kernel_config=state.sdpa_compute_kernel_config,
        sliding_window_size=SLIDING_WINDOW,
    )
    ttnn.deallocate(q_for_sdpa)
    attn_flat = ttnn.reshape(attn_out, [1, NQ_PER_CHIP * HEAD_DIM_SLIDING])
    ttnn.deallocate(attn_out)

    # o_proj column-sharded + all_reduce.
    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    out = all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


def _layer_forward_pos0_paged(state, h_in, layer_idx):
    """v0.3.0 layer forward — uses paged SDPA on sliding layers; falls back to
    the v0.2 global helper for global layers (v0.3.0 scope is sliding only;
    v0.3.1 ports global as well)."""
    w = state.per_layer_tt[layer_idx]
    lt = state.layer_types[layer_idx]
    residual_1 = ttnn.clone(h_in)
    h_norm = ttnn.rms_norm(h_in, weight=w["input_layernorm"], epsilon=EPS)
    if lt == "sliding_attention":
        mixer = _layer_pos0_sliding_paged(state, h_norm, w, layer_idx)
    else:
        # v0.3.0 still uses v0.2 global helper; v0.3.1 will port to paged.
        mixer = _layer_pos0_global(state, h_norm, w)
    ttnn.deallocate(h_norm)
    post_attn = ttnn.rms_norm(mixer, weight=w["post_attention_layernorm"], epsilon=EPS)
    ttnn.deallocate(mixer)
    h_after_attn = ttnn.add(residual_1, post_attn)
    ttnn.deallocate(residual_1); ttnn.deallocate(post_attn)
    pre_ff = ttnn.rms_norm(h_after_attn, weight=w["pre_feedforward_layernorm"], epsilon=EPS)
    gate = ttnn.matmul(pre_ff, w["gate_proj"], compute_kernel_config=HIFI4)
    up = ttnn.matmul(pre_ff, w["up_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(pre_ff)
    gelu_gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
    ttnn.deallocate(gate)
    mid = ttnn.mul(gelu_gate, up)
    ttnn.deallocate(gelu_gate); ttnn.deallocate(up)
    mlp_partial = ttnn.matmul(mid, w["down_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(mid)
    mlp_out = all_reduce_tt(mlp_partial, state.mesh)
    ttnn.deallocate(mlp_partial)
    post_ff = ttnn.rms_norm(mlp_out, weight=w["post_feedforward_layernorm"], epsilon=EPS)
    ttnn.deallocate(mlp_out)
    h_residual_2 = ttnn.add(h_after_attn, post_ff)
    ttnn.deallocate(h_after_attn); ttnn.deallocate(post_ff)
    h_out = ttnn.multiply(h_residual_2, w["layer_scalar"])
    ttnn.deallocate(h_residual_2)
    return h_out


def step_forward_v03(state, tok_id, capture=None):
    """v0.3.0 forward — paged SDPA on sliding layers; global layers still
    use v0.2's V-routing shortcut. Goal: argmax should still match HF.
    """
    tok_tt = ttnn.from_torch(
        torch.tensor([[int(tok_id)]], dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    embed = ttnn.embedding(tok_tt, state.embed_tt)
    ttnn.deallocate(tok_tt)
    h = ttnn.multiply(ttnn.to_layout(embed, ttnn.TILE_LAYOUT), EMBED_SCALE)
    ttnn.deallocate(embed)

    for L in range(NUM_LAYERS):
        h_new = _layer_forward_pos0_paged(state, h, L)
        ttnn.deallocate(h)
        h = h_new

    final = ttnn.rms_norm(h, weight=state.final_norm_tt, epsilon=EPS)
    ttnn.deallocate(h)
    if capture is not None:
        capture["final_norm"] = _readback_replicated(final, state.mesh)

    logits_raw = ttnn.matmul(final, state.lm_head_tt, compute_kernel_config=HIFI4)
    ttnn.deallocate(final)
    inv = ttnn.multiply(logits_raw, 1.0 / FINAL_LOGIT_SOFTCAP)
    ttnn.deallocate(logits_raw)
    th = ttnn.tanh(inv)
    ttnn.deallocate(inv)
    logits = ttnn.multiply(th, FINAL_LOGIT_SOFTCAP)
    ttnn.deallocate(th)
    if capture is not None:
        capture["logits"] = _readback_replicated(logits, state.mesh)

    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(logits)
    argmax_tt = ttnn.argmax(logits_rm, dim=-1, keepdim=True, use_multicore=True)
    ttnn.deallocate(logits_rm)
    arr = ttnn.to_torch(argmax_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    ttnn.deallocate(argmax_tt)
    argmax = int(arr.reshape(-1)[0].item())
    if capture is not None:
        capture["argmax"] = argmax
    return argmax


def step_forward_v02(state, tok_id, capture=None):
    """Full 48-layer forward at pos 0 → final_norm → lm_head → softcap → argmax.

    capture (optional dict) fills:
      embed_scaled, final_norm, logits, argmax
    Returns: argmax token id (int).
    """
    tok_tt = ttnn.from_torch(
        torch.tensor([[int(tok_id)]], dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    embed = ttnn.embedding(tok_tt, state.embed_tt)
    ttnn.deallocate(tok_tt)
    h = ttnn.multiply(ttnn.to_layout(embed, ttnn.TILE_LAYOUT), EMBED_SCALE)
    ttnn.deallocate(embed)
    if capture is not None:
        capture["embed_scaled"] = _readback_replicated(h, state.mesh)

    for L in range(NUM_LAYERS):
        h_new = _layer_forward_pos0(state, h, L)
        ttnn.deallocate(h)
        h = h_new
        if capture is not None and capture.get("per_layer", False):
            # Save THIS layer's output for cosine localization.
            capture[f"layer_{L}"] = _readback_replicated(h, state.mesh)

    final = ttnn.rms_norm(h, weight=state.final_norm_tt, epsilon=EPS)
    ttnn.deallocate(h)
    if capture is not None:
        capture["final_norm"] = _readback_replicated(final, state.mesh)

    # lm_head: replicated [HIDDEN, VOCAB]; per-chip matmul gives full logits.
    logits_raw = ttnn.matmul(final, state.lm_head_tt, compute_kernel_config=HIFI4)
    ttnn.deallocate(final)

    # Logit softcap: logits = SOFTCAP · tanh(logits / SOFTCAP). Monotonic so
    # argmax is invariant — but sampling distributions depend on it.
    inv = ttnn.multiply(logits_raw, 1.0 / FINAL_LOGIT_SOFTCAP)
    ttnn.deallocate(logits_raw)
    th = ttnn.tanh(inv)
    ttnn.deallocate(inv)
    logits = ttnn.multiply(th, FINAL_LOGIT_SOFTCAP)
    ttnn.deallocate(th)
    if capture is not None:
        capture["logits"] = _readback_replicated(logits, state.mesh)

    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(logits)
    argmax_tt = ttnn.argmax(logits_rm, dim=-1, keepdim=True, use_multicore=True)
    ttnn.deallocate(logits_rm)
    arr = ttnn.to_torch(argmax_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    ttnn.deallocate(argmax_tt)
    argmax = int(arr.reshape(-1)[0].item())
    if capture is not None:
        capture["argmax"] = argmax
    return argmax


if __name__ == "__main__":
    # Direct-run smoke: bootstrap + L0 in_norm forward on the canonical prompt.
    state = State()
    bootstrap(state)
    # Prompt: "The capital of France is" → ids [2, 818, 5279, 529, 7001, 563]
    # Run v0.1.0 forward at pos 0 (token id 2 = BOS).
    cap = {}
    step_forward_v01(state, tok_id=2, capture=cap)
    print(f"[smoke] embed_scaled[:5] = {cap['embed_scaled'][:5]}")
    print(f"[smoke] in_norm[:5]      = {cap['in_norm'][:5]}")
    print(f"[smoke] embed_scaled rms = {np.sqrt(np.mean(cap['embed_scaled']**2)):.4f}")
    print(f"[smoke] in_norm rms      = {np.sqrt(np.mean(cap['in_norm']**2)):.4f}")
    ttnn.close_device(state.mesh)
