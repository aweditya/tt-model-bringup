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

# Round 7 perf (2026-06-05) — HiFi2 NEGATIVE FINDING (reverted).
# Profile-driven hypothesis: matmuls were 23.58% of measured kernel time
# in the post-Round-6 tt-perf-report; HiFi2 was inlined-suggested by the
# report on every dense matmul site ("HiFi2 may also work, it discards
# the lowest bit of the activations and has 2x the throughput of HiFi4");
# Llama 70B Galaxy production ships HiFi2 for Q/K/V/O. Probe
# `experiments/cb/isolate/gm4_hifi2_matmul_probe.py` confirmed precision
# is acceptable: cos(HiFi4, HiFi2) 0.9999918-0.9999931, max|delta|
# 0.015-0.031 on all 5 representative decoder matmul shapes.
#
# RESULT (n=3 v04 traced validator, qb2): baseline 46.67 ± 0.05 ms/tok →
# HiFi2 46.60 ± 0.0 ms/tok = -0.07 ms (-0.15%, within noise). Eager:
# 183.2 → 183.2 ms/tok (no movement). 3×100/100 PASS.
#
# Diagnosis: Gemma 4 12B decode at B=1 is DRAM-bandwidth bound, not
# math-bound. The matmul reads the full weight tile from DRAM per token
# (no batching to amortise the BW); the lowest-bit activation precision
# bit doesn't change the DRAM-read cost. HiFi2's 2x math throughput is
# wasted when math isn't the bottleneck. The tt-perf-report DRAM% on
# 32x3840x1024 was 29% (vs peak); for larger reduction-axis matmuls it
# would be higher but still BW-pinned at B=1. Llama 70B Galaxy ships
# HiFi2 because at TP=8 the per-chip matmul is smaller and more
# compute-bound — different regime.
#
# Verdict: REVERTED — no perf gain, accepts a (tiny) precision concession
# for zero benefit, masks the real bottleneck. Documented as a Round 7
# negative finding in `research/gemma4_perf_qb2_2026-06-05/log.md`.
# The probe is kept as a future-reference isolation pattern.


# Round 9 ablation gate (2026-06-05). The Round 8 bfp8 win regressed
# long-context needle retrieval. To pin which of MLP-bfp8 or lm_head-bfp8
# is the culprit, the upload paths for both are routed through these
# env-driven dtype helpers. Default: bf16 (the Round-9-reverted baseline).
# Set TT_GM4_MLP_DTYPE=bfp8 and/or TT_GM4_LM_HEAD_DTYPE=bfp8 to re-enable
# the Round-8 shape on a per-piece basis. See `research/gemma4_perf_qb2_2026-06-05/log.md`
# §"Round 9" for the ablation results.
def _resolve_dtype(env_name: str, default=None):
    """Map env var value → ttnn dtype. Recognised values: 'bf16', 'bfp8'.
    Returns `default` (caller's default) when unset; otherwise returns the
    selected dtype. Unknown values raise.
    """
    import os as _os
    v = _os.environ.get(env_name, "").strip().lower()
    if v == "":
        return default if default is not None else ttnn.bfloat16
    if v in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if v in ("bfp8", "bfp8_b", "bfloat8_b"):
        return ttnn.bfloat8_b
    raise ValueError(
        f"{env_name}={v!r} not recognised (expected bf16 or bfp8)")


# ── Upload helpers (reused from 35B per REUSE MANDATE) ─────────────────
def np_to_replicated(arr, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch.from_numpy(arr.astype(np.float32)),
        dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def np_stacked_to_sharded(per_chip_list, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          memory_config=None):
    """Stack a list of NCHIPS numpy arrays as the leading axis; shard along it.

    Uses `ShardTensorToMesh(dim=0)` (1D sharder) — matches 35B's
    `np_stacked_to_sharded` (`server_35b_ttnn.py:96`). The 2D variant
    `ShardTensor2dMesh` keeps the leading 4-dim on each chip's tensor
    which breaks downstream matmul (`a=1 vs b=4` shape mismatch).

    Round 10 (2026-06-06) — add `memory_config` kwarg so callers can land
    the weight as WIDTH_SHARDED DRAM (Round 10 Phase 4 lever). Forwards
    to `ttnn.from_torch(..., memory_config=...)` when provided. Default
    behaviour (memory_config=None) preserves the pre-Round-10 contract:
    the runtime picks the default INTERLEAVED-DRAM memory config.
    """
    stacked = np.stack(per_chip_list, axis=0).astype(np.float32)
    kwargs = dict(
        dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    return ttnn.from_torch(torch.from_numpy(stacked), **kwargs)


# ── Round 10 — DRAM-sharded MLP helpers ─────────────────────────────────
# Forks `experiments/cb/isolate/gm4_dram_sharded_mlp_probe.py:82-156`
# (helpers `_dram_weight_mem_cfg`, `_activation_l1_width_sharded`,
# `_dram_sharded_program_config`). Probe PASSED on all 3 MLP per-chip
# shapes (cos = 0.9999937, marginally MORE accurate vs fp32). This is the
# production wire-up for Round 10 Phase 4. Gated behind env var
# TT_GM4_DRAM_PREFETCH (set to "1" to enable). See landscape map at
# `research/gemma4_perf_qb2_2026-06-05/tiling_sharding_plan.md` §A1.

TILE = 32


def _dram_sharded_enabled():
    """Module-level env-gate cache. Read once at first call to avoid the
    per-forward syscall."""
    import os as _os
    return _os.environ.get("TT_GM4_DRAM_PREFETCH", "").strip() == "1"


def _dram_weight_mem_cfg_mlp(mesh, K, N):
    """WIDTH_SHARDED DRAM memory config for a per-chip [K, N] MLP weight.

    Forks `gm4_dram_sharded_mlp_probe.py:_dram_weight_mem_cfg` (which
    itself forks `tt-metal/models/demos/llama3_70b_galaxy/tt/model_config.py:
    2312-2320 create_dram_sharded_mem_config`).

    For the Gemma 4 per-chip MLP triplet (K=3840, N=3840) on Blackhole
    P150 (`dram_grid_size = (8, 1)`): N already aligned to TILE × num_banks
    (= 32 × 8 = 256), so no padding.
    """
    dram_grid_size = mesh.dram_grid_size()
    num_banks = dram_grid_size.x
    assert dram_grid_size.y == 1, f"dram_grid_size.y must be 1; got {dram_grid_size.y}"
    padded_N = int(math.ceil(N / (TILE * num_banks)) * (TILE * num_banks))
    dram_grid = ttnn.CoreRangeSet({
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
        )
    })
    shard_spec = ttnn.ShardSpec(
        dram_grid,
        (K, padded_N // num_banks),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )


def _activation_l1_width_sharded(mesh, M, K, num_cores=8):
    """L1 WIDTH_SHARDED activation memory config for DRAM-sharded matmul in0.

    Forks `gm4_dram_sharded_mlp_probe.py:_activation_l1_width_sharded`
    + `tt-metal/.../test_matmul_dram_sharded.py:135-142`.
    """
    in0_block_w = K // num_cores // TILE
    in0_shard_grid = ttnn.CoreRangeSet({
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(num_cores - 1, 0),
        )
    })
    in0_shard_spec = ttnn.ShardSpec(
        in0_shard_grid,
        [M, in0_block_w * TILE],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        in0_shard_spec,
    )


def _dram_sharded_mlp_program_config(M, K, N, num_cores=8):
    """Matmul program config for DRAM-sharded MLP weight.

    Forks `gm4_dram_sharded_mlp_probe.py:_dram_sharded_program_config`
    + `tt-metal/.../test_matmul_dram_sharded.py:140-145`.

    Note (probe finding): in0_block_w = K/num_cores/TILE/4 = 15/4 = 3.75
    floored to 3 by max(1, ...). If the kernel rejects 3 we fall back to
    5 or 15 by reading the assertion message.
    """
    in0_block_w_unscaled = K // num_cores // TILE
    in0_block_w = max(1, in0_block_w_unscaled // 4)
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=M // TILE,
        per_core_N=N // num_cores // TILE,
        fused_activation=None,
    )


def _dram_sharded_mlp_program_config_with_gelu(M, K, N, num_cores=8):
    """Same as `_dram_sharded_mlp_program_config` but with fused GELU activation
    (the Round-4 lever, ported into the DRAM-sharded matmul). Used on gate_proj
    only.
    """
    in0_block_w_unscaled = K // num_cores // TILE
    in0_block_w = max(1, in0_block_w_unscaled // 4)
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=M // TILE,
        per_core_N=N // num_cores // TILE,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, False),
    )


def shard_along(arr, axis, n=NCHIPS):
    """Split a numpy array into n equal shards along an axis."""
    return [s.copy() for s in np.split(arr, n, axis=axis)]


def load_t(key_to_shard, key):
    path = key_to_shard[key]
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key).float().numpy()


def build_key_to_shard(variant="base"):
    """Build {key -> shard_path} index for google/gemma-4-12B (or its -it
    variant) safetensors. Variant selected by TT_GEMMA4_VARIANT in
    bootstrap; both variants share the same architecture so the keys
    are identical, only the snapshot directory changes.
    """
    cache_dirname = "models--google--gemma-4-12B-it" if variant == "it" else "models--google--gemma-4-12B"
    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_dirname / "snapshots"
    if not snapshot_root.exists():
        raise FileNotFoundError(f"no HF snapshot at {snapshot_root}. "
                                f"Run hf_reference_gemma4_12b.py once to fetch.")
    # IT variant has multiple snapshot dirs (one with just chat templates,
    # one with weights). Pick the snapshot that actually contains weights.
    # Bare `next(iterdir())` non-deterministically picks one — if it picks
    # the chat-template-only dir, `next(snap.glob("*.safetensors"))` raises
    # StopIteration which silently kills the process inside generator
    # contexts (uvloop _set_state / harness top-level).
    snap = None
    for cand in snapshot_root.iterdir():
        if not cand.is_dir():
            continue
        if (cand / "model.safetensors.index.json").exists() or \
                any(cand.glob("*.safetensors")):
            snap = cand
            break
    if snap is None:
        raise FileNotFoundError(
            f"no snapshot under {snapshot_root} contains safetensors weights "
            f"(only chat templates / configs found)")
    index = snap / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text())
        return {k: str(snap / v) for k, v in idx["weight_map"].items()}
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

    Round 8 / Round 9 (2026-06-05) — bfp8 weights REVERTED. Round 8 shipped
    `dtype=ttnn.bfloat8_b` here for a -1.86% traced perf win (block-fp8 with
    shared exponent per TILE, halves DRAM read per matmul). The 100/100
    short-token validator gate PASSed but the long-context needle-haystack
    diagnostic at L=128/512/1024 collapsed to deterministic template loops
    (0/3 Y @ L=128 / L=512, 1/3 P @ L=1024 — was 3/3 Y pre-Round-8). Round 9
    ablation pins the culprit (see research/gemma4_perf_qb2_2026-06-05/log.md
    "Round 9"); default reverted to bf16 for long-context safety. The
    bfp8 precedent and probe are kept in
    `experiments/cb/isolate/gm4_bfp8_weights_probe.py` for future use.
    """
    w = {}
    gate_w = layer_sd["mlp.gate_proj.weight"].T  # [HIDDEN, INTERMEDIATE]
    up_w   = layer_sd["mlp.up_proj.weight"].T
    down_w = layer_sd["mlp.down_proj.weight"].T  # [INTERMEDIATE, HIDDEN]
    # Round 9 ablation gate: set TT_GM4_MLP_DTYPE=bfp8 to re-enable Round 8.
    mlp_dtype = _resolve_dtype("TT_GM4_MLP_DTYPE", default=ttnn.bfloat16)
    # Round 10 (2026-06-06) — DRAM-sharded MLP weights. When env-gated,
    # upload weights with the default INTERLEAVED DRAM mem_config (so the
    # ShardTensorToMesh(dim=0) sharding + 3D-stacked input contract is
    # respected), then `to_memory_config` to WIDTH_SHARDED DRAM per-chip.
    # The per-chip shape after the leading-axis shard is [HIDDEN, IN_PC] =
    # [3840, 3840] for all three. We pass this per-chip shape to
    # `_dram_weight_mem_cfg_mlp`. Doing the reshard via `to_memory_config`
    # (rather than passing memory_config to from_torch) avoids the rank
    # mismatch between the 3D stacked input and the 2D WIDTH_SHARDED spec.
    w["gate_proj"] = np_stacked_to_sharded(
        shard_along(gate_w, axis=1), mesh, dtype=mlp_dtype)
    w["up_proj"]   = np_stacked_to_sharded(
        shard_along(up_w,   axis=1), mesh, dtype=mlp_dtype)
    w["down_proj"] = np_stacked_to_sharded(
        shard_along(down_w, axis=0), mesh, dtype=mlp_dtype)
    if _dram_sharded_enabled():
        mlp_mem_cfg = _dram_weight_mem_cfg_mlp(
            mesh, K=HIDDEN, N=INTERMEDIATE_PER_CHIP)
        for k in ("gate_proj", "up_proj", "down_proj"):
            old = w[k]
            w[k] = ttnn.to_memory_config(old, mlp_mem_cfg)
            ttnn.deallocate(old)
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
        # v0.4 trace state. tok_buf is the trace-resident token input;
        # cur_pos_buf / rot_idxs_buf are already in the v0.3 set. trace_id
        # is the captured-decode-trace handle; traced_argmax_tt is the
        # on-device argmax tensor produced by the captured forward.
        self.tok_buf = None
        self.trace_id = None
        self.traced_argmax_tt = None
        # ── Phase 2.A spec-dec: last sliding + last full attn layer KV
        #    for drafter cross-attention. Populated lazily on demand by
        #    `read_shared_kv_for_drafter(state, L_kv)`. None until first
        #    call (no perf cost when spec-dec disabled).
        #    Format: dict {"sliding_attention": (K_np, V_np),
        #                  "full_attention":    (K_np, V_np)}
        #    Shapes match HF: sliding K/V (B=1, NKV=8, L_kv, 256),
        #                     full    K/V (B=1, NKV=1, L_kv, 512).
        self.shared_kv_for_drafter = None
        # Cached layer indices — derived from state.layer_types at bootstrap.
        self.last_sliding_idx = None
        self.last_full_idx = None
        # ── Phase 2.B.1 spec-dec: B=K+1 verify trace state. Allocated by
        #    setup_verify_kp1_state(state, K); captured by
        #    capture_verify_trace_kp1(state). All None until spec-dec is
        #    enabled (no perf cost otherwise).
        #
        #    Read-only verify decision (see research/gemma4_verify_kp1_readonly_decision.md):
        #    K+1 candidate Q rows attend to the SAME KV history through cur_pos via
        #    the alias page-table. NO paged_update_cache during verify; cache is
        #    advanced only by the target's canonical B=1 decode step after accept.
        self.verify_K = None              # int; lookahead depth (5 default)
        self.verify_tok_buf = None        # uint32 [1, K+1, 1] — K+1 candidate token IDs
        self.verify_pos_buf = None        # int32  [K+1]      — K+1 positions (all = cur_pos)
        self.verify_rot_idxs_buf = None   # uint32 [K+1]      — K+1 RoPE indices (all = cur_pos)
        self.verify_page_table_tt = None  # int32 [K+1, num_blocks] alias page-table
        self.verify_trace_id = None       # captured trace id
        self.verify_output_tt = None      # captured logits handle [K+1, vocab_size]


# ── Bootstrap ──────────────────────────────────────────────────────────
def bootstrap(state, log=None):
    if log is None:
        log = print

    log("[bootstrap] open mesh + fabric…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    # trace_region_size: the v0.4 decode trace (48 layers × paged SDPA
    # + MLP per layer + lm_head + argmax) needs ~250 MB by analogy with
    # 27B's 0.2 GB decode trace. Use 400 MB to leave headroom for chunked
    # prefill later. 27B uses 800 MB (decode + chunked prefill); 35B is
    # larger still. Default 50 MB triggers TT_THROW.
    state.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS),
                                        trace_region_size=400_000_000)
    log(f"  mesh: {state.mesh}")

    # Variant select: base (default) or instruct (TT_GEMMA4_VARIANT=it).
    # Same architecture / shapes; the IT variant has different weights, a
    # real chat template, and <end_of_turn> as a proper special token.
    import os as _os
    variant = _os.environ.get("TT_GEMMA4_VARIANT", "base").lower()
    if variant not in ("base", "it"):
        raise ValueError(f"TT_GEMMA4_VARIANT must be 'base' or 'it', got {variant!r}")
    if variant == "it":
        hf_model_id = "google/gemma-4-12B-it"
        hf_cache_dirname = "models--google--gemma-4-12B-it"
    else:
        hf_model_id = "google/gemma-4-12B"
        hf_cache_dirname = "models--google--gemma-4-12B"
    state.variant = variant
    state.hf_model_id = hf_model_id

    log(f"[bootstrap] variant={variant} model={hf_model_id} — config + tokenizer…")
    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / hf_cache_dirname / "snapshots"
    snap = next(snapshot_root.iterdir())
    cfg_json = json.loads((snap / "config.json").read_text())
    try:
        from transformers import AutoTokenizer
        state.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        state.tok = state.tokenizer  # cb_api convention
        # The base model doesn't ship a chat template (that's the instruct
        # variant). cb_api always calls apply_chat_template, so install a
        # minimal Gemma-style template — `<start_of_turn>{role}\n{message}<end_of_turn>\n`
        # with `<start_of_turn>model\n` appended when add_generation_prompt=True.
        if variant == "base":
            # The BASE model doesn't ship a chat template; install a minimal
            # Gemma-style one. The IT variant has its own — leave it.
            if not getattr(state.tokenizer, "chat_template", None):
                state.tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "<start_of_turn>{{ message['role'] }}\n"
                    "{{ message['content'] }}<end_of_turn>\n"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
                )
        # For chat, the response should stop on <end_of_turn> (dialog
        # boundary), not the corpus-level <eos>. The IT generation_config
        # lists three EOS candidates [1=<eos>, 106=<end_of_turn>, 50];
        # cb_engine only matches a single eos_token_id, so we point it at
        # <end_of_turn> for BOTH variants. (For BASE: <end_of_turn> is
        # multi-token text, so this still won't stop reliably, but it's a
        # no-op when unmatched. For IT: 106 fires after the model's reply.)
        eot_id = state.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot_id is not None and eot_id != state.tokenizer.unk_token_id:
            state.tokenizer.eos_token = "<end_of_turn>"
        log(f"  tokenizer: {state.tokenizer.__class__.__name__}, "
            f"eos_token={state.tokenizer.eos_token!r} "
            f"id={state.tokenizer.eos_token_id}, "
            f"chat_template_installed={state.tokenizer.chat_template is not None}")
    except Exception as e:
        log(f"  tokenizer load skipped: {e!r}")
    text_cfg_json = cfg_json["text_config"]
    state.text_cfg = text_cfg_json  # dict, not pydantic
    state.layer_types = list(text_cfg_json["layer_types"])
    log(f"  {len(state.layer_types)} layers; "
        f"{sum(1 for t in state.layer_types if t == 'sliding_attention')} sliding / "
        f"{sum(1 for t in state.layer_types if t == 'full_attention')} global")
    # Phase 2.A spec-dec: cache last sliding + last full attn layer indices.
    # Drafter cross-attends to target's LAST-layer KV per layer_type. Per the
    # 12B IT config + HF oracle (.cache/hf_oracle_gemma4_12b_assistant/meta.json),
    # last_sliding_idx = 46, last_full_idx = 47. We derive from layer_types
    # rather than hardcoding so a different variant (e.g. 27B with different
    # last indices) keeps working.
    state.last_sliding_idx = max(
        i for i, t in enumerate(state.layer_types) if t == "sliding_attention")
    state.last_full_idx = max(
        i for i, t in enumerate(state.layer_types) if t == "full_attention")

    log("[bootstrap] enumerate shards + load top-level weights to mesh…")
    key_to_shard = build_key_to_shard(variant=variant)
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

    # Tied lm_head (plan §1.8): lm_head ≡ embed_table.T. Vocab-sharded on
    # dim=-1 across the (1, NCHIPS) mesh — each chip holds [HIDDEN,
    # VOCAB/NCHIPS]. Forward computes per-chip [B, VOCAB/NCHIPS] logits
    # via TP matmul, all_gathers on dim=-1 to replicate full logits,
    # slices to vocab_size, untilizes and on-device argmaxes — small
    # readback. Forks 27B P22 (`server_tp.py:1680-1687`, commit `ef3f336`).
    # Gemma 4 12B's VOCAB=262144 is cleanly divisible by 4 (=65536) and
    # tile-aligned (65536 % 32 = 0). Pre-fix this was REPLICATED, costing
    # ~2 GB/chip and forcing a [1, 262144] readback per token.
    VOCAB = int(embed_w_np.shape[0])
    assert VOCAB % NCHIPS == 0, f"VOCAB {VOCAB} not divisible by NCHIPS {NCHIPS}"
    state.vocab_size = VOCAB  # cb_api / scheduler expect this attr
    # Round 8 / Round 9 (2026-06-05) — lm_head bfp8 REVERTED. Round 8 shipped
    # `dtype=ttnn.bfloat8_b` here (-0.4 ms/tok traced incremental). See the
    # `upload_mlp_layer` docstring above for the long-context regression that
    # forced the revert; Round 9 ablation pinned the culprit. Default is bf16
    # for long-context safety; set TT_GM4_LM_HEAD_DTYPE=bfp8 to re-enable.
    lm_head_dtype = _resolve_dtype("TT_GM4_LM_HEAD_DTYPE", default=ttnn.bfloat16)
    state.lm_head_tt = ttnn.from_torch(
        torch.from_numpy(embed_w_np.T.astype(np.float32)),
        dtype=lm_head_dtype, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
    )
    log(f"  lm_head (tied, vocab-sharded dim=1, "
        f"per-chip {VOCAB // NCHIPS}): {embed_w_np.T.shape}")

    # v0.3: KV cache + paged SDPA infrastructure (forked from 35B
    # `server_35b_ttnn.py:1700-1820` setup). Two head_dim variants —
    # sliding (256) and global (512); allocate caches and pos buffers
    # for each per layer.
    state.MAX_KV = MAX_KV
    state.sdpa_block_size = 32  # tile-height; matches 35B
    # cur_pos_buf: int32 [1] device-resident — kernel asserts INT32 at
    # `paged_update_cache_device_operation.cpp:112`. rot_idxs_buf: uint32 [1]
    # for ttnn.embedding into the cos/sin tables. Both pre-allocated ONCE
    # here and updated in place via copy_host_to_device_tensor in `_set_pos`
    # (27B prod pattern at server_tp.py:1607-1619; recreating per step is
    # the [[ttnn-list-rebinding-leaks]] anti-pattern).
    state.cur_pos_buf = ttnn.from_torch(
        torch.zeros((1,), dtype=torch.int32),
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    state.rot_idxs_buf = ttnn.from_torch(
        torch.zeros((1,), dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    # tok_buf: uint32 [1, 1] for ttnn.embedding(tok_buf, embed_tt) inside
    # the traced forward. Updated out-of-trace via copy_host_to_device_tensor.
    state.tok_buf = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    # Page table: v0.3.0.1 uses NKV=1-per-cache layout (35B's clean
    # contract), so all caches use a single `[1, num_blocks]` page table.
    num_blocks = MAX_KV // state.sdpa_block_size
    pt_identity = np.arange(num_blocks, dtype=np.int32).reshape(1, num_blocks)
    state.page_table_tt = ttnn.from_torch(
        torch.from_numpy(pt_identity),
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    state.num_blocks = num_blocks
    log(f"  paged SDPA: MAX_KV={MAX_KV}, block_size={state.sdpa_block_size}, "
        f"num_blocks={num_blocks}")

    # KV caches per layer. v0.3.0.1: two caches per sliding layer, each
    # with NKV_PER_CHIP=1 effective (mirrors 35B's clean contract). The 8
    # KV heads are split: cache_0 holds even-indexed (KV head 2c on chip c),
    # cache_1 holds odd (KV head 2c+1 on chip c). Then HF GQA mapping
    # (Q head q → KV head q//2) translates to:
    #   - Q heads (4c, 4c+1) attend to cache_0 (KV head 2c)
    #   - Q heads (4c+2, 4c+3) attend to cache_1 (KV head 2c+1)
    # Each SDPA call has NQ_per_call=2, NKV_per_call=1, GQA group=2.
    # Memory: same total as previous single-cache approach (~320 MB/chip).
    state.kv_caches_tt = []
    for L in range(NUM_LAYERS):
        if state.layer_types[L] == "sliding_attention":
            layer_caches = []
            for _ in range(NKV_PER_CHIP_SLIDING):  # 2 caches per sliding layer
                cs = (state.num_blocks, NCHIPS, state.sdpa_block_size, HEAD_DIM_SLIDING)
                init = np.zeros(cs, dtype=np.float32)
                kc = ttnn.from_torch(
                    torch.from_numpy(init), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
                )
                vc = ttnn.from_torch(
                    torch.from_numpy(init), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
                )
                layer_caches.append((kc, vc))
            state.kv_caches_tt.append(layer_caches)
        else:  # full_attention (global) — single cache, NKV=1, replicated
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
            state.kv_caches_tt.append([(kc, vc)])  # list-wrap for uniform indexing
    log(f"  KV caches: {NUM_LAYERS} layers allocated "
        f"(sliding: {NKV_PER_CHIP_SLIDING} caches/layer × {sum(1 for t in state.layer_types if t == 'sliding_attention')} layers)")

    # SDPA program/memory/compute configs (35B-style B3 recipe — HiFi2
    # +fp32_dest_acc_en=False per `[[fp32-sdpa-cliff-probe]]`; HiFi4+fp32
    # has a Blackhole bug at large positions).
    compute_grid = state.mesh.compute_with_storage_grid_size()
    # v0.3.0.1: each cache write is NKV=1 effective (one KV head at a time).
    # 1 core for the [BLOCK_SIZE, head_dim] shard.
    state.paged_write_mem_cfg_sliding = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.num_cores_to_corerangeset(1, compute_grid, row_wise=True),
                       [state.sdpa_block_size, HEAD_DIM_SLIDING],
                       ttnn.ShardOrientation.ROW_MAJOR),
    )
    state.paged_write_mem_cfg_global = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.num_cores_to_corerangeset(NUM_KV_HEADS_GLOBAL, compute_grid, row_wise=True),
                       [state.sdpa_block_size, HEAD_DIM_GLOBAL],
                       ttnn.ShardOrientation.ROW_MAJOR),
    )
    # paged_fused_update_cache requires K and V tensors to be sharded on
    # DISJOINT cores with the same total core count
    # (`paged_fused_update_cache_device_operation.cpp:226`: !is_overlap).
    # The K cfgs above start at core (0,0); build matching V cfgs that
    # offset by the K width into the next column.
    state.paged_write_mem_cfg_sliding_v = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            [state.sdpa_block_size, HEAD_DIM_SLIDING],
            ttnn.ShardOrientation.ROW_MAJOR),
    )
    # Global: K uses cores [0..NKV-1], V uses cores [NKV..2*NKV-1] on the
    # same row (row_wise=True). For NUM_KV_HEADS_GLOBAL=1, V is core (1,0).
    state.paged_write_mem_cfg_global_v = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(
                ttnn.CoreCoord(NUM_KV_HEADS_GLOBAL, 0),
                ttnn.CoreCoord(2 * NUM_KV_HEADS_GLOBAL - 1, 0))]),
            [state.sdpa_block_size, HEAD_DIM_GLOBAL],
            ttnn.ShardOrientation.ROW_MAJOR),
    )
    # Sliding (head_dim=256): CoreCoord(4,4) = 16 cores, same as 35B.
    state.paged_sdpa_progcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
        q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False,
    )
    # Global (head_dim=512): canonical config from tt-metal's in-tree Gemma 4
    # demo at `models/demos/gemma4/tt/attention/decode.py:126-160`. The CB
    # math is `k_tiles = k_chunk_size * DHt * 2` (double-buffered). At
    # head_dim=512, DHt=16 (vs 8 at 256); halving k_chunk_size from the
    # auto-default (128) to 64 restores the per-core CB footprint to within
    # the 1.5 MB L1 budget. CoreCoord(8,4) = 32 cores.
    state.paged_sdpa_progcfg_global = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        q_chunk_size=32, k_chunk_size=64, exp_approx_mode=False,
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
    # Round 5 perf (2026-06-05): pre-bake the rotate_half sign mask into the
    # sin tables so `_apply_full_rope` can replace its `neg + concat` (2 ops)
    # with a single `ttnn.roll` (1 op). The sign mask is [-1]*half + [+1]*half;
    # multiplying it into sin makes `rotated * sin_signed` = `concat([-x2, x1]) * sin`
    # bit-identical (probe: gm4_roll_rope_probe.py max|delta|=0.0).
    half_sliding = HEAD_DIM_SLIDING // 2
    sin_sliding[:, :half_sliding] *= -1.0
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
    # Round 5 perf: same sign-mask bake as sliding above.
    half_global = HEAD_DIM_GLOBAL // 2
    sin_global[:, :half_global] *= -1.0
    state.cos_global_tt = np_to_replicated(cos_global, state.mesh,
                                           layout=ttnn.ROW_MAJOR_LAYOUT)
    state.sin_global_tt = np_to_replicated(sin_global, state.mesh,
                                           layout=ttnn.ROW_MAJOR_LAYOUT)
    log(f"  RoPE tables: sliding [{cos_sliding.shape}] + global [{cos_global.shape}] "
        f"(MAX_KV={MAX_KV}, sin half-sign-baked for round-5 roll fusion)")

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


def _lm_head_argmax(state, final, capture_logits=False):
    """Sharded lm_head + softcap + all_gather + on-device argmax. Forks
    27B P22 (`server_tp.py:1680-1687`, commit `ef3f336`).

    Pipeline (per chip):
      x [B, HIDDEN] @ lm_head[HIDDEN, VOCAB/4] → sharded_logits [B, VOCAB/4]
      softcap: SOFTCAP·tanh(sharded_logits / SOFTCAP) (elementwise, sharded)
      all_gather dim=-1                       → [B, VOCAB] replicated
      slice [B, vocab_size]                   → ROW-MAJOR-friendly
      untilize → argmax(keepdim=True, multicore) → [B, 1] UINT32

    Returns (argmax_tt, full_logits_or_None). `capture_logits=True` keeps
    the full [B, vocab_size] tensor for cosine validators (incurs a
    bf16 readback, ~2 MB at vocab=262144 — fine for debug, off by
    default).
    """
    # Per-chip matmul: weights are sharded on dim 1 (VOCAB axis); result
    # is sharded the same way.
    sharded = ttnn.matmul(final, state.lm_head_tt, compute_kernel_config=HIFI4)
    ttnn.deallocate(final)
    # Softcap on the sharded tensor (cheaper than on the gathered).
    inv = ttnn.multiply(sharded, 1.0 / FINAL_LOGIT_SOFTCAP)
    ttnn.deallocate(sharded)
    th = ttnn.tanh(inv)
    ttnn.deallocate(inv)
    sharded_softcapped = ttnn.multiply(th, FINAL_LOGIT_SOFTCAP)
    ttnn.deallocate(th)
    # All_gather to replicate the full logits on every chip.
    gathered = ttnn.all_gather(sharded_softcapped, dim=-1)
    ttnn.deallocate(sharded_softcapped)
    # Slice to true vocab (in case of any tile padding); for Gemma 4 12B
    # this is a no-op (262144 is tile-aligned). The slice's begins/ends
    # must match the input rank — ttnn matmul output here may be rank-2
    # `[B, VOCAB]` or rank-3 `[1, B, VOCAB]` depending on path. Build
    # the indices from gathered.shape so we work for either.
    vocab_size = getattr(state, "vocab_size", None)
    if vocab_size is None:
        vocab_size = int(gathered.shape[-1])
    gshape = list(gathered.shape)
    begins = [0] * len(gshape)
    ends = list(gshape)
    ends[-1] = vocab_size
    sliced = ttnn.slice(gathered, begins, ends)
    # Keep `gathered` alive: `sliced` is a VIEW. The full-logits readback
    # (capture path) uses it; the argmax path consumes it.
    full_logits = gathered if capture_logits else None
    rm = ttnn.untilize(sliced, use_multicore=True)
    if not capture_logits:
        ttnn.deallocate(gathered)
    # use_multicore=False for determinism (cross-core tie-break race; see
    # research/35b_determinism_2026-06-04.md). Negligible perf cost at vocab=262144/4.
    argmax_tt = ttnn.argmax(rm, dim=-1, keepdim=True, use_multicore=False)
    ttnn.deallocate(rm)
    # Normalize rank to match the 27B contract that cb_scheduler expects:
    # argmax → [B, 1]; full_logits → [B, vocab_size]. Gemma 4's activations
    # carry a seq dim of 1, so without this reshape both are rank-3
    # `[1, B, ...]`, which breaks `idxs[s, 0]` in cb_scheduler._step_sampled_topk.
    if argmax_tt.shape[0] == 1 and len(argmax_tt.shape) == 3:
        argmax_tt = ttnn.reshape(argmax_tt, [argmax_tt.shape[1], argmax_tt.shape[2]])
    if full_logits is not None and len(full_logits.shape) == 3 and full_logits.shape[0] == 1:
        full_logits = ttnn.reshape(full_logits, [full_logits.shape[1], full_logits.shape[2]])
    return argmax_tt, full_logits


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
    # Round 6 perf (2026-06-05): see comment in `_layer_forward_pos0_paged`
    # below for why the defensive `residual_1 = ttnn.clone(h_in)` is now
    # gone (rms_norm is functional, h_in is safe to reuse as residual).
    h_norm = ttnn.rms_norm(h_in, weight=w["input_layernorm"], epsilon=EPS)
    if lt == "sliding_attention":
        mixer = _layer_pos0_sliding(state, h_norm, w)
    else:
        mixer = _layer_pos0_global(state, h_norm, w)
    ttnn.deallocate(h_norm)
    post_attn = ttnn.rms_norm(mixer, weight=w["post_attention_layernorm"], epsilon=EPS)
    ttnn.deallocate(mixer)
    h_after_attn = ttnn.add(h_in, post_attn)
    ttnn.deallocate(post_attn)
    pre_ff = ttnn.rms_norm(h_after_attn, weight=w["pre_feedforward_layernorm"], epsilon=EPS)
    # Round-4 perf (2026-06-05): fuse gelu into gate_proj matmul; see the
    # round-4 note in `_layer_forward_pos0_paged` below for details + probe
    # reference.
    gelu_gate = ttnn.matmul(pre_ff, w["gate_proj"], compute_kernel_config=HIFI4,
                            activation="gelu")
    up = ttnn.matmul(pre_ff, w["up_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(pre_ff)
    mid = ttnn.mul(gelu_gate, up)
    ttnn.deallocate(gelu_gate); ttnn.deallocate(up)
    mlp_partial = ttnn.matmul(mid, w["down_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(mid)
    mlp_out = all_reduce_tt(mlp_partial, state.mesh)
    ttnn.deallocate(mlp_partial)
    post_ff = ttnn.rms_norm(mlp_out, weight=w["post_feedforward_layernorm"], epsilon=EPS)
    ttnn.deallocate(mlp_out)
    # Gemma 4 final per-layer scalar multiplication (HF decoder layer
    # forward:540). Without this, L0 has cos~1 but mad ~18× too big at
    # `layer_scalar=0.054`, and L1+ collapses to near-zero cos.
    # Round 6 perf (2026-06-05): the post-residual scalar multiply is
    # fused into `ttnn.add` via the `activations` parameter (same fusion
    # as in `_layer_forward_pos0_paged` below — see comment there for
    # rationale).
    _layer_scalar_act = [ttnn.UnaryWithParam(
        ttnn.UnaryOpType.MUL_UNARY_SFPU, float(w["layer_scalar"]))]
    h_out = ttnn.add(h_after_attn, post_ff, activations=_layer_scalar_act)
    ttnn.deallocate(h_after_attn); ttnn.deallocate(post_ff)
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


def _apply_full_rope(x, cos_tt, sin_tt, n_heads, head_dim):
    """Apply RoPE (rotate-half) to x [n_heads, head_dim] using cos/sin tables.

    Gemma 4 sliding rotates the full head_dim (rope_type=default).
    Gemma 4 global rotates the first 128 of 512 dims (p-RoPE,
    partial_rotary_factor=0.25). But OUR cos/sin tables already encode
    p-RoPE structure inline: non-rotated dims have `inv_freq=0` so
    `cos=1, sin=0` there — RoPE on those dims is identity. So we can
    safely apply the SAME "rotate full head_dim" math to both layer
    types and the global identity-region just passes through.

    Math: x_rope = x * cos + rotate_half(x) * sin
          where rotate_half([a, b]) = [-b, a], a,b each = head_dim/2.

    At pos 0 cos = 1, sin = 0 → x_rope = x (identity). v0.3.1.0 uses
    this to validate the RoPE plumbing without changing the answer.
    """
    # Round 5 perf (2026-06-05): replace the `slice + slice + neg + concat`
    # chain (2 device ops: neg + concat; slices are views) with a single
    # `ttnn.roll` that swaps the two halves circularly. The negation that
    # belongs to the rotate_half is pre-baked into the sin tables at bootstrap
    # (`sin_sliding[:, :half] *= -1` + same for global) so the math stays
    # equivalent:
    #     rotate_half(x) * sin
    #   = concat([-x2, x1]) * concat([sin_a, sin_a])              (cos1==cos2, sin1==sin2)
    #   = concat([-x2 * sin_a, x1 * sin_a])
    #   = concat([x2, x1]) * concat([-sin_a, sin_a])              (factor neg into sin)
    #   = roll(x, half, dim=-1) * sin_signed
    # Production: 3 device ops (roll + mul + addcmul) vs prior 4 (neg +
    # concat + mul + addcmul). Saves 1 op per RoPE call × 96 calls/forward
    # = 96 ops/forward. Per [[feedback-kernel-vs-dispatch-realization]] the
    # saved ops are real kernel work (neg = UnaryNg, concat = data-movement
    # kernel), so the win should realize ~1:1 in trace.
    #
    # Probe: experiments/cb/isolate/gm4_roll_rope_probe.py — max|delta| = 0.0
    # for both sliding (head_dim=256, n_heads=4) and global (head_dim=512,
    # n_heads=8) — BIT-IDENTICAL on bf16. Bootstrap sign-bake is at
    # `bootstrap(state, log)` ~line 575 / 590 (sliding + global).
    #
    # Round-4 history kept here for reference: the addcmul fusion (4 ops →
    # final mul+add → 1 op) shipped 2026-06-05; this round-5 attacks the
    # leading neg+concat. The cos table is UNCHANGED (no sign mask); only sin.
    half = head_dim // 2
    swapped = ttnn.roll(x, shifts=half, dim=-1)
    x_cos = ttnn.mul(x, cos_tt)
    # Fused: x_rope = x_cos + 1.0 * swapped * sin_tt   (sin pre-signed)
    x_rope = ttnn.addcmul(x_cos, swapped, sin_tt, value=1.0)
    ttnn.deallocate(x_cos); ttnn.deallocate(swapped)
    return x_rope


def _lookup_rope(state, cos_table_tt, sin_table_tt, head_dim):
    """Index the precomputed cos/sin tables by state.rot_idxs_buf (current
    position). Returns (cos_tt, sin_tt) shape [1, head_dim] TILE_LAYOUT,
    ready to broadcast over n_heads in _apply_full_rope.
    """
    cos_row = ttnn.embedding(state.rot_idxs_buf, cos_table_tt)  # [1, 1, head_dim]
    sin_row = ttnn.embedding(state.rot_idxs_buf, sin_table_tt)
    cos_tt = ttnn.to_layout(cos_row, ttnn.TILE_LAYOUT)
    sin_tt = ttnn.to_layout(sin_row, ttnn.TILE_LAYOUT)
    ttnn.deallocate(cos_row); ttnn.deallocate(sin_row)
    cos_tt = ttnn.reshape(cos_tt, [1, head_dim])
    sin_tt = ttnn.reshape(sin_tt, [1, head_dim])
    return cos_tt, sin_tt


def _compute_rope_for_forward(state):
    """Round 3 perf (2026-06-05): hoist `_lookup_rope` out of the per-layer
    hot path. The cos/sin rows are identical across ALL 48 layers within one
    forward (same `state.rot_idxs_buf`); recomputing them per layer is pure
    waste. Returns a 4-tuple ``(cos_sliding, sin_sliding, cos_global,
    sin_global)`` consumed by the attention sub-layers. Caller MUST deallocate
    via ``_release_rope_for_forward`` at end of forward.

    Per-forward savings: 47 redundant sliding lookups + 7 redundant global =
    54 hoisted lookups × 4 ops (embedding+to_layout+reshape per cos/sin) ≈
    216 dispatches/forward + 108 fewer Tilize device-ops (~60 μs each per
    round-1 Tracy v2 → ~6.5 ms/forward kernel time). Per
    [[feedback-kernel-vs-dispatch-realization]] this is a kernel-time win
    expected to realize ~1:1 in trace.

    Forks: NONE — caching identical results across a loop is a generic
    optimization. The pattern is the same one used in
    `models/demos/llama3_70b_galaxy/tt/llama_attention.py` where rot_mats
    are constructed once per decode step in the model wrapper, not per layer.
    """
    cos_s, sin_s = _lookup_rope(state, state.cos_sliding_tt,
                                state.sin_sliding_tt, HEAD_DIM_SLIDING)
    cos_g, sin_g = _lookup_rope(state, state.cos_global_tt,
                                state.sin_global_tt, HEAD_DIM_GLOBAL)
    return (cos_s, sin_s, cos_g, sin_g)


def _release_rope_for_forward(rope_cache):
    """Tear down the 4 cached cos/sin tensors from `_compute_rope_for_forward`.
    Call at end of forward; the tensors are tile-layout L1 ([[ttnn-list-
    rebinding-leaks]] anti-pattern requires explicit dealloc, not GC).
    """
    for t in rope_cache:
        ttnn.deallocate(t)


def _shard_for_paged_write(t_2d, state, n_kv_heads, head_dim, mem_cfg, dbg=False):
    """Reshard a [n_kv_heads, head_dim] TILE-layout tensor for paged_update_cache.

    Output: HEIGHT_SHARDED L1 [1, n_kv_heads, BLOCK_SIZE, head_dim] per chip.

    Round 2 simplification (2026-06-05): the prior 5-op chain
    (to_layout(RM) → reshape → pad → to_layout(TILE) → to_memory_config)
    was pure overhead. The post-RoPE input is already TILE-layout and
    tile-padded along seq to TILE_HEIGHT=32 = sdpa_block_size; its byte
    layout matches the L1-sharded [1, n_kv_heads, BLOCK_SIZE, head_dim]
    target verbatim (tile-pad rows beyond logical n_kv_heads carry zeros
    either way; the fused-update kernel writes only row 0 per
    paged_tiled_fused_update_cache_program_factory.cpp:61-64). So one
    metadata reshape + one to_memory_config reshard suffices.

    Validated bit-equivalent to the old chain at
    experiments/cb/isolate/gm4_shard_for_paged_write_v2.py (probe_shard_v2.log,
    K and V cross-variant max|delta| = 0.0). Kernel-time win comes from
    eliminating ~352 dispatches/forward (4 ops × 176 calls) of which
    Untilize + Tilize round-trips dominate per round-1 Tracy v2 capture
    (Tilize 19.3 ms + TilizeWithValPadding 8.5 ms = 28 ms / forward, >50%
    of traced budget at 51 ms).

    Forks `models/demos/llama3_70b_galaxy/tt/llama_attention.py:509-514`
    where K, V also enter paged_fused_update_cache as already-tile-layout
    L1-sharded tensors fresh out of `rotary_embedding_llama_fused_qk` —
    no intermediate untile/repad/retile.
    """
    if dbg: print(f"  [shard dbg] t_2d shape={list(t_2d.shape)} "
                  f"padded={list(t_2d.padded_shape)}", flush=True)
    # Logical [n_kv_heads, head_dim] → [1, n_kv_heads, 1, head_dim]. Volumes
    # match (n_kv_heads*head_dim = 1*n_kv_heads*1*head_dim) so ttnn.reshape
    # accepts. The underlying TILE-padded buffer (32-row pad) is untouched.
    t_4d = ttnn.reshape(t_2d, [1, n_kv_heads, 1, head_dim])
    if dbg: print(f"  [shard dbg] t_4d shape={list(t_4d.shape)} "
                  f"padded={list(t_4d.padded_shape)}", flush=True)
    out = ttnn.to_memory_config(t_4d, mem_cfg)
    if dbg: print(f"  [shard dbg] out shape={list(out.shape)} "
                  f"padded={list(out.padded_shape)}", flush=True)
    ttnn.deallocate(t_4d)
    return out


def _layer_pos0_sliding_paged(state, h_norm, w, layer_idx, capture=None, rope=None):
    """v0.3.0.1 sliding-attention via TWO paged SDPA calls (one per KV head).

    Each call uses NKV_PER_CHIP=1 effective — matches 35B's clean contract.
    HF GQA mapping: Q heads (4c, 4c+1) → KV head 2c (cache_0); Q heads
    (4c+2, 4c+3) → KV head 2c+1 (cache_1). Output [1, 1, 4, head_dim]
    assembled via concat along the Q-head axis.

    capture (optional dict): if provided, captures L0-style sub-op
    readbacks (q_proj_out, k_proj_out, v_proj_out, q_norm_out, k_norm_out,
    v_norm_out, q_rope_out, k_rope_out, mixer_out). Used by
    gm4_v031_L0_subops_pos1.py to bisect sub-ops at pos 0 vs pos 1.

    rope (optional tuple): ``(cos_sliding, sin_sliding)`` from
    ``_compute_rope_for_forward`` — when provided, skip the per-layer
    embedding+tile lookup (round-3 perf). Pre-existing callers (probes /
    sub-op captures) can still call without `rope` and get the legacy
    per-layer path.
    """
    layer_caches = state.kv_caches_tt[layer_idx]  # list of 2 (kc, vc) tuples

    # Q/K/V projections.
    q = ttnn.matmul(h_norm, w["q_proj"], compute_kernel_config=HIFI4)
    k = ttnn.matmul(h_norm, w["k_proj"], compute_kernel_config=HIFI4)
    v = ttnn.matmul(h_norm, w["v_proj"], compute_kernel_config=HIFI4)

    if capture is not None:
        capture["q_proj_out"] = _readback_sharded_head(q, state.mesh, NQ_PER_CHIP, HEAD_DIM_SLIDING)
        capture["k_proj_out"] = _readback_sharded_head(k, state.mesh, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING)
        capture["v_proj_out"] = _readback_sharded_head(v, state.mesh, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING)

    q_h = ttnn.reshape(q, [NQ_PER_CHIP, HEAD_DIM_SLIDING])  # [4, 256] per chip
    ttnn.deallocate(q)
    k_h = ttnn.reshape(k, [NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING])  # [2, 256]
    ttnn.deallocate(k)
    v_h = ttnn.reshape(v, [NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING])  # [2, 256]
    ttnn.deallocate(v)

    # q_norm, k_norm (learned weight) + v_norm (all-ones).
    q_n_pre = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    k_n_pre = ttnn.rms_norm(k_h, weight=w["k_norm"], epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h)
    v_n = ttnn.rms_norm(v_h, weight=state.ones_head_dim_sliding, epsilon=EPS)
    ttnn.deallocate(v_h)

    if capture is not None:
        capture["q_norm_out"] = _readback_sharded_head(q_n_pre, state.mesh, NQ_PER_CHIP, HEAD_DIM_SLIDING)
        capture["k_norm_out"] = _readback_sharded_head(k_n_pre, state.mesh, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING)
        capture["v_norm_out"] = _readback_sharded_head(v_n, state.mesh, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING)

    # RoPE on Q and K (NOT V). v0.3.1.0: at pos 0 cos=1, sin=0, so RoPE
    # is identity — this validates the plumbing without changing the
    # answer. v0.3.1.1 will advance rot_idxs_buf per step for non-trivial
    # rotation at pos > 0.
    # Round-3 perf: prefer the forward-scoped rope cache when available;
    # the per-layer lookup is wasteful (same cos/sin across all 48 layers
    # within a single forward — see _compute_rope_for_forward).
    if rope is not None:
        cos_tt, sin_tt = rope
        owned_rope = False
    else:
        cos_tt, sin_tt = _lookup_rope(state, state.cos_sliding_tt,
                                      state.sin_sliding_tt, HEAD_DIM_SLIDING)
        owned_rope = True
    q_n = _apply_full_rope(q_n_pre, cos_tt, sin_tt, NQ_PER_CHIP, HEAD_DIM_SLIDING)
    ttnn.deallocate(q_n_pre)
    k_n = _apply_full_rope(k_n_pre, cos_tt, sin_tt, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING)
    ttnn.deallocate(k_n_pre)
    if owned_rope:
        ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

    # Two SDPA passes — one per (cache, KV-head, Q-half) trio.
    attn_outs = []
    Q_HALF = NQ_PER_CHIP // NKV_PER_CHIP_SLIDING  # 2 Q heads per KV head
    for kv_idx in range(NKV_PER_CHIP_SLIDING):
        kc, vc = layer_caches[kv_idx]

        # Slice K, V to this KV head: row kv_idx → [1, head_dim]. These
        # are VIEWS of k_n / v_n: _shard_for_paged_write materialises
        # independent storage via reshape→pad→to_layout, so k_sharded is
        # safe — but the k_i / v_i view handles must NOT be deallocated
        # ([[ttnn-slice-view-decay]]). k_n / v_n stay alive for the loop.
        k_i = ttnn.slice(k_n, [kv_idx, 0], [kv_idx + 1, HEAD_DIM_SLIDING])
        v_i = ttnn.slice(v_n, [kv_idx, 0], [kv_idx + 1, HEAD_DIM_SLIDING])
        k_sharded = _shard_for_paged_write(k_i, state, 1, HEAD_DIM_SLIDING,
                                            state.paged_write_mem_cfg_sliding)
        # V on disjoint core so the fused-update kernel can dispatch both in
        # parallel (see paged_write_mem_cfg_sliding_v above).
        v_sharded = _shard_for_paged_write(v_i, state, 1, HEAD_DIM_SLIDING,
                                            state.paged_write_mem_cfg_sliding_v)
        # Fused K+V cache update: one device dispatch instead of two.
        # Forks tt-metal `models/demos/llama3_70b_galaxy/tt/llama_attention.py:509-511`.
        # tt-metal #44946 / multi-chip-opt menu item #11. K, V have identical
        # shapes/layouts here so the kernel can write both in lockstep.
        ttnn.experimental.paged_fused_update_cache(
            kc, k_sharded, vc, v_sharded,
            update_idxs_tensor=state.cur_pos_buf,
            page_table=state.page_table_tt,
        )
        ttnn.deallocate(k_sharded); ttnn.deallocate(v_sharded)

        # Slice Q to the Q-half for this KV head. slice+reshape return
        # VIEWS into q_n (see [[ttnn-slice-view-decay]]): do NOT deallocate
        # either handle — q_n stays alive across the whole loop, the SDPA
        # reads through these views, and the views auto-die at scope exit.
        q_half = ttnn.slice(q_n,
                            [kv_idx * Q_HALF, 0],
                            [(kv_idx + 1) * Q_HALF, HEAD_DIM_SLIDING])
        q_for_sdpa = ttnn.reshape(q_half, [1, 1, Q_HALF, HEAD_DIM_SLIDING])

        attn_i = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_for_sdpa, kc, vc,
            cur_pos_tensor=state.cur_pos_buf,
            page_table_tensor=state.page_table_tt,
            scale=1.0,  # Gemma 4 text attention sets self.scaling=1.0 (modeling_gemma4.py:1178); does NOT use 1/sqrt(d_k). Tenstorrent demo confirms (decode.py:144). Bug was masked at pos 0 (single-token softmax = 1.0 regardless of scale).
            program_config=state.paged_sdpa_progcfg,
            compute_kernel_config=state.sdpa_compute_kernel_config,
            sliding_window_size=SLIDING_WINDOW,
        )
        attn_outs.append(attn_i)
    ttnn.deallocate(q_n); ttnn.deallocate(k_n); ttnn.deallocate(v_n)

    # Concat the two halves along Q-head axis (dim 2). attn_i shape
    # [1, 1, Q_HALF, head_dim] → concat → [1, 1, NQ_PER_CHIP, head_dim].
    attn_concat = ttnn.concat(attn_outs, dim=2)
    for a in attn_outs:
        ttnn.deallocate(a)
    attn_flat = ttnn.reshape(attn_concat, [1, NQ_PER_CHIP * HEAD_DIM_SLIDING])
    ttnn.deallocate(attn_concat)

    # o_proj column-sharded + all_reduce.
    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    out = all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


def _layer_pos0_global_paged(state, h_norm, w, layer_idx, rope=None):
    """v0.3.1 global-attention via paged SDPA. NKV=1 (replicated across
    chips), head_dim=512, p-RoPE (rotate first 128 of 512 inline via the
    global cos/sin tables), attention_k_eq_v=True (V aliases K post-norm).
    Single SDPA call (no GQA split — NKV=1 already matches kernel contract).

    rope (optional tuple): ``(cos_global, sin_global)`` from
    ``_compute_rope_for_forward`` — when provided, skip the per-layer
    embedding+tile lookup (round-3 perf).
    """
    layer_caches = state.kv_caches_tt[layer_idx]  # [(kc, vc)] — single-entry list
    kc, vc = layer_caches[0]

    # Q/K projections. V is aliased from K_raw (pre-norm, pre-RoPE) per HF.
    # attention_k_eq_v=True for global means v_proj is None in the weights.
    q = ttnn.matmul(h_norm, w["q_proj"], compute_kernel_config=HIFI4)
    k = ttnn.matmul(h_norm, w["k_proj"], compute_kernel_config=HIFI4)

    q_h = ttnn.reshape(q, [NQ_PER_CHIP, HEAD_DIM_GLOBAL])
    ttnn.deallocate(q)
    k_h = ttnn.reshape(k, [NUM_KV_HEADS_GLOBAL, HEAD_DIM_GLOBAL])  # [1, 512]
    ttnn.deallocate(k)
    # V aliases K_raw (pre-norm). Clone before any K-normalization op.
    v_raw = ttnn.clone(k_h)

    # q_norm, k_norm. v_norm applied to V = K_raw (HF aliases V from K_raw
    # then v_norm normalizes it).
    q_n_pre = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    k_n_pre = ttnn.rms_norm(k_h, weight=w["k_norm"], epsilon=EPS)
    v_n = ttnn.rms_norm(v_raw, weight=state.ones_head_dim_global, epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h); ttnn.deallocate(v_raw)

    # p-RoPE applied to Q and K (NOT V). The global cos/sin tables encode
    # the partial-RoPE structure inline (last 384 dims have inv_freq=0 so
    # cos=1, sin=0 acts as identity).
    # Round-3 perf: per-forward rope cache (see _compute_rope_for_forward).
    if rope is not None:
        cos_tt, sin_tt = rope
        owned_rope = False
    else:
        cos_tt, sin_tt = _lookup_rope(state, state.cos_global_tt,
                                      state.sin_global_tt, HEAD_DIM_GLOBAL)
        owned_rope = True
    q_n = _apply_full_rope(q_n_pre, cos_tt, sin_tt, NQ_PER_CHIP, HEAD_DIM_GLOBAL)
    ttnn.deallocate(q_n_pre)
    k_n = _apply_full_rope(k_n_pre, cos_tt, sin_tt, NUM_KV_HEADS_GLOBAL, HEAD_DIM_GLOBAL)
    ttnn.deallocate(k_n_pre)
    if owned_rope:
        ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

    # Write K_rope, V to cache (NKV=1). Fused K+V dispatch — same op as
    # the sliding path above. Forks tt-metal `llama_attention.py:509-511`.
    # V is sharded to disjoint cores from K so the fused-update kernel
    # parallelism contract holds (see paged_write_mem_cfg_global_v).
    k_sharded = _shard_for_paged_write(k_n, state, NUM_KV_HEADS_GLOBAL,
                                        HEAD_DIM_GLOBAL, state.paged_write_mem_cfg_global)
    v_sharded = _shard_for_paged_write(v_n, state, NUM_KV_HEADS_GLOBAL,
                                        HEAD_DIM_GLOBAL, state.paged_write_mem_cfg_global_v)
    ttnn.deallocate(k_n); ttnn.deallocate(v_n)
    ttnn.experimental.paged_fused_update_cache(
        kc, k_sharded, vc, v_sharded,
        update_idxs_tensor=state.cur_pos_buf,
        page_table=state.page_table_tt,
    )
    ttnn.deallocate(k_sharded); ttnn.deallocate(v_sharded)

    # Single SDPA call — NKV=1 = clean kernel contract. reshape returns
    # a VIEW of q_n ([[ttnn-slice-view-decay]]); keep q_n alive until SDPA
    # reads it, and don't dealloc the view handle.
    q_for_sdpa = ttnn.reshape(q_n, [1, 1, NQ_PER_CHIP, HEAD_DIM_GLOBAL])
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_for_sdpa, kc, vc,
        cur_pos_tensor=state.cur_pos_buf,
        page_table_tensor=state.page_table_tt,
        scale=1.0,  # Gemma 4: self.scaling=1.0 (see sliding SDPA above).
        program_config=state.paged_sdpa_progcfg_global,
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )
    ttnn.deallocate(q_n)
    attn_flat = ttnn.reshape(attn_out, [1, NQ_PER_CHIP * HEAD_DIM_GLOBAL])

    # o_proj column-sharded + all_reduce.
    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    out = all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


# ── Phase 2.B.1 — B=K+1 verify variants of the paged-attention layers ─
#
# Read-only verify: K+1 candidate Q rows attend to the SAME KV history
# via the alias page-table (state.verify_page_table_tt). We do NOT call
# `paged_fused_update_cache` here — cache is written only by the canonical
# B=1 decode step after the accept walk picks the longest matching prefix.
# Consequence: K/V projections + K/V norms + K-RoPE + K/V shard are all
# SKIPPED (the K/V we would compute would be discarded). Compute is Q only.
#
# Shape contract:
#   Input  h_norm_kp1  shape [K+1, HIDDEN_PER_CHIP]            (3D after reshape)
#   Output             shape [K+1, HIDDEN_PER_CHIP]
#   SDPA Q             shape [1, K+1, Q_HALF, HEAD_DIM]        (per kernel gate c3124d2)
#   SDPA cur_pos       shape [K+1]                             (state.verify_pos_buf)
#   SDPA page_table    shape [K+1, num_blocks]                 (alias, rows 0..K → row 0's blocks)
#
# RoPE: cos/sin lookup of shape [1, HEAD_DIM] broadcasts cleanly across
# [K+1, n_heads, HEAD_DIM] via ttnn.mul + ttnn.addcmul. Both verify rows
# share the same current_pos so a single lookup suffices.


def _layer_pos0_sliding_paged_kp1(state, h_norm_kp1, w, layer_idx, rope=None):
    """B=K+1 read-only verify variant of `_layer_pos0_sliding_paged`.

    Differs from the B=1 path:
    - h_norm_kp1: [K+1, HIDDEN_PER_CHIP]
    - SKIP paged_fused_update_cache (read-only — cache write is owned by
      the canonical B=1 decode step that runs after accept-walk)
    - SKIP k_proj/v_proj/k_norm/v_norm/k_RoPE (their outputs would only
      feed the skipped cache update)
    - SDPA reads state.verify_pos_buf + state.verify_page_table_tt
    - Returns [K+1, HIDDEN_PER_CHIP]
    """
    assert state.verify_K is not None, \
        "call setup_verify_kp1_state(state, K) before the verify forward"
    Bv = state.verify_K + 1
    layer_caches = state.kv_caches_tt[layer_idx]

    # Q projection only. K/V are read from the cache (built by prior decode
    # steps); the canonical B=1 decode is the sole writer of the cache.
    q = ttnn.matmul(h_norm_kp1, w["q_proj"], compute_kernel_config=HIFI4)
    q_h = ttnn.reshape(q, [Bv, NQ_PER_CHIP, HEAD_DIM_SLIDING])
    ttnn.deallocate(q)

    # q_norm (RMS over last dim). Broadcasts across leading [Bv, n_heads].
    q_n_pre = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    ttnn.deallocate(q_h)

    # RoPE on Q. cos_tt/sin_tt shape [1, head_dim] broadcasts over
    # [Bv, n_heads, head_dim] via the mul/addcmul ops in _apply_full_rope.
    # All Bv candidates share current_pos, so a single lookup is sufficient.
    if rope is not None:
        cos_tt, sin_tt = rope
        owned_rope = False
    else:
        cos_tt, sin_tt = _lookup_rope(state, state.cos_sliding_tt,
                                      state.sin_sliding_tt, HEAD_DIM_SLIDING)
        owned_rope = True
    q_n = _apply_full_rope(q_n_pre, cos_tt, sin_tt, NQ_PER_CHIP, HEAD_DIM_SLIDING)
    ttnn.deallocate(q_n_pre)
    if owned_rope:
        ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

    # Two SDPA passes — one per KV head (same GQA split as B=1 path).
    # All Bv candidate rows read the same KV cache slots through the
    # alias page-table.
    attn_outs = []
    Q_HALF = NQ_PER_CHIP // NKV_PER_CHIP_SLIDING  # 2 Q heads per KV head
    for kv_idx in range(NKV_PER_CHIP_SLIDING):
        kc, vc = layer_caches[kv_idx]
        # Slice Q across n_heads dim, keep all Bv rows. 3D slice:
        # q_n [Bv, NQ_PER_CHIP, HEAD_DIM] → [Bv, Q_HALF, HEAD_DIM].
        q_half = ttnn.slice(
            q_n,
            [0, kv_idx * Q_HALF, 0],
            [Bv, (kv_idx + 1) * Q_HALF, HEAD_DIM_SLIDING],
        )
        q_for_sdpa = ttnn.reshape(q_half, [1, Bv, Q_HALF, HEAD_DIM_SLIDING])
        attn_i = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_for_sdpa, kc, vc,
            cur_pos_tensor=state.verify_pos_buf,
            page_table_tensor=state.verify_page_table_tt,
            scale=1.0,  # Gemma 4: self.scaling=1.0 (see B=1 sliding above).
            program_config=state.paged_sdpa_progcfg,
            compute_kernel_config=state.sdpa_compute_kernel_config,
            sliding_window_size=SLIDING_WINDOW,
        )
        attn_outs.append(attn_i)
    ttnn.deallocate(q_n)

    # Concat along Q-head axis (dim 2 of [1, Bv, Q_HALF, HEAD_DIM]) →
    # [1, Bv, NQ_PER_CHIP, HEAD_DIM]. Then flatten to [Bv, NQ*HEAD_DIM]
    # for o_proj.
    attn_concat = ttnn.concat(attn_outs, dim=2)
    for a in attn_outs:
        ttnn.deallocate(a)
    attn_flat = ttnn.reshape(attn_concat, [Bv, NQ_PER_CHIP * HEAD_DIM_SLIDING])
    ttnn.deallocate(attn_concat)

    # o_proj column-sharded + all_reduce. Bv is just a batch dim from the
    # matmul's perspective — TP TPxTP semantics unchanged.
    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    out = all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


def _layer_pos0_global_paged_kp1(state, h_norm_kp1, w, layer_idx, rope=None):
    """B=K+1 read-only verify variant of `_layer_pos0_global_paged`.

    Same read-only contract as sliding kp1 fork:
    - h_norm_kp1: [K+1, HIDDEN_PER_CHIP]
    - SKIP paged_fused_update_cache (cache writes owned by B=1 decode)
    - SKIP k_proj/v_proj/k_norm/v_norm/k_RoPE
    - Returns [K+1, HIDDEN_PER_CHIP]

    Global-specific:
    - SINGLE SDPA call (NKV=1 → no GQA split)
    - head_dim = HEAD_DIM_GLOBAL=512 (vs 256 for sliding)
    - p-RoPE handled inline by cos/sin global tables (identity on last 384 dims)
    - No `sliding_window_size` kwarg
    """
    assert state.verify_K is not None, \
        "call setup_verify_kp1_state(state, K) before the verify forward"
    Bv = state.verify_K + 1
    layer_caches = state.kv_caches_tt[layer_idx]
    kc, vc = layer_caches[0]

    # Q projection only — K/V come from cache.
    q = ttnn.matmul(h_norm_kp1, w["q_proj"], compute_kernel_config=HIFI4)
    q_h = ttnn.reshape(q, [Bv, NQ_PER_CHIP, HEAD_DIM_GLOBAL])
    ttnn.deallocate(q)

    q_n_pre = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    ttnn.deallocate(q_h)

    if rope is not None:
        cos_tt, sin_tt = rope
        owned_rope = False
    else:
        cos_tt, sin_tt = _lookup_rope(state, state.cos_global_tt,
                                      state.sin_global_tt, HEAD_DIM_GLOBAL)
        owned_rope = True
    q_n = _apply_full_rope(q_n_pre, cos_tt, sin_tt, NQ_PER_CHIP, HEAD_DIM_GLOBAL)
    ttnn.deallocate(q_n_pre)
    if owned_rope:
        ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

    # Single SDPA call — NKV=1 matches kernel contract cleanly.
    q_for_sdpa = ttnn.reshape(q_n, [1, Bv, NQ_PER_CHIP, HEAD_DIM_GLOBAL])
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_for_sdpa, kc, vc,
        cur_pos_tensor=state.verify_pos_buf,
        page_table_tensor=state.verify_page_table_tt,
        scale=1.0,  # Gemma 4: self.scaling=1.0
        program_config=state.paged_sdpa_progcfg_global,
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )
    ttnn.deallocate(q_n)
    attn_flat = ttnn.reshape(attn_out, [Bv, NQ_PER_CHIP * HEAD_DIM_GLOBAL])

    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    out = all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


def _layer_forward_pos0_paged(state, h_in, layer_idx, rope_cache=None):
    """v0.3.1 layer forward — uses paged SDPA on BOTH sliding and global
    layers. Sliding has 2 caches (per-KV-head, NKV=1 each); global has 1
    cache (NKV=1 replicated).

    rope_cache (optional 4-tuple from `_compute_rope_for_forward`):
    ``(cos_sliding, sin_sliding, cos_global, sin_global)`` — when present,
    the per-layer ``_lookup_rope`` call is skipped (round-3 perf).
    """
    w = state.per_layer_tt[layer_idx]
    lt = state.layer_types[layer_idx]
    # Round 6 perf (2026-06-05): drop the defensive `residual_1 = ttnn.clone(h_in)`.
    # `ttnn.rms_norm` returns a NEW tensor (`h_norm`) without mutating its
    # input (verified by static analysis of
    # tt-metal `ttnn/cpp/ttnn/operations/normalization/rmsnorm/` — the kernel
    # writes to a fresh output buffer). Subsequent ops in the layer body
    # (mixer, post_attn) never alias h_in either. So `h_in` itself is safe
    # to use as the residual operand in the trailing add. Saves 1 op per
    # layer × 48 layers = 48 clones/forward (~0.3-0.5 ms expected).
    # The clone was added during v0.1 bringup ("L0 PASS, L1 hard-FAIL")
    # under a now-disproved aliasing hypothesis; the real L0/L1 bug at
    # the time was elsewhere (q_norm zero-centered offset, see
    # [[feedback-qwen36-qnorm-knorm-zero-centered]]).
    h_norm = ttnn.rms_norm(h_in, weight=w["input_layernorm"], epsilon=EPS)
    # DEBUG: GM4_SKIP_SLIDING=1 / GM4_SKIP_GLOBAL=1 short-circuit one
    # attention type to a zero-mixer (residual passes through). Used to
    # bisect which layer-type contributes the pos > 0 drift.
    import os as _os
    skip_sliding = bool(_os.environ.get("GM4_SKIP_SLIDING"))
    skip_global = bool(_os.environ.get("GM4_SKIP_GLOBAL"))
    if lt == "sliding_attention":
        if skip_sliding:
            mixer = ttnn.mul(h_norm, 0.0)
        else:
            rope = (rope_cache[0], rope_cache[1]) if rope_cache is not None else None
            mixer = _layer_pos0_sliding_paged(state, h_norm, w, layer_idx, rope=rope)
    else:
        if skip_global:
            mixer = ttnn.mul(h_norm, 0.0)
        else:
            rope = (rope_cache[2], rope_cache[3]) if rope_cache is not None else None
            mixer = _layer_pos0_global_paged(state, h_norm, w, layer_idx, rope=rope)
    ttnn.deallocate(h_norm)
    post_attn = ttnn.rms_norm(mixer, weight=w["post_attention_layernorm"], epsilon=EPS)
    ttnn.deallocate(mixer)
    h_after_attn = ttnn.add(h_in, post_attn)
    ttnn.deallocate(post_attn)
    pre_ff = ttnn.rms_norm(h_after_attn, weight=w["pre_feedforward_layernorm"], epsilon=EPS)
    # Round 10 (2026-06-06) — DRAM-sharded MLP matmul path. Gated on
    # TT_GM4_DRAM_PREFETCH=1. Reshards `pre_ff` (and `mid` for down_proj) to
    # WIDTH_SHARDED L1 to match the DRAM-sharded matmul's in0 contract; passes
    # the dedicated `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`.
    # Output stays WIDTH_SHARDED L1 between the three matmuls; reshard to
    # interleaved before `all_reduce`. Probe `gm4_dram_sharded_mlp_probe.py`
    # PASSED cos=0.9999937 vs default (3/3 MLP shapes, marginally MORE
    # accurate vs fp32). Fork rationale + landscape: see
    # `research/gemma4_perf_qb2_2026-06-05/tiling_sharding_plan.md` §A1.
    if _dram_sharded_enabled():
        # Per-chip in0 shape for gate/up: [M=32, K=HIDDEN=3840]; for down:
        # [M=32, K=INTERMEDIATE_PER_CHIP=3840] (same K by coincidence of
        # Gemma 4 INTERMEDIATE/NCHIPS == HIDDEN).
        x_mem_cfg_gate_up = _activation_l1_width_sharded(
            state.mesh, M=TILE, K=HIDDEN, num_cores=8)
        x_mem_cfg_down = _activation_l1_width_sharded(
            state.mesh, M=TILE, K=INTERMEDIATE_PER_CHIP, num_cores=8)
        out_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
        prog_gate = _dram_sharded_mlp_program_config_with_gelu(
            M=TILE, K=HIDDEN, N=INTERMEDIATE_PER_CHIP, num_cores=8)
        prog_up = _dram_sharded_mlp_program_config(
            M=TILE, K=HIDDEN, N=INTERMEDIATE_PER_CHIP, num_cores=8)
        # N for down_proj is HIDDEN=3840 (NOT HIDDEN_PER_CHIP=960). The
        # per-chip down weight after `shard_along(axis=0)` is [3840, 3840]
        # — the HIDDEN dim is fully replicated across chips; the partial
        # sum gets all_reduce'd downstream. Bug found 2026-06-06 in Round
        # 10 Phase 4 validator (TT_THROW "Tensor is not allocated" on the
        # mlp_partial matmul because per_core_N = 960/8/32 = 3.75 floored
        # to wrong shape).
        prog_down = _dram_sharded_mlp_program_config(
            M=TILE, K=INTERMEDIATE_PER_CHIP, N=HIDDEN, num_cores=8)

        pre_ff_sh = ttnn.to_memory_config(pre_ff, x_mem_cfg_gate_up)
        ttnn.deallocate(pre_ff)
        gelu_gate_sh = ttnn.matmul(
            pre_ff_sh, w["gate_proj"], program_config=prog_gate,
            memory_config=out_mem_cfg, compute_kernel_config=HIFI4)
        up_sh = ttnn.matmul(
            pre_ff_sh, w["up_proj"], program_config=prog_up,
            memory_config=out_mem_cfg, compute_kernel_config=HIFI4)
        ttnn.deallocate(pre_ff_sh)
        mid_sh = ttnn.mul(gelu_gate_sh, up_sh)
        ttnn.deallocate(gelu_gate_sh); ttnn.deallocate(up_sh)
        # Reshard mid → WIDTH_SHARDED L1 with K=INTERMEDIATE_PER_CHIP contract
        # for down_proj's in0.
        # NOTE 2026-06-06: when the source mem_cfg already matches the target
        # shard spec (as is the case here — gate/up output is [32, 480] per
        # core on 8 cores, same as `x_mem_cfg_down`), `to_memory_config` may
        # return the SAME tensor handle. If we deallocate `mid_sh` after,
        # `mid` becomes unallocated → "Tensor is not allocated" TT_THROW on
        # the next matmul. Keep `mid_sh` alive by reusing the alias and only
        # deallocating after the final matmul.
        mid = ttnn.to_memory_config(mid_sh, x_mem_cfg_down)
        mlp_partial_sh = ttnn.matmul(
            mid, w["down_proj"], program_config=prog_down,
            memory_config=out_mem_cfg, compute_kernel_config=HIFI4)
        ttnn.deallocate(mid_sh)  # also covers `mid` (no-op reshard alias)
        # all_reduce expects INTERLEAVED; reshard back.
        mlp_partial = ttnn.sharded_to_interleaved(
            mlp_partial_sh, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_partial_sh)
    else:
        # Round-4 perf (2026-06-05): fuse gelu into gate_proj matmul via the
        # `activation="gelu"` fused-activation parameter. Per tt-metal
        # unary_op_utils.cpp:833 the string "gelu" maps to
        # UnaryOpType::GELU with fast_and_approximate=false — exactly matches
        # our prior `ttnn.gelu(gate, fast_and_approximate_mode=False)`. Saves
        # 1 op per layer × 48 = 48 ops/forward (fully kernel-time: the SFPU
        # GELU runs inside the matmul out-block writeback, no extra pass).
        # Isolation probe: `experiments/cb/isolate/gm4_matmul_gelu_probe.py` —
        # max|delta| = 0.0 vs separate matmul + gelu (bit-identical).
        gelu_gate = ttnn.matmul(pre_ff, w["gate_proj"], compute_kernel_config=HIFI4,
                                activation="gelu")
        up = ttnn.matmul(pre_ff, w["up_proj"], compute_kernel_config=HIFI4)
        ttnn.deallocate(pre_ff)
        mid = ttnn.mul(gelu_gate, up)
        ttnn.deallocate(gelu_gate); ttnn.deallocate(up)
        mlp_partial = ttnn.matmul(mid, w["down_proj"], compute_kernel_config=HIFI4)
        ttnn.deallocate(mid)
    mlp_out = all_reduce_tt(mlp_partial, state.mesh)
    ttnn.deallocate(mlp_partial)
    post_ff = ttnn.rms_norm(mlp_out, weight=w["post_feedforward_layernorm"], epsilon=EPS)
    ttnn.deallocate(mlp_out)
    # Round 6 perf (2026-06-05): fuse the trailing scalar multiply
    # `h_residual_2 * layer_scalar` into the residual add via the
    # `activations` parameter on `ttnn.add` (UnaryOpType.MUL_UNARY_SFPU).
    # The LLK exposes `mul_unary_tile(idst, scalar)` as a post-add SFPU
    # pass within the same kernel; see tt-metal
    # `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:340`.
    # Saves 1 op per layer × 48 = 48 ops/forward. Per
    # [[feedback-kernel-vs-dispatch-realization]] this is real SFPU work
    # (one less kernel pass over the tile), so the win should realize ~1:1
    # in trace, similar magnitude to the round-4 matmul-gelu fusion.
    # Isolation probe: `experiments/cb/isolate/gm4_add_mul_scalar_probe.py` —
    # cos(baseline, fused) = 0.9999961, max|delta| = 0.000977 (bf16
    # round-off, expected from same-op-order reordering).
    _layer_scalar_act = [ttnn.UnaryWithParam(
        ttnn.UnaryOpType.MUL_UNARY_SFPU, float(w["layer_scalar"]))]
    h_out = ttnn.add(h_after_attn, post_ff, activations=_layer_scalar_act)
    ttnn.deallocate(h_after_attn); ttnn.deallocate(post_ff)
    return h_out


def _set_pos(state, pos):
    """Update cur_pos_buf + rot_idxs_buf for the new decode position via
    in-place copy_host_to_device_tensor (the 27B prod pattern at
    server_tp.py:1607-1619). Recreating the buffers every step (deallocate +
    from_torch) is the [[ttnn-list-rebinding-leaks]] anti-pattern and
    produces garbage attention output past pos 0.
    """
    cur_host = ttnn.from_torch(
        torch.tensor([int(pos)], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    ttnn.copy_host_to_device_tensor(cur_host, state.cur_pos_buf)
    # DEBUG: GM4_ROPE_ZERO forces rot_idxs=0 → RoPE identity at all
    # positions. Use to isolate whether RoPE math or cache-read is the bug.
    import os as _os
    rot_pos = 0 if _os.environ.get("GM4_ROPE_ZERO") else int(pos)
    rot_host = ttnn.from_torch(
        torch.tensor([rot_pos], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    ttnn.copy_host_to_device_tensor(rot_host, state.rot_idxs_buf)


# ── Phase 2.A spec-dec: expose last-layer KV for drafter cross-attention ──
#
# The Gemma 4 12B assistant drafter (server_gemma4_12b_assistant_ttnn.py)
# is STATELESS w.r.t. KV — it cross-attends to the target's KV cache at
# the LAST sliding (L=46) and LAST full (L=47) attention layers per
# `Gemma4UnifiedAssistantForCausalLM.forward(shared_kv_states=...)`.
#
# Per the Phase 2.A.0 layout probe verdict
# (commit `8fbecd5` / `experiments/cb/isolate/gemma4_target_kv_layout_probe.py`):
# the on-device cache layout maps cleanly to HF's
# `(B=1, NKV_TOTAL, L_kv, head_dim)` contract via per-chip slice
# reassembly. Reading back ~5KB/step for sliding + ~1KB/step for full is
# negligible vs the 50 ms target decode step; we keep the readback to
# numpy at v0 and revisit on-device hand-off in Phase 3 perf if needed.
#
# Storage layout per server_gemma4_unified_ttnn.py:683-711:
#   Sliding (L=46): 2 caches per layer.
#     each cache [num_blocks, NCHIPS=4, BLOCK_SIZE=32, HEAD_DIM_SLIDING=256]
#     sharded on dim=1; cache_0 on chip c → KV head 2c (even), cache_1 →
#     KV head 2c+1 (odd). Mesh-wide KV heads = {0..7}.
#   Full (L=47): 1 cache per layer.
#     [num_blocks, NKV_GLOBAL=1, BLOCK_SIZE=32, HEAD_DIM_GLOBAL=512]
#     REPLICATED. Chip-0 view suffices.
#   Token at pos lives at block=pos//32, row=pos%32.

def _read_sliding_cache_per_chip(kc, mesh):
    """Read sliding cache to per-chip numpy stack [NCHIPS, num_blocks, 1,
    BLOCK_SIZE, HEAD_DIM_SLIDING]. Forked verbatim from
    `experiments/cb/isolate/gemma4_target_kv_layout_probe.py` (probe verdict
    in commit `8fbecd5`).
    """
    t = ttnn.to_torch(kc,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    arr = t.float().cpu().numpy()
    total_dim0, n_kv, bs, hd = arr.shape
    assert total_dim0 % NCHIPS == 0, \
        f"sliding cache dim-0 {total_dim0} not divisible by NCHIPS={NCHIPS}"
    per_chip_blocks = total_dim0 // NCHIPS
    return arr.reshape(NCHIPS, per_chip_blocks, n_kv, bs, hd)


def _read_full_cache_replicated(kc, mesh):
    """Read full cache (replicated) to chip-0 numpy view [num_blocks, 1,
    BLOCK_SIZE, HEAD_DIM_GLOBAL]. Forked from probe."""
    t = ttnn.to_torch(kc,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    arr = t.float().cpu().numpy()
    total_dim0, n_kv, bs, hd = arr.shape
    assert total_dim0 % NCHIPS == 0, \
        f"full cache dim-0 {total_dim0} not divisible by NCHIPS={NCHIPS}"
    per_chip_blocks = total_dim0 // NCHIPS
    return arr.reshape(NCHIPS, per_chip_blocks, n_kv, bs, hd)[0]


def read_shared_kv_for_drafter(state, L_kv):
    """Phase 2.A — populate and return `state.shared_kv_for_drafter`.

    Reads the LAST sliding (state.last_sliding_idx) and LAST full
    (state.last_full_idx) paged KV caches off-device, reassembles into
    HF's `(B=1, NKV_TOTAL, L_kv, head_dim)` contract for the drafter's
    `shared_kv_states` consumption.

    Should be called AFTER decoding has advanced `cur_pos_buf` to position
    `L_kv - 1` (so positions 0..L_kv-1 are written in the cache).

    Each call OVERWRITES `state.shared_kv_for_drafter`; the prior round's
    snapshot is discarded (drafter consumes within the same spec-dec
    round per `[[gemma4-mtp-plan-of-action]]` §"Architectural clarifications"
    point 7).

    Returns the dict directly for convenience; also stashed on state for
    callers that want to read it later in the same round (e.g. the spec-dec
    scheduler exposing it to a drafter not co-resident in this Python
    process).

    Args:
        state: Gemma 4 server State (post-bootstrap, post-prefill or
               mid-decode).
        L_kv: number of valid positions in the cache to extract (1..MAX_KV).

    Returns:
        dict with two entries:
          "sliding_attention": (K_np [1, 8, L_kv, 256], V_np same)
          "full_attention":    (K_np [1, 1, L_kv, 512], V_np same)
    """
    assert state.last_sliding_idx is not None and state.last_full_idx is not None, \
        "bootstrap did not set last_sliding_idx / last_full_idx (stale State?)"
    assert 1 <= L_kv <= MAX_KV, f"L_kv={L_kv} out of [1, MAX_KV={MAX_KV}]"
    BLOCK_SIZE = state.sdpa_block_size  # = 32

    # ── sliding (last sliding attn layer) ──
    sliding_caches = state.kv_caches_tt[state.last_sliding_idx]
    assert len(sliding_caches) == 2, \
        f"sliding cache count {len(sliding_caches)} != 2"
    kc0_tt, vc0_tt = sliding_caches[0]
    kc1_tt, vc1_tt = sliding_caches[1]
    K0 = _read_sliding_cache_per_chip(kc0_tt, state.mesh)
    V0 = _read_sliding_cache_per_chip(vc0_tt, state.mesh)
    K1 = _read_sliding_cache_per_chip(kc1_tt, state.mesh)
    V1 = _read_sliding_cache_per_chip(vc1_tt, state.mesh)

    NKV_TOTAL = NUM_KV_HEADS_SLIDING  # = 8
    HD = HEAD_DIM_SLIDING              # = 256
    K_sliding = np.zeros((1, NKV_TOTAL, L_kv, HD), dtype=np.float32)
    V_sliding = np.zeros((1, NKV_TOTAL, L_kv, HD), dtype=np.float32)
    for h in range(NKV_TOTAL):
        chip = h // 2
        cache_idx = h % 2  # 0 → cache_0, 1 → cache_1
        Karr = K0 if cache_idx == 0 else K1
        Varr = V0 if cache_idx == 0 else V1
        for pos in range(L_kv):
            block = pos // BLOCK_SIZE
            row = pos % BLOCK_SIZE
            K_sliding[0, h, pos, :] = Karr[chip, block, 0, row, :]
            V_sliding[0, h, pos, :] = Varr[chip, block, 0, row, :]

    # ── full (last full attn layer) ──
    full_caches = state.kv_caches_tt[state.last_full_idx]
    assert len(full_caches) == 1, \
        f"full cache count {len(full_caches)} != 1"
    kcf_tt, vcf_tt = full_caches[0]
    Kf = _read_full_cache_replicated(kcf_tt, state.mesh)
    Vf = _read_full_cache_replicated(vcf_tt, state.mesh)
    HDF = HEAD_DIM_GLOBAL  # = 512
    K_full = np.zeros((1, 1, L_kv, HDF), dtype=np.float32)
    V_full = np.zeros((1, 1, L_kv, HDF), dtype=np.float32)
    for pos in range(L_kv):
        block = pos // BLOCK_SIZE
        row = pos % BLOCK_SIZE
        K_full[0, 0, pos, :] = Kf[block, 0, row, :]
        V_full[0, 0, pos, :] = Vf[block, 0, row, :]

    state.shared_kv_for_drafter = {
        "sliding_attention": (K_sliding, V_sliding),
        "full_attention":    (K_full,    V_full),
    }
    return state.shared_kv_for_drafter


def reset_shared_kv_for_drafter(state):
    """Drop the prior round's KV snapshot. Called by spec-dec scheduler at
    the start of each new sequence (so a stale snapshot from a finished
    sequence can't leak into the next one). No-op if never populated.
    """
    state.shared_kv_for_drafter = None


def step_forward_v031(state, tok_id, pos, capture=None):
    """v0.3.1 multi-step forward — wraps `step_forward_v03` with a position
    update. KV cache accumulates across calls; RoPE rotates by pos.
    """
    _set_pos(state, pos)
    # DEBUG: confirm on-device buffer values after _set_pos (per-chip read).
    import os as _os
    if _os.environ.get("GM4_DEBUG_POS"):
        cur_val = ttnn.to_torch(state.cur_pos_buf,
                                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
        rot_val = ttnn.to_torch(state.rot_idxs_buf,
                                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
        print(f"  [dbg pos={pos}] cur_pos_buf={cur_val.flatten().tolist()} "
              f"rot_idxs_buf={rot_val.flatten().tolist()}", flush=True)
    return step_forward_v03(state, tok_id, capture=capture)


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

    # Round-3 perf: compute RoPE cos/sin tables ONCE per forward (sliding +
    # global), reuse across all 48 layers. See _compute_rope_for_forward —
    # eliminates 47/48 sliding lookups + 7/8 global lookups (~6.5 ms kernel
    # time / forward per round-1 Tracy v2). Lifetime is exactly one forward.
    rope_cache = _compute_rope_for_forward(state)

    for L in range(NUM_LAYERS):
        h_new = _layer_forward_pos0_paged(state, h, L, rope_cache=rope_cache)
        ttnn.deallocate(h)
        h = h_new
        if capture is not None and capture.get("per_layer", False):
            capture.setdefault("layer_h", {})[L] = _readback_replicated(h, state.mesh)

    _release_rope_for_forward(rope_cache)

    final = ttnn.rms_norm(h, weight=state.final_norm_tt, epsilon=EPS)
    ttnn.deallocate(h)
    if capture is not None:
        capture["final_norm"] = _readback_replicated(final, state.mesh)

    argmax_tt, full_logits = _lm_head_argmax(state, final,
                                              capture_logits=(capture is not None))
    if capture is not None and full_logits is not None:
        capture["logits"] = _readback_replicated(full_logits, state.mesh)
        ttnn.deallocate(full_logits)
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

    argmax_tt, full_logits = _lm_head_argmax(state, final,
                                              capture_logits=(capture is not None))
    if capture is not None and full_logits is not None:
        capture["logits"] = _readback_replicated(full_logits, state.mesh)
        ttnn.deallocate(full_logits)
    arr = ttnn.to_torch(argmax_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    ttnn.deallocate(argmax_tt)
    argmax = int(arr.reshape(-1)[0].item())
    if capture is not None:
        capture["argmax"] = argmax
    return argmax


# ── v0.4 traced decode ────────────────────────────────────────────────
# Mirrors 27B prod (`experiments/serve/server_tp.py:1587-1694, 2142-2186`).
# Pre-allocate `tok_buf` + `cur_pos_buf` + `rot_idxs_buf` in bootstrap;
# `update_input_buffers` writes them out-of-trace via
# `copy_host_to_device_tensor`; `forward_token_gm4_inner` reads ONLY from
# those buffers and returns an on-device argmax tensor; capture once, then
# `execute_trace` per token. Two-phase warmup (2 eager forwards to JIT-compile
# all kernels before capture) — JIT inside `begin_trace_capture` hangs
# Blackhole per [[ttnn-multi-trace-two-phase-warmup]].


def update_input_buffers(state, token_id, cur_pos):
    """Host→device write to tok_buf, cur_pos_buf, rot_idxs_buf. Outside any
    captured trace. Three tiny index writes (no embeds, no logits).

    Lazy-allocates `state.tok_buf` if absent — defensive for the dev
    harness, which keeps state alive across `importlib.reload(base)` and
    so may have a state object that pre-dates a bootstrap-added buffer.
    """
    if getattr(state, "tok_buf", None) is None:
        state.tok_buf = ttnn.from_torch(
            torch.zeros((1, 1), dtype=torch.int32),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
    tok_host = ttnn.from_torch(
        torch.tensor([[int(token_id)]], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    ttnn.copy_host_to_device_tensor(tok_host, state.tok_buf)
    _set_pos(state, cur_pos)  # writes cur_pos_buf + rot_idxs_buf in place


def forward_token_gm4_inner(state):
    """Trace-captureable forward — reads ONLY state.tok_buf / cur_pos_buf /
    rot_idxs_buf, does the embed + RoPE-table-row lookup on-device via
    ttnn.embedding, runs all 48 layers, final_norm + lm_head + softcap +
    argmax. Returns an on-device UINT32 [1, 1] argmax tensor.

    Bit-equivalent to `step_forward_v03` modulo the tok-input plumbing
    (tok_buf vs the from_torch+embedding inside step_forward_v03). v0.4
    validator gate: 100 traced steps produce the same argmax sequence as
    100 eager steps.
    """
    embed = ttnn.embedding(state.tok_buf, state.embed_tt)
    h = ttnn.multiply(ttnn.to_layout(embed, ttnn.TILE_LAYOUT), EMBED_SCALE)
    ttnn.deallocate(embed)
    # Round-3 perf: per-forward rope cache hoisted out of the per-layer
    # hot path (see _compute_rope_for_forward; ~6.5 ms / forward of removed
    # kernel time per round-1 Tracy v2). MUST appear inside this function
    # so it's captured into the trace command list — the rope_cache
    # tensors are produced + consumed within one traced forward.
    rope_cache = _compute_rope_for_forward(state)
    for L in range(NUM_LAYERS):
        h_new = _layer_forward_pos0_paged(state, h, L, rope_cache=rope_cache)
        ttnn.deallocate(h)
        h = h_new
    _release_rope_for_forward(rope_cache)
    final = ttnn.rms_norm(h, weight=state.final_norm_tt, epsilon=EPS)
    ttnn.deallocate(h)
    argmax_tt, _ = _lm_head_argmax(state, final, capture_logits=False)
    return argmax_tt


def ensure_decode_trace(state, log=print):
    """Capture the decode forward once. Subsequent calls reuse the captured
    trace. Two warmup forwards JIT all kernels FIRST (the trace capture
    cannot tolerate JIT — [[ttnn-multi-trace-two-phase-warmup]]).
    """
    if getattr(state, "trace_id", None) is not None:
        return
    log("[trace] warmup + capture decode trace…")
    t0 = time.time()
    # Warmup eager twice — JIT-compile all kernel programs the inner forward
    # touches. Use BOS (token=2) at pos=0,1 so the cache lookups stay valid.
    update_input_buffers(state, token_id=2, cur_pos=0)
    a = forward_token_gm4_inner(state); ttnn.deallocate(a)
    ttnn.synchronize_device(state.mesh)
    update_input_buffers(state, token_id=2, cur_pos=1)
    a = forward_token_gm4_inner(state); ttnn.deallocate(a)
    ttnn.synchronize_device(state.mesh)
    # Capture. Pre-set buffers to the position we'll never replay at, so
    # the first execute_trace call sees only the new values that
    # update_input_buffers writes.
    update_input_buffers(state, token_id=2, cur_pos=2)
    state.trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
    state.traced_argmax_tt = forward_token_gm4_inner(state)
    ttnn.end_trace_capture(state.mesh, state.trace_id, cq_id=0)
    log(f"[trace] captured in {(time.time()-t0)*1000:.0f} ms "
        f"(id={state.trace_id})")


def step_forward_traced(state, token_id, cur_pos):
    """Equivalent to step_forward_v031 but uses the captured trace. Caller
    must have run `ensure_decode_trace(state)` once before the first call.
    Reads back the argmax via to_torch (small UINT32 [1, 1] tensor).
    """
    update_input_buffers(state, token_id, cur_pos)
    ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False)
    arr = ttnn.to_torch(state.traced_argmax_tt,
                        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    return int(arr.reshape(-1)[0].item())


# ── Phase 2.B.1: B=K+1 verify trace for spec-dec ──────────────────────
#
# Layout / decision document: research/gemma4_verify_kp1_readonly_decision.md
# Kernel isolation gate (PASS): experiments/cb/isolate/gemma4_kp1_paged_kernels_smoke.py
#
# READ-ONLY verify variant: K+1 candidate token IDs go through the full
# decoder stack at B=K+1, but the K/V projections' outputs are NEVER
# written to the paged KV cache. All K+1 alias rows attend to the SAME
# physical KV history (current cur_pos's worth of tokens, written by the
# canonical B=1 decode path). This isolates v0 from cache-rewind logic;
# Phase 3 may revisit with write-then-rewind if measured α at K=5
# falls below the projected ~2× speedup floor.
#
# Pre-allocated buffers (set up in setup_verify_kp1_state, captured into
# the verify trace) — never re-bound per step, per
# [[ttnn-list-rebinding-leaks]]:
#   state.verify_tok_buf       uint32 [1, K+1, 1] — K+1 candidate token IDs
#   state.verify_pos_buf       int32  [K+1]      — K+1 positions (all = cur_pos)
#   state.verify_rot_idxs_buf  uint32 [K+1]      — K+1 RoPE indices
#   state.verify_page_table_tt int32  [K+1, num_blocks] — alias page-table
#                                     (rows 0..K all point at row 0's blocks)


def setup_verify_kp1_state(state, K=5, log=print):
    """Allocate B=K+1 verify-trace buffers ONCE. Idempotent: re-calling
    with the same K is a no-op (returns immediately if already set up).
    Re-calling with a DIFFERENT K errors (would require buffer reallocation
    + trace re-capture; caller should pick K at bootstrap time).
    """
    if getattr(state, "verify_K", None) is not None:
        if state.verify_K != K:
            raise ValueError(
                f"verify state already set up at K={state.verify_K}; "
                f"cannot reconfigure to K={K} without restart")
        return
    log(f"[verify] allocating B=K+1={K+1} verify-trace buffers (K={K})")

    state.verify_K = int(K)
    Bv = K + 1
    # tok_buf: uint32 [1, K+1, 1] — one slot per candidate. Updated outside
    # trace via copy_host_to_device_tensor (mirrors decode tok_buf pattern).
    state.verify_tok_buf = ttnn.from_torch(
        torch.zeros((1, Bv, 1), dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    # pos_buf: int32 [K+1] — all entries set to current decode position N.
    # paged SDPA's cur_pos_tensor contract is shape [B] (per-row positions).
    state.verify_pos_buf = ttnn.from_torch(
        torch.zeros((Bv,), dtype=torch.int32),
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    # rot_idxs_buf: uint32 [K+1] — all entries set to current cur_pos.
    # ttnn.embedding takes uint32 indices; gather into cos/sin tables.
    state.verify_rot_idxs_buf = ttnn.from_torch(
        torch.zeros((Bv,), dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    # Alias page-table: rows 0..K all point at the SAME physical blocks
    # (the active prompt's KV blocks). For single-stream spec-dec the active
    # prompt's blocks are just np.arange(num_blocks) (state.page_table_tt's
    # base contents). We materialize the aliased version by replicating that
    # row K+1 times. Forks spec_dec_scheduler.build_verify_alias_page_table_host.
    base_pt_row = np.arange(state.num_blocks, dtype=np.int32)
    alias_pt = np.tile(base_pt_row[None, :], (Bv, 1))  # [K+1, num_blocks]
    state.verify_page_table_tt = ttnn.from_torch(
        torch.from_numpy(alias_pt),
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    log(f"  verify buffers ready: tok[1,{Bv},1] pos[{Bv}] rot[{Bv}] "
        f"page_table[{Bv},{state.num_blocks}]")


def update_verify_inputs(state, current_pos, candidate_token_ids):
    """Host→device write to verify_{tok,pos,rot_idxs}_buf. Outside any
    captured trace. K+1 small writes (no embeds, no big tensors).

    Args:
      state: post-setup_verify_kp1_state state.
      current_pos: int — the position at which all K+1 candidates verify.
        cur_pos_tensor[i] = current_pos for i in 0..K. SDPA reads cache
        history through position current_pos (inclusive).
      candidate_token_ids: sequence of K+1 ints — the K+1 candidate token
        IDs to verify in parallel. Convention: index 0 is the "current"
        token (the bonus continuation if all K draft tokens accept),
        indices 1..K are the K draft tokens. Caller assembles this list.

    Mirrors update_input_buffers (decode B=1) pattern.
    """
    assert state.verify_K is not None, \
        "call setup_verify_kp1_state(state, K) before update_verify_inputs"
    Bv = state.verify_K + 1
    assert len(candidate_token_ids) == Bv, \
        f"need {Bv} candidate token IDs, got {len(candidate_token_ids)}"

    tok_np = np.asarray(candidate_token_ids, dtype=np.int32).reshape(1, Bv, 1)
    tok_host = ttnn.from_torch(
        torch.from_numpy(tok_np),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    ttnn.copy_host_to_device_tensor(tok_host, state.verify_tok_buf)

    pos_np = np.full((Bv,), int(current_pos), dtype=np.int32)
    pos_host = ttnn.from_torch(
        torch.from_numpy(pos_np),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    ttnn.copy_host_to_device_tensor(pos_host, state.verify_pos_buf)

    rot_np = np.full((Bv,), int(current_pos), dtype=np.int32)
    rot_host = ttnn.from_torch(
        torch.from_numpy(rot_np),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    ttnn.copy_host_to_device_tensor(rot_host, state.verify_rot_idxs_buf)


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
