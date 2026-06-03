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
    """All-reduce sum across the (1, NCHIPS) mesh; matches 35B prod path."""
    return ttnn.experimental.all_reduce_async(
        x_tt, math_op=ttnn.ReduceType.Sum, num_links=2,
    )


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
    ttnn.deallocate(v)  # v is not normed (sliding layer; V is raw projection)
    q_n = ttnn.rms_norm(q_h, weight=w0["q_norm"], epsilon=EPS)
    k_n = ttnn.rms_norm(k_h, weight=w0["k_norm"], epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h)

    capture["q_norm_out"] = _readback_sharded_head(q_n, state.mesh, NQ_PER_CHIP, HEAD_DIM_SLIDING)
    capture["k_norm_out"] = _readback_sharded_head(k_n, state.mesh, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING)
    ttnn.deallocate(q_n); ttnn.deallocate(k_n)


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
