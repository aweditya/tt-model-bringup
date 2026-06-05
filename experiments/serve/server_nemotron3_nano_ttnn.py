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
N_GROUP_MOE = 8
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
    state.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NCHIPS))
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
            else:
                raise NotImplementedError(
                    f"v0.1.1 only supports attention layers; L{L} is {kind!r}")
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


def numpy_sdpa_gqa_causal(q_np, k_np, v_np, num_q_heads, num_kv_heads, head_dim):
    """Numpy fp32 reference: causal SDPA with GQA over q/k/v of shape
    [B, S, NQ*HD], [B, S, NKV*HD], [B, S, NKV*HD]. Returns
    [B, S, NQ*HD]. Use ONLY for v0.1.1 attention validation —
    on-device prefill SDPA lands at v0.5 perf.
    """
    B, S, _ = q_np.shape
    HD = head_dim
    NQ = num_q_heads
    NKV = num_kv_heads
    G = NQ // NKV  # GQA group size

    # Reshape to [B, S, NH, HD] → [B, NH, S, HD]
    q = q_np.reshape(B, S, NQ, HD).transpose(0, 2, 1, 3).astype(np.float32)
    k = k_np.reshape(B, S, NKV, HD).transpose(0, 2, 1, 3).astype(np.float32)
    v = v_np.reshape(B, S, NKV, HD).transpose(0, 2, 1, 3).astype(np.float32)

    # GQA broadcast — repeat k, v G times along the head dim.
    k = np.repeat(k, G, axis=1)  # [B, NQ, S, HD]
    v = np.repeat(v, G, axis=1)

    # Scores: q @ k.T / sqrt(HD)  →  [B, NQ, S, S]
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(HD))

    # Causal mask (upper-triangular -inf)
    mask = np.triu(np.ones((S, S), dtype=np.float32), k=1) * -1e9
    scores = scores + mask[None, None, :, :]

    # Softmax over last dim
    scores -= scores.max(axis=-1, keepdims=True)
    np.exp(scores, out=scores)
    scores /= scores.sum(axis=-1, keepdims=True)

    # attn_out = scores @ v → [B, NQ, S, HD] → [B, S, NQ*HD]
    out = scores @ v
    out = out.transpose(0, 2, 1, 3).reshape(B, S, NQ * HD)
    return out


def attn_block_eager(state: State, h_input_np, layer_idx: int):
    """v0.1.1.b — full attention block (pre-norm + qkv + SDPA + o_proj
    + residual). SDPA runs in numpy fp32 (NKV=2 × NCHIPS=4 doesn't
    shard; on-device SDPA is a v0.5 perf optimization).

    Input:  h_input_np  numpy fp32  [B, S, HIDDEN] (or [S, HIDDEN])
    Returns dict with: h_norm, q, k, v (from v0.1.1.a path),
                       attn_out (post-SDPA, pre-o_proj),
                       o_proj_out (post-o_proj),
                       block_out (post-residual = L5 block output).
    """
    w = state.per_layer_tt[layer_idx]
    assert w is not None and w["kind"] == "attention"

    # Stage 1: pre-norm + projections on TT (already validated v0.1.1.a).
    res = attn_projections_only(state, h_input_np, layer_idx)
    q_np, k_np, v_np = res["q"], res["k"], res["v"]
    h_input_np_3d = h_input_np if h_input_np.ndim == 3 else h_input_np[None]
    if q_np.ndim == 2:
        q_np = q_np[None]; k_np = k_np[None]; v_np = v_np[None]

    # Stage 2: numpy SDPA (causal + GQA broadcast). Replaceable with
    # an on-device prefill SDPA at v0.5.
    attn_out_np = numpy_sdpa_gqa_causal(
        q_np, k_np, v_np,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM_ATTN,
    )

    # Stage 3: o_proj on TT.
    attn_out_tt = ttnn.from_torch(
        torch.from_numpy(attn_out_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    o_tt = ttnn.matmul(attn_out_tt, w["o_proj"], compute_kernel_config=HIFI4)
    o_np = ttnn.to_torch(
        o_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )[:1].float().numpy()

    # Stage 4: residual add (host-side; cheap).
    block_out = h_input_np_3d.astype(np.float32) + o_np

    out = {
        "h_norm":     res["h_norm"],
        "q":          q_np,
        "k":          k_np,
        "v":          v_np,
        "attn_out":   attn_out_np,
        "o_proj_out": o_np,
        "block_out":  block_out,
    }
    if h_input_np.ndim == 2:
        for k in ["attn_out", "o_proj_out", "block_out"]:
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
