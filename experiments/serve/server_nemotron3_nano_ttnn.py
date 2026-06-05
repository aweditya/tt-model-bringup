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


# ── State ──────────────────────────────────────────────────────────────
class State:
    """All bootstrap state lives here so cb_api can stash it on a single object.

    v0.1.0 attrs:
      mesh, tokenizer, tok (alias), text_cfg, layer_types
      embed_tt, embed_w_np (fallback)
      final_norm_tt
      lm_head_tt  (shape [HIDDEN, VOCAB] for the matmul)
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


def main():
    """Tiny direct-invoke entry point for ad-hoc smoke testing."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = State()
    bootstrap(state, lambda m: print(m, flush=True))
    print("v0.1.0 scaffold bootstrap PASS — no probes wired here, "
          "use nemotron3_v010_bootstrap_smoke.py for the actual gate.")


if __name__ == "__main__":
    main()
