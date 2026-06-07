#!/usr/bin/env python3
"""Gemma 4 12B IT assistant DRAFTER — bringup server for spec-dec.

Fork base: `experiments/serve/server_gemma4_unified_ttnn.py` per REUSE
MANDATE. Target host: qb2 (same as target). This is the spec-dec drafter
described in `research/gemma4_assistant_feasibility.md` (verdict (a)) and
phased in `research/gemma4_mtp_plan_of_action.md`.

## Drafter architecture (from feasibility doc + introspect)

- 4 Gemma 4 layers: layer_types = [sliding, sliding, sliding, full]
- hidden_size = 1024 (drafter), backbone_hidden_size = 3840 (target)
- num_q_heads = 16, num_kv_heads_sliding = 8, num_kv_heads_full = 1
- head_dim_sliding = 256, head_dim_full = 512  (DUAL like target)
- intermediate = 8192, sliding_window = 1024, vocab = 262144
- pre_projection: Linear(7680, 1024) on concat(target_h_last, target_h_prev)
- post_projection: Linear(1024, 3840) back to target's hidden_size
- lm_head: Linear(1024, 262144), tied to embed (tie_word_embeddings=True)
- masked_embedding = None (use_ordered_embeddings=False for 12B)
- **NO k_proj, v_proj, k_norm per layer** — drafter cross-attends to
  target's shared_kv_states (K, V come from target's last sliding + last
  full attn layers; injected via the spec-dec scheduler).
- per-layer: input_layernorm, post_attention_layernorm,
  pre_feedforward_layernorm, post_feedforward_layernorm, layer_scalar,
  self_attn.q_proj, self_attn.q_norm, self_attn.o_proj, mlp.{gate,up,down}

Weight key prefix: `model.` (NOT `model.language_model.` like target). Per
`experiments/utils/gemma4_assistant_weights_introspect.py`.

Determinism patches B (use_multicore=False on argmax) + D (fp32_dest_acc on
lm_head matmul) inherited from target — already shipped via HIFI4 config.

## Staging

- **v0.1.0** (THIS file initially): bootstrap (open mesh, upload embed +
  pre_projection + post_projection + lm_head + 4 layer weight dicts).
  Embed-only probe + pre_projection probe validate against HF oracle.
- v0.2: full 4-layer forward with shared_kv_states injection — argmax
  matches HF on the 5-prompt oracle.
- v0.3: multi-step decode + trace capture.

## v0.1 probes

- `experiments/cb/isolate/gemma4_assistant_embed_smoke.py`
  Embed-only — load input_ids from oracle, embed on device, compare to
  numpy reference. Gate: cos >= 0.999.
- `experiments/cb/isolate/gemma4_assistant_pre_projection_smoke.py`
  Run target_h_last + target_h_prev concat through pre_projection.
  Gate: cos >= 0.999 vs HF (drafter_inputs_embeds_after_pre_proj).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402

# ── Model constants (drafter shapes; verified via weights introspect) ──
MODEL_ID = "google/gemma-4-12b-it-assistant"
HIDDEN = 1024                  # drafter hidden_size
BACKBONE_HIDDEN = 3840         # target hidden_size; pre_projection in=2*3840
N_LAYERS = 4
NUM_Q_HEADS = 16
NUM_KV_HEADS_SLIDING = 8
NUM_KV_HEADS_FULL = 1          # from num_global_key_value_heads
HEAD_DIM_SLIDING = 256
HEAD_DIM_FULL = 512
INTERMEDIATE = 8192
SLIDING_WINDOW = 1024
VOCAB = 262144
ROPE_THETA_SLIDING = 10000.0
ROPE_THETA_FULL = 1000000.0
PARTIAL_ROTARY_FULL = 0.25
EMBED_SCALE = math.sqrt(HIDDEN)  # = 32.0
EPS = 1e-6

# Layer-type schedule, fixed by config (NOT plumbed from disk):
LAYER_TYPES = ["sliding_attention", "sliding_attention",
               "sliding_attention", "full_attention"]

NCHIPS = 4
NQ_PER_CHIP = NUM_Q_HEADS // NCHIPS  # 4
NKV_PER_CHIP_SLIDING = NUM_KV_HEADS_SLIDING // NCHIPS  # 2
GQA_GROUP_SLIDING = NUM_Q_HEADS // NUM_KV_HEADS_SLIDING  # 2
HIDDEN_PER_CHIP = HIDDEN // NCHIPS  # 256
INTERMEDIATE_PER_CHIP = INTERMEDIATE // NCHIPS  # 2048
VOCAB_PER_CHIP = VOCAB // NCHIPS  # 65536

# HiFi4 + fp32_dest_acc — same recipe as target. Determinism patch D.
HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ── Upload helpers (reused from target per REUSE MANDATE) ──────────────
def np_to_replicated(arr, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch.from_numpy(arr.astype(np.float32)),
        dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def np_stacked_to_sharded(per_chip_list, mesh, dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT):
    """Forks `server_gemma4_unified_ttnn.np_stacked_to_sharded`. 1D sharder."""
    stacked = np.stack(per_chip_list, axis=0).astype(np.float32)
    return ttnn.from_torch(
        torch.from_numpy(stacked),
        dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def shard_along(arr, axis, n=NCHIPS):
    return [s.copy() for s in np.split(arr, n, axis=axis)]


def load_t(key_to_shard, key):
    path = key_to_shard[key]
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key).float().numpy()


def build_key_to_shard():
    """Build {key -> shard_path} index for the drafter snapshot.

    Drafter ships as a SINGLE safetensors file (no index json). Forks the
    target's `build_key_to_shard` else-branch.
    """
    cache_dirname = "models--google--gemma-4-12b-it-assistant"
    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_dirname / "snapshots"
    if not snapshot_root.exists():
        raise FileNotFoundError(
            f"no HF snapshot at {snapshot_root}. "
            f"Fetch with huggingface_hub.snapshot_download(MODEL_ID).")
    snap = None
    for cand in snapshot_root.iterdir():
        if not cand.is_dir():
            continue
        if (cand / "model.safetensors.index.json").exists() or \
                any(cand.glob("*.safetensors")):
            snap = cand
            break
    if snap is None:
        raise FileNotFoundError(f"no snapshot weights under {snapshot_root}")
    index = snap / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text())
        return {k: str(snap / v) for k, v in idx["weight_map"].items()}
    sf = next(snap.glob("*.safetensors"))
    with safe_open(sf, framework="pt") as f:
        return {k: str(sf) for k in f.keys()}


# ── Per-layer attention weight upload (drafter shapes) ─────────────────
def upload_attn_layer_sliding(layer_sd, mesh):
    """Upload q_proj / q_norm / o_proj for a SLIDING layer.

    NO k_proj, v_proj, k_norm — drafter cross-attends to target's KV.

    Shapes (HF stores [out, in]; we transpose to [in, out] for ttnn.matmul):
    - q_proj: [HIDDEN=1024, NUM_Q*HEAD_DIM=4096]
              → shard along OUTPUT axis: each chip [1024, NQ_PER_CHIP*256=1024]
    - o_proj: [NUM_Q*HEAD_DIM=4096, HIDDEN=1024]
              → shard along INPUT axis: each chip [1024, 1024]
    - q_norm: [HEAD_DIM=256] replicated
    """
    w = {}
    q_w = layer_sd["self_attn.q_proj.weight"].T  # [1024, 4096]
    o_w = layer_sd["self_attn.o_proj.weight"].T  # [4096, 1024]
    w["q_proj"] = np_stacked_to_sharded(shard_along(q_w, axis=1), mesh)
    w["o_proj"] = np_stacked_to_sharded(shard_along(o_w, axis=0), mesh)
    w["q_norm"] = np_to_replicated(layer_sd["self_attn.q_norm.weight"], mesh)
    return w


def upload_attn_layer_full(layer_sd, mesh):
    """Upload q_proj / q_norm / o_proj for a FULL (head_dim=512) layer.

    L3 only in drafter. q_proj [1024, 8192], o_proj [8192, 1024], q_norm [512].
    """
    w = {}
    q_w = layer_sd["self_attn.q_proj.weight"].T  # [1024, 8192]
    o_w = layer_sd["self_attn.o_proj.weight"].T  # [8192, 1024]
    w["q_proj"] = np_stacked_to_sharded(shard_along(q_w, axis=1), mesh)
    w["o_proj"] = np_stacked_to_sharded(shard_along(o_w, axis=0), mesh)
    w["q_norm"] = np_to_replicated(layer_sd["self_attn.q_norm.weight"], mesh)
    return w


def upload_mlp_layer(layer_sd, mesh):
    """Upload MLP (gate, up, down) for a drafter layer.

    Drafter shapes (smaller than target):
    - gate_proj, up_proj: [HIDDEN=1024, INTERMEDIATE=8192]
              → shard along OUTPUT axis (each chip [1024, 2048])
    - down_proj: [INTERMEDIATE=8192, HIDDEN=1024]
              → shard along INPUT axis (each chip [2048, 1024])

    bf16 default. (No bfp8 path — drafter is small enough not to need it.)
    """
    w = {}
    gate_w = layer_sd["mlp.gate_proj.weight"].T  # [1024, 8192]
    up_w = layer_sd["mlp.up_proj.weight"].T
    down_w = layer_sd["mlp.down_proj.weight"].T  # [8192, 1024]
    w["gate_proj"] = np_stacked_to_sharded(shard_along(gate_w, axis=1), mesh)
    w["up_proj"]   = np_stacked_to_sharded(shard_along(up_w,   axis=1), mesh)
    w["down_proj"] = np_stacked_to_sharded(shard_along(down_w, axis=0), mesh)
    return w


# ── State ──────────────────────────────────────────────────────────────
class State:
    """Per-process drafter state. v0.1.0 carries minimal fields; expands
    with shared_kv_states injection points in v0.2.
    """
    def __init__(self):
        self.mesh = None
        self.cfg = None
        self.text_cfg = None
        self.layer_types = None
        self.tokenizer = None
        self.tok = None
        # Top-level weights
        self.embed_tt = None
        self.embed_w_np = None
        self.pre_projection_tt = None
        self.post_projection_tt = None
        self.final_norm_tt = None
        self.lm_head_tt = None  # tied to embed; uploaded as separate vocab-sharded
        # Per-layer weight dicts
        self.per_layer_tt = []
        # head_dim helpers for v_norm (ones, no scale) — sliding only here.
        self.ones_head_dim_sliding = None
        self.ones_head_dim_full = None
        # SDPA compute kernel config (HiFi4 + fp32_dest_acc).
        self.sdpa_compute_kernel_config = None
        # Misc
        self.vocab_size = None


# ── Bootstrap ──────────────────────────────────────────────────────────
def bootstrap(state, log=None):
    if log is None:
        log = print

    t_total = time.time()
    log("[drafter bootstrap] open mesh + fabric…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    # trace_region_size: drafter is small (4 layers). 100 MB is generous.
    # v0.4 will tune; for now leave headroom for future verify trace.
    state.mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, NCHIPS), trace_region_size=100_000_000,
    )
    log(f"  mesh: {state.mesh}")

    log(f"[drafter bootstrap] model={MODEL_ID} — config + tokenizer…")
    cache_dirname = "models--google--gemma-4-12b-it-assistant"
    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_dirname / "snapshots"
    snap = next(snapshot_root.iterdir())
    cfg_json = json.loads((snap / "config.json").read_text())
    state.cfg = cfg_json
    state.text_cfg = cfg_json["text_config"]
    state.layer_types = list(state.text_cfg["layer_types"])
    log(f"  {len(state.layer_types)} layers; "
        f"{sum(1 for t in state.layer_types if t == 'sliding_attention')} sliding / "
        f"{sum(1 for t in state.layer_types if t == 'full_attention')} full")
    assert state.layer_types == LAYER_TYPES, (
        f"layer_types mismatch — config={state.layer_types}, "
        f"hardcoded={LAYER_TYPES}. Reconfirm via introspect.")

    # Tokenizer (shared with target; lives in target's cache).
    try:
        from transformers import AutoTokenizer
        # Use target's tokenizer — drafter shares vocab (262144 ≡ target's).
        state.tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
        state.tok = state.tokenizer
        log(f"  tokenizer: {state.tokenizer.__class__.__name__}")
    except Exception as e:
        log(f"  tokenizer load skipped: {e!r}")

    log("[drafter bootstrap] enumerate shards + load top-level weights…")
    key_to_shard = build_key_to_shard()
    log(f"  {len(key_to_shard)} keys total (single safetensors expected)")

    # v_norm helper (drafter inherits Gemma 4's v_norm with_scale=False
    # pattern — but actually drafter has NO v_proj... but K/V come from
    # target which is normalised on target's side). Still preload the
    # all-ones vectors in case v0.2 needs to renormalize the injected V.
    # Cheap; one copy per head_dim.
    state.ones_head_dim_sliding = np_to_replicated(
        np.ones(HEAD_DIM_SLIDING, dtype=np.float32), state.mesh)
    state.ones_head_dim_full = np_to_replicated(
        np.ones(HEAD_DIM_FULL, dtype=np.float32), state.mesh)

    # SDPA compute kernel config — HiFi4 + fp32_dest_acc (matches target's
    # `sdpa_compute_kernel_config` setup at server_gemma4_unified_ttnn.py
    # bootstrap).
    state.sdpa_compute_kernel_config = HIFI4

    # ── Embed: replicated, ROW_MAJOR (ttnn.embedding requires it) ──
    # Tied to lm_head. Drafter prefix is `model.` (NOT `model.language_model.`).
    embed_w_np = load_t(key_to_shard, "model.embed_tokens.weight")  # [V, H] = [262144, 1024]
    state.embed_tt = ttnn.from_torch(
        torch.from_numpy(embed_w_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    state.embed_w_np = embed_w_np
    log(f"  embed: {embed_w_np.shape} (replicated bf16)")

    # ── Final norm (model.norm.weight) — Gemma 4 Llama RMSNorm, NO +1.0 ──
    final_norm_w = load_t(key_to_shard, "model.norm.weight")  # [1024]
    state.final_norm_tt = np_to_replicated(final_norm_w, state.mesh)
    log(f"  final_norm: {final_norm_w.shape}")

    # ── pre_projection: Linear(7680, 1024). Weight shape on disk = [out, in]
    #    = [1024, 7680]. Transpose to [7680, 1024] for ttnn.matmul(x, W).
    #    Input is [B, L, 7680] (concat 2 target hidden states). Replicate the
    #    weight (small enough — 7680*1024*2 = 15.7 MB / chip).
    pre_proj_w = load_t(key_to_shard, "pre_projection.weight")  # [1024, 7680]
    assert pre_proj_w.shape == (HIDDEN, 2 * BACKBONE_HIDDEN), \
        f"pre_projection.weight shape {pre_proj_w.shape} != ({HIDDEN}, {2*BACKBONE_HIDDEN})"
    pre_proj_wT = pre_proj_w.T.copy()  # [7680, 1024]
    state.pre_projection_tt = np_to_replicated(pre_proj_wT, state.mesh)
    log(f"  pre_projection: HF shape {pre_proj_w.shape} → ttnn [{pre_proj_wT.shape}] replicated")

    # ── post_projection: Linear(1024, 3840). HF = [out, in] = [3840, 1024].
    #    Replicate (small: 1024*3840*2 = 7.9 MB / chip).
    post_proj_w = load_t(key_to_shard, "post_projection.weight")  # [3840, 1024]
    assert post_proj_w.shape == (BACKBONE_HIDDEN, HIDDEN), \
        f"post_projection.weight shape {post_proj_w.shape} != ({BACKBONE_HIDDEN}, {HIDDEN})"
    post_proj_wT = post_proj_w.T.copy()  # [1024, 3840]
    state.post_projection_tt = np_to_replicated(post_proj_wT, state.mesh)
    log(f"  post_projection: HF shape {post_proj_w.shape} → ttnn [{post_proj_wT.shape}] replicated")

    # ── lm_head: tied to embed. Same vocab-shard pattern as target P22
    #    (each chip holds [HIDDEN, VOCAB/NCHIPS]). ShardTensorToMesh(dim=1).
    #    VOCAB=262144 / 4 = 65536 per chip (tile-aligned).
    assert VOCAB % NCHIPS == 0, f"VOCAB {VOCAB} not divisible by NCHIPS"
    state.vocab_size = VOCAB
    state.lm_head_tt = ttnn.from_torch(
        torch.from_numpy(embed_w_np.T.astype(np.float32)),  # [1024, 262144]
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
    )
    log(f"  lm_head (tied, vocab-sharded dim=1, per-chip {VOCAB // NCHIPS}): "
        f"{embed_w_np.T.shape}")

    # ── Per-layer weights ──
    log(f"[drafter bootstrap] uploading {N_LAYERS} layer weights to mesh…")
    t0 = time.time()
    state.per_layer_tt = []
    for L in range(N_LAYERS):
        prefix = f"model.layers.{L}."
        layer_sd = {k.replace(prefix, ""): load_t(key_to_shard, k)
                    for k in key_to_shard if k.startswith(prefix)}
        layer_tt = {}
        # All four norms: Gemma 4 Llama-style `w` (NO +1.0).
        layer_tt["input_layernorm"] = np_to_replicated(
            layer_sd["input_layernorm.weight"], state.mesh)
        layer_tt["post_attention_layernorm"] = np_to_replicated(
            layer_sd["post_attention_layernorm.weight"], state.mesh)
        layer_tt["pre_feedforward_layernorm"] = np_to_replicated(
            layer_sd["pre_feedforward_layernorm.weight"], state.mesh)
        layer_tt["post_feedforward_layernorm"] = np_to_replicated(
            layer_sd["post_feedforward_layernorm.weight"], state.mesh)

        if state.layer_types[L] == "sliding_attention":
            layer_tt.update(upload_attn_layer_sliding(layer_sd, state.mesh))
        else:
            layer_tt.update(upload_attn_layer_full(layer_sd, state.mesh))
        layer_tt.update(upload_mlp_layer(layer_sd, state.mesh))
        # layer_scalar: stored as float on host (target precedent).
        layer_tt["layer_scalar"] = float(
            np.asarray(layer_sd["layer_scalar"]).reshape(-1)[0])
        state.per_layer_tt.append(layer_tt)
        log(f"  L{L} ({state.layer_types[L]}): "
            f"q_proj +o_proj +q_norm +4 norms +MLP +layer_scalar="
            f"{layer_tt['layer_scalar']:.4f} uploaded")
    log(f"  all layer weights uploaded in {time.time()-t0:.1f}s")
    log(f"[drafter bootstrap] DONE in {time.time()-t_total:.1f}s "
        f"— v0.1.0 (embed + pre_proj + post_proj + lm_head + 4 layers)")


# ── v0.1 helpers exposed for the probes ────────────────────────────────
def _readback_replicated(t_tt, mesh):
    """Read back a replicated tensor; return as a flat fp32 numpy array."""
    arr = ttnn.to_torch(t_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    if arr.ndim >= 1 and arr.shape[0] == NCHIPS:
        arr = arr[0]
    return arr.float().reshape(-1).numpy()


def embed_lookup_tt(state, input_ids_np):
    """v0.1 embed probe — lookup token IDs and return numpy [B, L, HIDDEN].

    NO embed scale, NO rms norm — pure table lookup. The drafter's HF code
    applies `inputs_embeds * sqrt(hidden_size)` like target Gemma 4; this
    probe is the un-scaled table lookup only, matching the un-scaled
    `target_h_*` we'd compare against in the most basic gate. But the
    cleanest test compares against `state.embed_w_np[input_ids]` — pure
    table indexing. That's bit-perfect modulo bf16 conversion.

    Args:
        input_ids_np: numpy int array of shape [B, L] (B=1 supported).
    Returns:
        numpy fp32 array of shape [B, L, HIDDEN].
    """
    assert input_ids_np.ndim == 2
    B, L = input_ids_np.shape
    tok_tt = ttnn.from_torch(
        torch.from_numpy(input_ids_np.astype(np.int32)),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    embed = ttnn.embedding(tok_tt, state.embed_tt)  # [B, L, HIDDEN] replicated
    ttnn.deallocate(tok_tt)
    out_np = _readback_replicated(embed, state.mesh).reshape(B, L, HIDDEN)
    ttnn.deallocate(embed)
    return out_np


def pre_projection_tt(state, inputs_embeds_np):
    """v0.1 pre_projection probe — Linear(7680, 1024).

    Args:
        inputs_embeds_np: numpy fp32 [B, L, 2*3840=7680].
    Returns:
        numpy fp32 [B, L, HIDDEN=1024].
    """
    assert inputs_embeds_np.ndim == 3
    B, L, D = inputs_embeds_np.shape
    assert D == 2 * BACKBONE_HIDDEN, \
        f"expected last dim {2*BACKBONE_HIDDEN}, got {D}"
    # Upload as TILE_LAYOUT replicated (the weight is replicated too).
    # ttnn.matmul on replicated x replicated → replicated.
    x_tt = ttnn.from_torch(
        torch.from_numpy(inputs_embeds_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    y_tt = ttnn.matmul(x_tt, state.pre_projection_tt,
                       compute_kernel_config=HIFI4)
    ttnn.deallocate(x_tt)
    out_np = _readback_replicated(y_tt, state.mesh).reshape(B, L, HIDDEN)
    ttnn.deallocate(y_tt)
    return out_np


# ── v0.2: cross-attention helpers + drafter layer forward ──────────────
#
# Drafter v0.2 design (cross-attention to target's KV per HF
# Gemma4UnifiedAssistantForCausalLM, modeling_gemma4_unified_assistant.py +
# Gemma4TextAttention.is_kv_shared_layer branch in modeling_gemma4.py:
# 1252-1262):
#
# 1. h_in is REPLICATED [1, 1, HIDDEN=1024] across the mesh.
# 2. h_norm = rms_norm(h_in, input_layernorm_weight).
# 3. q = h_norm @ q_proj_sharded → per chip [1, 1, NQ_PER_CHIP*head_dim].
#    Reshape to [NQ_PER_CHIP, head_dim].
# 4. q_norm via ttnn.rms_norm with `weight=q_norm` (Gemma 4 Llama-style — NO
#    +1.0 offset; see target's _layer_pos0_sliding_paged at unified server
#    line 1426).
# 5. RoPE on q at position_ids=[0] (drafter L=1 ⇒ position 0 ⇒ cos=1, sin=0
#    ⇒ identity; SKIP entirely for clarity + perf).
# 6. K, V from `shared_kv_states[layer_type]` — uploaded per-call as
#    sharded (sliding) or replicated (full) onto the mesh. They were
#    already RoPE'd + v_norm'd by the target.
# 7. SDPA: ttnn.transformer.scaled_dot_product_attention(Q, K, V,
#    is_causal=False, scale=1.0) — supports GQA natively per
#    [[reference-ttnn-prefill-sdpa-gqa-native]]. scale=1.0 because Gemma 4
#    text attention sets self.scaling=1.0 (modeling_gemma4.py:1184).
# 8. attn → reshape [1, NQ_PER_CHIP*head_dim] → o_proj sharded matmul +
#    all_reduce.
# 9. residual_1 = h_in + post_attention_layernorm(attn_out)
# 10. mlp: pre_feedforward_layernorm → gate_proj(gelu) * up_proj → down_proj
#     + all_reduce → post_feedforward_layernorm.
# 11. h_out = (residual_1 + mlp_out) * layer_scalar.


def _upload_kv_sliding(state, K_np, V_np):
    """Upload sliding shared_kv_states to mesh.

    Shapes from HF oracle: K, V [B=1, NKV=8, L_kv, head_dim=256] fp32 (already
    RoPE'd + v_norm'd by target).

    Shard along NKV=8 → NKV_PER_CHIP=2 per chip [1, 2, L_kv, 256].
    `ShardTensorToMesh(dim=1)` keeps B leading and shards heads cleanly.

    Q-head h ∈ [0..15] across the mesh maps to KV-head `h // GQA_GROUP = h//2`.
    Chip c holds Q-heads `[4c..4c+3]` (via q_proj output sharding) → KV-heads
    `[2c, 2c+1]` — same per-chip alignment as sharding K, V on dim=1.
    """
    assert K_np.shape[1] == NUM_KV_HEADS_SLIDING, \
        f"K_np head dim mismatch: {K_np.shape} expected NKV=8"
    K_tt = ttnn.from_torch(
        torch.from_numpy(K_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
    )
    V_tt = ttnn.from_torch(
        torch.from_numpy(V_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
    )
    return K_tt, V_tt


def _upload_kv_full(state, K_np, V_np):
    """Upload full shared_kv_states to mesh.

    Shapes from HF oracle: K, V [B=1, NKV=1, L_kv, head_dim=512] fp32 (already
    p-RoPE'd + v_norm'd by target).

    Replicate across mesh — each chip's 4 Q heads (NQ_PER_CHIP=4) all attend
    to this single KV head (GQA group = NQ/NKV = 16 mesh-wide; per chip 4).
    """
    assert K_np.shape[1] == NUM_KV_HEADS_FULL, \
        f"K_np head dim mismatch: {K_np.shape} expected NKV=1"
    K_tt = ttnn.from_torch(
        torch.from_numpy(K_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    V_tt = ttnn.from_torch(
        torch.from_numpy(V_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    return K_tt, V_tt


def _drafter_attn_sliding(state, h_norm, w, K_tt, V_tt):
    """Drafter sliding attention: cross-attend Q (drafter) to K, V (target).

    Shapes:
      h_norm: replicated [1, 1, HIDDEN=1024]
      K_tt, V_tt: sharded dim=1, per chip [1, NKV_PER_CHIP_SLIDING=2, L_kv, 256]
      Returns: replicated [1, 1, HIDDEN] (after o_proj + all_reduce).
    """
    # q = h_norm @ q_proj_sharded   per chip [1, 1, NQ_PER_CHIP*head_dim=1024]
    q = ttnn.matmul(h_norm, w["q_proj"], compute_kernel_config=HIFI4)
    # Reshape to per-head: [1, 1, NQ_PER_CHIP, head_dim].
    q_h = ttnn.reshape(q, [1, 1, NQ_PER_CHIP, HEAD_DIM_SLIDING])
    ttnn.deallocate(q)
    # q_norm: rms_norm with learned weight, NO +1.0 (Gemma 4 Llama-style).
    # ttnn.rms_norm needs rank ≥ 2; pass weight of shape [head_dim]. q_h's
    # last dim is head_dim — broadcasts cleanly.
    q_n = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    ttnn.deallocate(q_h)
    # RoPE skipped — position_ids=[0] → cos=1, sin=0 → identity.

    # SDPA contract: Q [B, NQ, sq, dh], K/V [B, NKV, sk, dh]. Transpose Q so
    # that NQ is dim 1 (Q is currently [1, 1, NQ_PER_CHIP, head_dim] = [B, sq,
    # NQ, dh]). Permute (0, 2, 1, 3).
    q_for_sdpa = ttnn.permute(q_n, [0, 2, 1, 3])
    ttnn.deallocate(q_n)
    # K, V already [1, NKV_PER_CHIP_SLIDING, L_kv, 256] — correct contract.
    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q_for_sdpa, K_tt, V_tt,
        is_causal=False,
        scale=1.0,  # Gemma 4 text attention: self.scaling=1.0.
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )
    ttnn.deallocate(q_for_sdpa)
    # attn_out [1, NQ_PER_CHIP, 1, head_dim] → flatten to [1, 1, NQ_PER_CHIP*hd]
    attn_perm = ttnn.permute(attn_out, [0, 2, 1, 3])
    ttnn.deallocate(attn_out)
    attn_flat = ttnn.reshape(attn_perm,
                              [1, 1, NQ_PER_CHIP * HEAD_DIM_SLIDING])
    ttnn.deallocate(attn_perm)
    # o_proj input-sharded + all_reduce.
    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    out = all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


def _drafter_attn_full(state, h_norm, w, K_tt, V_tt):
    """Drafter full attention (L3): cross-attend Q to single KV head (replicated).

    Shapes:
      h_norm: replicated [1, 1, HIDDEN=1024]
      K_tt, V_tt: REPLICATED [1, NKV=1, L_kv, head_dim=512]
      Returns: replicated [1, 1, HIDDEN].
    """
    q = ttnn.matmul(h_norm, w["q_proj"], compute_kernel_config=HIFI4)
    q_h = ttnn.reshape(q, [1, 1, NQ_PER_CHIP, HEAD_DIM_FULL])
    ttnn.deallocate(q)
    q_n = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    ttnn.deallocate(q_h)
    # RoPE: position 0 → identity. SKIPPED.
    q_for_sdpa = ttnn.permute(q_n, [0, 2, 1, 3])  # [1, NQ_PER_CHIP, 1, 512]
    ttnn.deallocate(q_n)
    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q_for_sdpa, K_tt, V_tt,
        is_causal=False,
        scale=1.0,
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )
    ttnn.deallocate(q_for_sdpa)
    attn_perm = ttnn.permute(attn_out, [0, 2, 1, 3])
    ttnn.deallocate(attn_out)
    attn_flat = ttnn.reshape(attn_perm,
                              [1, 1, NQ_PER_CHIP * HEAD_DIM_FULL])
    ttnn.deallocate(attn_perm)
    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    out = all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


def drafter_layer_forward(state, h_in, layer_idx, K_tt, V_tt):
    """One Gemma 4 drafter decoder layer.

    Args:
      h_in:  replicated [1, 1, HIDDEN] (TILE_LAYOUT bf16)
      layer_idx: int in [0..N_LAYERS-1]
      K_tt, V_tt: the shared KV for this layer's `layer_type` — produced
                  once per forward by `_upload_kv_sliding` or `_upload_kv_full`.
    Returns: replicated [1, 1, HIDDEN].

    Forks `experiments/serve/server_gemma4_unified_ttnn.py:_layer_forward_pos0`
    structurally; drops K/V projection (drafter has none) and the per-layer
    KV cache (drafter cross-attends to target's KV).
    """
    w = state.per_layer_tt[layer_idx]
    lt = state.layer_types[layer_idx]

    h_norm = ttnn.rms_norm(h_in, weight=w["input_layernorm"], epsilon=EPS)
    if lt == "sliding_attention":
        mixer = _drafter_attn_sliding(state, h_norm, w, K_tt, V_tt)
    else:
        mixer = _drafter_attn_full(state, h_norm, w, K_tt, V_tt)
    ttnn.deallocate(h_norm)

    post_attn = ttnn.rms_norm(mixer, weight=w["post_attention_layernorm"],
                               epsilon=EPS)
    ttnn.deallocate(mixer)
    h_after_attn = ttnn.add(h_in, post_attn)
    ttnn.deallocate(post_attn)

    pre_ff = ttnn.rms_norm(h_after_attn, weight=w["pre_feedforward_layernorm"],
                            epsilon=EPS)
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
    post_ff = ttnn.rms_norm(mlp_out, weight=w["post_feedforward_layernorm"],
                             epsilon=EPS)
    ttnn.deallocate(mlp_out)

    # h_out = (h_after_attn + post_ff) * layer_scalar — fused via the
    # `activations` parameter on `ttnn.add` (Round-6 fork from target).
    _layer_scalar_act = [ttnn.UnaryWithParam(
        ttnn.UnaryOpType.MUL_UNARY_SFPU, float(w["layer_scalar"]))]
    h_out = ttnn.add(h_after_attn, post_ff, activations=_layer_scalar_act)
    ttnn.deallocate(h_after_attn); ttnn.deallocate(post_ff)
    return h_out


def drafter_forward(state, inputs_embeds_np, shared_kv_states):
    """Top-level drafter forward: pre_projection → 4 layers → post_projection
    + lm_head.

    Args:
      inputs_embeds_np: numpy fp32 [B=1, L=1, 2*BACKBONE_HIDDEN=7680] —
                       concat(target_h_last, target_h_prev) (HF order: prev
                       then last; see hf_oracle_gemma4_assistant.py:166).
      shared_kv_states: dict mapping layer_type → (K_np, V_np)
        - "sliding_attention": K, V shape (1, 8, L_kv, 256)
        - "full_attention":    K, V shape (1, 1, L_kv, 512)

    Returns:
      dict with numpy arrays:
        - "logits": [B, L, VOCAB=262144]
        - "hidden": [B, L, BACKBONE_HIDDEN=3840] (= post_projection output)
        - "argmax": [B, L] int64

    NOTE: this is a single-forward eager probe; v0.4 captures a trace and
    re-runs the same path. v0.2/v0.3 are correctness-first.
    """
    assert inputs_embeds_np.ndim == 3 and inputs_embeds_np.shape[2] == 2*BACKBONE_HIDDEN
    B, L, _ = inputs_embeds_np.shape
    assert B == 1 and L == 1, f"drafter forward only supports B=1, L=1; got [{B}, {L}]"

    # 1. Upload inputs_embeds replicated.
    x_tt = ttnn.from_torch(
        torch.from_numpy(inputs_embeds_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    # 2. pre_projection → [1, 1, HIDDEN].
    h = ttnn.matmul(x_tt, state.pre_projection_tt,
                    compute_kernel_config=HIFI4)
    ttnn.deallocate(x_tt)

    # 3. Upload shared KV (once for all 4 layers — all sliding layers reuse
    # the same sliding shared_kv; the single full layer uses the full one).
    K_sl_np, V_sl_np = shared_kv_states["sliding_attention"]
    K_fl_np, V_fl_np = shared_kv_states["full_attention"]
    K_sl, V_sl = _upload_kv_sliding(state, K_sl_np, V_sl_np)
    K_fl, V_fl = _upload_kv_full(state, K_fl_np, V_fl_np)

    # 4. 4 layers.
    for li in range(N_LAYERS):
        lt = state.layer_types[li]
        if lt == "sliding_attention":
            K_tt, V_tt = K_sl, V_sl
        else:
            K_tt, V_tt = K_fl, V_fl
        h_next = drafter_layer_forward(state, h, li, K_tt, V_tt)
        ttnn.deallocate(h)
        h = h_next

    ttnn.deallocate(K_sl); ttnn.deallocate(V_sl)
    ttnn.deallocate(K_fl); ttnn.deallocate(V_fl)

    # 5. final_norm (model.norm) on `h` [1, 1, HIDDEN].
    final = ttnn.rms_norm(h, weight=state.final_norm_tt, epsilon=EPS)
    ttnn.deallocate(h)

    # 6. post_projection → [1, 1, BACKBONE_HIDDEN=3840]. Replicated.
    hidden_tt = ttnn.matmul(final, state.post_projection_tt,
                             compute_kernel_config=HIFI4)
    hidden_np = _readback_replicated(hidden_tt, state.mesh).reshape(B, L, BACKBONE_HIDDEN)
    ttnn.deallocate(hidden_tt)

    # 7. lm_head: vocab-sharded matmul + all_gather. NO softcap (drafter
    # config has final_logit_softcapping=null; verified in
    # ~/.cache/huggingface/hub/.../config.json).
    sharded = ttnn.matmul(final, state.lm_head_tt, compute_kernel_config=HIFI4)
    ttnn.deallocate(final)
    gathered = ttnn.all_gather(sharded, dim=-1)
    ttnn.deallocate(sharded)
    # Untilize + argmax on device (forks target's `_lm_head_argmax`).
    rm = ttnn.untilize(gathered, use_multicore=True)
    argmax_tt = ttnn.argmax(rm, dim=-1, keepdim=True, use_multicore=False)
    logits_np = _readback_replicated(gathered, state.mesh).reshape(B, L, VOCAB)
    ttnn.deallocate(gathered)
    ttnn.deallocate(rm)
    argmax_np = ttnn.to_torch(argmax_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    # argmax replicated across chips; pick first.
    if argmax_np.shape[0] == NCHIPS:
        argmax_np = argmax_np[0]
    argmax_np = argmax_np.long().reshape(B, L).cpu().numpy()
    ttnn.deallocate(argmax_tt)
    return {
        "logits": logits_np,
        "hidden": hidden_np,
        "argmax": argmax_np,
    }


# ── CLI entrypoint: bootstrap-only smoke ────────────────────────────────
if __name__ == "__main__":
    state = State()
    bootstrap(state)
    print("[main] bootstrap returned cleanly.")
    print(f"[main] embed_tt: {state.embed_tt.shape} dtype={state.embed_tt.dtype}")
    print(f"[main] pre_projection_tt: {state.pre_projection_tt.shape}")
    print(f"[main] post_projection_tt: {state.post_projection_tt.shape}")
    print(f"[main] lm_head_tt: {state.lm_head_tt.shape}")
    print(f"[main] final_norm_tt: {state.final_norm_tt.shape}")
    print(f"[main] N_LAYERS uploaded: {len(state.per_layer_tt)}")
    for L, w in enumerate(state.per_layer_tt):
        print(f"[main]   L{L}: keys = {sorted(w.keys())}")
    ttnn.close_mesh_device(state.mesh)
    print("[main] mesh closed. v0.1 bootstrap PASS.")
