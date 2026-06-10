#!/usr/bin/env python3
"""#290 Phase 1 — chunked prefill for Gemma 4 at L=128.

Replaces N × `step_forward_v031` (~47 ms/tok sequential) with one parallel
forward over L=128 tokens. Goal at this phase: cos >= 0.999 vs sequential
ground truth + last-position argmax match + >=2x TTFT win.

Reuse map:
- `experiments/serve/server_tp.py:forward_prefill_chunked_tp` (line 1945)
  — 27B chunked prefill outer structure (embed all L → per-pos RoPE →
  parallel SDPA over L → MLP leading-dim agnostic → last-pos LM head).
- `experiments/serve/server_tp.py:gated_attn_step_prefill_tp` (line 1702)
  — 27B parallel SDPA layer: is_causal=True, ttnn.transformer.scaled_dot_product_attention,
  K/V cache via paged_fill_cache.
- `server_gemma4_unified_ttnn._layer_pos0_sliding_paged` /
  `_layer_pos0_global_paged` — Gemma 4 sub-op contracts (Q/K/V norm
  + v_norm + RoPE + per-head SDPA + o_proj + all_reduce).
- `server_gemma4_unified_ttnn._apply_full_rope` — single-position RoPE
  (rotate-half via roll). Forked into _apply_full_rope_seq below for
  multi-position with [L, n_heads, head_dim] x [L, head_dim] broadcast.

P1 SCOPE
- SKIP K/V cache writes (correctness gate validates the forward math only;
  cache write + handoff is P1.6.5, after the gate is green).
- IGNORE sliding-window mask — at L=128 every position sees all prior
  tokens within SLIDING_WINDOW=1024, so causal == sliding-causal. P2
  scales L > 1024 and adds the mask back.

Run (default L=128):
  scripts/run_remote.sh experiments/cb/isolate/gemma4_chunked_prefill_L128.py
Or with a smaller L (bring-up debug):
  scripts/run_remote.sh experiments/cb/isolate/gemma4_chunked_prefill_L128.py 4
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as srv  # noqa: E402

L = int(sys.argv[1]) if len(sys.argv) > 1 else 128

HIDDEN = srv.HIDDEN
EMBED_SCALE = srv.EMBED_SCALE
NQ_PER_CHIP = srv.NQ_PER_CHIP
NKV_PER_CHIP_SLIDING = srv.NKV_PER_CHIP_SLIDING
NUM_KV_HEADS_GLOBAL = srv.NUM_KV_HEADS_GLOBAL
HEAD_DIM_SLIDING = srv.HEAD_DIM_SLIDING
HEAD_DIM_GLOBAL = srv.HEAD_DIM_GLOBAL
EPS = srv.EPS
HIFI4 = srv.HIFI4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── P1.1: multi-token embed + RoPE table lookup over positions [0, L) ──

def _embed_and_lookup_rope_seq(state, token_ids):
    """Embed L tokens → [1, L, HIDDEN] TILE bf16 (rank-3 throughout, matches
    decode's `[1, 1, HIDDEN]` activation rank); lookup cos/sin tables at
    positions [0..L-1] → 4-tuple ([L, HD_SLIDING], same, [L, HD_GLOBAL], same).

    Caller MUST deallocate h and all four rope tensors after the forward.
    """
    Ltok = len(token_ids)

    tok_tt = ttnn.from_torch(
        torch.tensor([list(token_ids)], dtype=torch.int32),  # [1, L]
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    embed_rm = ttnn.embedding(tok_tt, state.embed_tt)  # [1, L, HIDDEN] row-major
    ttnn.deallocate(tok_tt)
    embed_tile = ttnn.to_layout(embed_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(embed_rm)
    h = ttnn.multiply(embed_tile, EMBED_SCALE)  # [1, L, HIDDEN] rank-3
    ttnn.deallocate(embed_tile)

    # Multi-position RoPE table lookup for sliding + global.
    pos_tt = ttnn.from_torch(
        torch.tensor([list(range(Ltok))], dtype=torch.int32),  # [1, L]
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )

    def _lookup(table_tt, head_dim):
        row_rm = ttnn.embedding(pos_tt, table_tt)  # [1, L, head_dim] row-major
        row_tile = ttnn.to_layout(row_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(row_rm)
        out = ttnn.reshape(row_tile, [Ltok, head_dim])
        return out

    cos_s = _lookup(state.cos_sliding_tt, HEAD_DIM_SLIDING)
    sin_s = _lookup(state.sin_sliding_tt, HEAD_DIM_SLIDING)
    cos_g = _lookup(state.cos_global_tt, HEAD_DIM_GLOBAL)
    sin_g = _lookup(state.sin_global_tt, HEAD_DIM_GLOBAL)
    ttnn.deallocate(pos_tt)
    return h, (cos_s, sin_s, cos_g, sin_g)


def _release_rope_seq(rope_seq):
    for t in rope_seq:
        ttnn.deallocate(t)


# ── P1.2 helper: multi-token RoPE over [L, n_heads, head_dim] ──

def _apply_full_rope_seq(x_seq, cos_seq, sin_seq, Ltok, n_heads, head_dim):
    """Multi-token RoPE. x_seq shape [L, n_heads, head_dim],
    cos_seq/sin_seq shape [L, head_dim] → out [L, n_heads, head_dim].

    Forks `_apply_full_rope` (single position [n_heads, head_dim]) with a
    [L, 1, head_dim] broadcast over the n_heads axis. Roll is per-position
    (acts on last dim), so the rotate-half optimization survives the
    leading-L dim addition.
    """
    half = head_dim // 2
    swapped = ttnn.roll(x_seq, shifts=half, dim=-1)
    cos_b = ttnn.reshape(cos_seq, [Ltok, 1, head_dim])
    sin_b = ttnn.reshape(sin_seq, [Ltok, 1, head_dim])
    x_cos = ttnn.mul(x_seq, cos_b)
    x_rope = ttnn.addcmul(x_cos, swapped, sin_b, value=1.0)
    ttnn.deallocate(x_cos); ttnn.deallocate(swapped)
    return x_rope


# ── P1.2: _layer_prefill_sliding — causal SDPA over q_len=L ──

def _layer_prefill_sliding(state, h_norm_seq, w, layer_idx, rope, Ltok):
    """Sliding-attention prefill. Forks `_layer_pos0_sliding_paged`:
    - h_norm_seq [L, HIDDEN] (vs [1, HIDDEN])
    - matmul outputs reshape with leading L
    - Q/K/V norms broadcast over [L, n_heads] via rank-3 [L, nh, hd]
      (rms_norm last-dim contract — avoids [[feedback-ttnn-rms-norm-shape-drift]])
    - RoPE batched via _apply_full_rope_seq
    - Causal SDPA via ttnn.transformer.scaled_dot_product_attention with
      Q [1, NQ_PER_CHIP, L, HD] and K/V [1, NKV_PER_CHIP_SLIDING, L, HD]
      — SDPA handles GQA natively per [[reference-ttnn-sdpa-gqa-native]].
    - P1: SKIP cache writes (validates forward math; cache+handoff = P1.6.5).
    - P1: IGNORE sliding window — at L=128 << SLIDING_WINDOW=1024, causal == sliding.
    """
    cos_seq, sin_seq = rope

    q = ttnn.matmul(h_norm_seq, w["q_proj"], compute_kernel_config=HIFI4)
    k = ttnn.matmul(h_norm_seq, w["k_proj"], compute_kernel_config=HIFI4)
    v = ttnn.matmul(h_norm_seq, w["v_proj"], compute_kernel_config=HIFI4)

    q_h = ttnn.reshape(q, [Ltok, NQ_PER_CHIP, HEAD_DIM_SLIDING])
    ttnn.deallocate(q)
    k_h = ttnn.reshape(k, [Ltok, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING])
    ttnn.deallocate(k)
    v_h = ttnn.reshape(v, [Ltok, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING])
    ttnn.deallocate(v)

    q_n_pre = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    k_n_pre = ttnn.rms_norm(k_h, weight=w["k_norm"], epsilon=EPS)
    v_n = ttnn.rms_norm(v_h, weight=state.ones_head_dim_sliding, epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h); ttnn.deallocate(v_h)

    q_n = _apply_full_rope_seq(q_n_pre, cos_seq, sin_seq, Ltok,
                                NQ_PER_CHIP, HEAD_DIM_SLIDING)
    ttnn.deallocate(q_n_pre)
    k_n = _apply_full_rope_seq(k_n_pre, cos_seq, sin_seq, Ltok,
                                NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING)
    ttnn.deallocate(k_n_pre)

    # Reshape for SDPA: [L, nh, hd] → [1, nh, L, hd] via permute + reshape.
    q_t = ttnn.permute(q_n, (1, 0, 2))   # [NQ, L, HD]
    ttnn.deallocate(q_n)
    q_for_sdpa = ttnn.reshape(q_t, [1, NQ_PER_CHIP, Ltok, HEAD_DIM_SLIDING])
    ttnn.deallocate(q_t)
    k_t = ttnn.permute(k_n, (1, 0, 2))   # [NKV, L, HD]
    ttnn.deallocate(k_n)
    k_for_sdpa = ttnn.reshape(k_t, [1, NKV_PER_CHIP_SLIDING, Ltok, HEAD_DIM_SLIDING])
    ttnn.deallocate(k_t)
    v_t = ttnn.permute(v_n, (1, 0, 2))
    ttnn.deallocate(v_n)
    v_for_sdpa = ttnn.reshape(v_t, [1, NKV_PER_CHIP_SLIDING, Ltok, HEAD_DIM_SLIDING])
    ttnn.deallocate(v_t)

    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q_for_sdpa, k_for_sdpa, v_for_sdpa,
        is_causal=True,
        scale=1.0,  # Gemma 4: self.scaling=1.0
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )
    ttnn.deallocate(q_for_sdpa); ttnn.deallocate(k_for_sdpa); ttnn.deallocate(v_for_sdpa)

    # attn_out [1, NQ, L, HD] → [L, NQ*HD] for o_proj.
    attn_perm = ttnn.permute(attn_out, (0, 2, 1, 3))  # [1, L, NQ, HD]
    ttnn.deallocate(attn_out)
    attn_flat = ttnn.reshape(attn_perm, [Ltok, NQ_PER_CHIP * HEAD_DIM_SLIDING])
    ttnn.deallocate(attn_perm)

    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    out = srv.all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


# ── P1.3: _layer_prefill_global — head_dim=512, p-RoPE inline, V=K_raw ──

def _layer_prefill_global(state, h_norm_seq, w, layer_idx, rope, Ltok):
    """Global-attention prefill. Forks `_layer_pos0_global_paged`:
    - V aliases K_raw (pre-norm), per attention_k_eq_v
    - NUM_KV_HEADS_GLOBAL=1 (per-chip replicated)
    - HEAD_DIM_GLOBAL=512
    - p-RoPE encoded INLINE in cos/sin global tables (last 384 dims have
      inv_freq=0 → cos=1/sin=0 act as identity).
    - SKIP cache writes (P1).
    """
    cos_seq, sin_seq = rope

    q = ttnn.matmul(h_norm_seq, w["q_proj"], compute_kernel_config=HIFI4)
    k = ttnn.matmul(h_norm_seq, w["k_proj"], compute_kernel_config=HIFI4)

    q_h = ttnn.reshape(q, [Ltok, NQ_PER_CHIP, HEAD_DIM_GLOBAL])
    ttnn.deallocate(q)
    k_h = ttnn.reshape(k, [Ltok, NUM_KV_HEADS_GLOBAL, HEAD_DIM_GLOBAL])
    ttnn.deallocate(k)
    v_raw = ttnn.clone(k_h)  # V aliases K_raw pre-norm

    q_n_pre = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    k_n_pre = ttnn.rms_norm(k_h, weight=w["k_norm"], epsilon=EPS)
    v_n = ttnn.rms_norm(v_raw, weight=state.ones_head_dim_global, epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h); ttnn.deallocate(v_raw)

    q_n = _apply_full_rope_seq(q_n_pre, cos_seq, sin_seq, Ltok,
                                NQ_PER_CHIP, HEAD_DIM_GLOBAL)
    ttnn.deallocate(q_n_pre)
    k_n = _apply_full_rope_seq(k_n_pre, cos_seq, sin_seq, Ltok,
                                NUM_KV_HEADS_GLOBAL, HEAD_DIM_GLOBAL)
    ttnn.deallocate(k_n_pre)

    q_t = ttnn.permute(q_n, (1, 0, 2))   # [NQ, L, HD]
    ttnn.deallocate(q_n)
    q_for_sdpa = ttnn.reshape(q_t, [1, NQ_PER_CHIP, Ltok, HEAD_DIM_GLOBAL])
    ttnn.deallocate(q_t)
    k_t = ttnn.permute(k_n, (1, 0, 2))
    ttnn.deallocate(k_n)
    k_for_sdpa = ttnn.reshape(k_t, [1, NUM_KV_HEADS_GLOBAL, Ltok, HEAD_DIM_GLOBAL])
    ttnn.deallocate(k_t)
    v_t = ttnn.permute(v_n, (1, 0, 2))
    ttnn.deallocate(v_n)
    v_for_sdpa = ttnn.reshape(v_t, [1, NUM_KV_HEADS_GLOBAL, Ltok, HEAD_DIM_GLOBAL])
    ttnn.deallocate(v_t)

    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q_for_sdpa, k_for_sdpa, v_for_sdpa,
        is_causal=True,
        scale=1.0,
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )
    ttnn.deallocate(q_for_sdpa); ttnn.deallocate(k_for_sdpa); ttnn.deallocate(v_for_sdpa)

    attn_perm = ttnn.permute(attn_out, (0, 2, 1, 3))  # [1, L, NQ, HD]
    ttnn.deallocate(attn_out)
    attn_flat = ttnn.reshape(attn_perm, [Ltok, NQ_PER_CHIP * HEAD_DIM_GLOBAL])
    ttnn.deallocate(attn_perm)

    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_flat)
    out = srv.all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


# ── P1.4: _layer_prefill — full layer orchestrator (attn + MLP block) ──

def _layer_prefill(state, h_in, layer_idx, rope_seq, Ltok):
    """Full layer (sandwich norm + attn + MLP). Forks
    `_layer_forward_pos0_paged` minus DRAM-sharded MLP path (uses simpler
    matmul-gelu fused default path — leading-dim agnostic per
    `gated_attn_step_prefill_tp` precedent).
    """
    w = state.per_layer_tt[layer_idx]
    lt = state.layer_types[layer_idx]

    h_norm = ttnn.rms_norm(h_in, weight=w["input_layernorm"], epsilon=EPS)
    if lt == "sliding_attention":
        mixer = _layer_prefill_sliding(state, h_norm, w, layer_idx,
                                        (rope_seq[0], rope_seq[1]), Ltok)
    else:
        mixer = _layer_prefill_global(state, h_norm, w, layer_idx,
                                       (rope_seq[2], rope_seq[3]), Ltok)
    ttnn.deallocate(h_norm)

    post_attn = ttnn.rms_norm(mixer, weight=w["post_attention_layernorm"],
                                epsilon=EPS)
    ttnn.deallocate(mixer)
    h_after_attn = ttnn.add(h_in, post_attn)
    ttnn.deallocate(post_attn)

    pre_ff = ttnn.rms_norm(h_after_attn, weight=w["pre_feedforward_layernorm"],
                            epsilon=EPS)
    # MLP — leading-dim agnostic per 27B `gated_attn_step_prefill_tp` precedent.
    gelu_gate = ttnn.matmul(pre_ff, w["gate_proj"],
                             compute_kernel_config=HIFI4,
                             activation="gelu")
    up = ttnn.matmul(pre_ff, w["up_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(pre_ff)
    mid = ttnn.mul(gelu_gate, up)
    ttnn.deallocate(gelu_gate); ttnn.deallocate(up)
    mlp_partial = ttnn.matmul(mid, w["down_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(mid)
    mlp_out = srv.all_reduce_tt(mlp_partial, state.mesh)
    ttnn.deallocate(mlp_partial)

    post_ff = ttnn.rms_norm(mlp_out, weight=w["post_feedforward_layernorm"],
                              epsilon=EPS)
    ttnn.deallocate(mlp_out)

    # layer_scalar fused into the final residual add via SFPU.
    _layer_scalar_act = [ttnn.UnaryWithParam(
        ttnn.UnaryOpType.MUL_UNARY_SFPU, float(w["layer_scalar"]))]
    h_out = ttnn.add(h_after_attn, post_ff, activations=_layer_scalar_act)
    ttnn.deallocate(h_after_attn); ttnn.deallocate(post_ff)
    return h_out


# ── P1.5: step_forward_prefill orchestrator ──

def step_forward_prefill(state, token_ids, capture_hidden=False):
    """Multi-token parallel prefill. Returns:
      (last_argmax: int, last_hidden_np: np.ndarray | None)

    Last hidden is post-final-norm at position L-1 (matches what
    step_forward_v03 stashes in state.last_target_hidden_cur).
    """
    Ltok = len(token_ids)
    NUM_LAYERS = srv.NUM_LAYERS

    h, rope_seq = _embed_and_lookup_rope_seq(state, token_ids)

    for layer_idx in range(NUM_LAYERS):
        h_new = _layer_prefill(state, h, layer_idx, rope_seq, Ltok)
        ttnn.deallocate(h)
        h = h_new

    _release_rope_seq(rope_seq)

    final = ttnn.rms_norm(h, weight=state.final_norm_tt, epsilon=EPS)
    ttnn.deallocate(h)

    # Slice last position on device — final is rank-3 [1, L, HIDDEN] (rms_norm
    # preserves rank; reshape upstream is metadata-only).
    last_row = ttnn.slice(final, [0, Ltok - 1, 0], [1, Ltok, HIDDEN])
    ttnn.deallocate(final)

    last_hidden_np = None
    if capture_hidden:
        last_hidden_np = srv._readback_replicated(last_row, state.mesh).astype(
            np.float32).reshape(1, 1, HIDDEN)

    argmax_tt, _ = srv._lm_head_argmax(state, last_row, capture_logits=False)
    ttnn.deallocate(last_row)
    arr = ttnn.to_torch(argmax_tt,
                        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    ttnn.deallocate(argmax_tt)
    argmax = int(arr.reshape(-1)[0].item())
    return argmax, last_hidden_np


# ── ground-truth baseline + gate driver ──

def baseline_sequential(state, token_ids):
    """L × step_forward_v031 in order. Returns (last_argmax, wall_s, last_hidden)."""
    t0 = time.time()
    last_argmax = None
    for pos, tok in enumerate(token_ids):
        last_argmax = srv.step_forward_v031(state, tok_id=int(tok), pos=pos)
    elapsed = time.time() - t0
    return int(last_argmax), elapsed, state.last_target_hidden_cur


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    log("=" * 72)
    log(f"#290 P1 — chunked prefill at L={L}")
    log("=" * 72)

    log("STAGE 1: bootstrap target (~90s)…")
    state = srv.State()
    srv.bootstrap(state, log=log)

    log(f"STAGE 2: build {L}-token prompt (reuse seed paragraph)")
    # Use the same SEED_PARAGRAPH from the long-context gate so we're
    # gating on the same input distribution we've already validated against.
    from gemma4_long_context_argmax_gate import build_prompt
    token_ids = build_prompt(state.tokenizer, target_len=L)
    assert len(token_ids) == L, f"build_prompt returned {len(token_ids)} != {L}"
    log(f"  prompt: {len(token_ids)} tokens, first 6 = {token_ids[:6]}")

    log("STAGE 3: baseline sequential prefill (ground truth)…")
    base_argmax, base_wall, base_hidden = baseline_sequential(state, token_ids)
    log(f"  baseline last-argmax = {base_argmax}, wall = {base_wall:.1f}s")
    log(f"  baseline hidden shape = {base_hidden.shape if base_hidden is not None else 'None'}")

    log("STAGE 4: chunked prefill…")
    # Reset the cache before the chunked path — it was written by the
    # sequential baseline. We don't write the cache in the chunked path
    # (P1 scope), but resetting prevents the readback in the LM head from
    # accidentally consuming stale state. The reset is also a stress
    # check for the bootstrap-time cache allocation.
    # Actually — the cache write happens in _layer_pos0_*, NOT in the
    # forward output. final / argmax depend only on the forward math.
    # So the cache state doesn't affect this gate.
    t0 = time.time()
    chunk_argmax, chunk_hidden = step_forward_prefill(state, token_ids,
                                                     capture_hidden=True)
    chunk_wall = time.time() - t0
    log(f"  chunked last-argmax = {chunk_argmax}, wall = {chunk_wall:.1f}s")

    log("STAGE 5: gates")
    pass_argmax = (chunk_argmax == base_argmax)
    log(f"  GATE C (argmax): {'PASS' if pass_argmax else 'FAIL'} "
        f"(chunk={chunk_argmax} base={base_argmax})")

    cos_score = cosine(base_hidden, chunk_hidden)
    pass_cos = cos_score >= 0.999
    log(f"  GATE A (cos): {'PASS' if pass_cos else 'FAIL'} "
        f"(cos={cos_score:.6f}, threshold=0.999)")

    speedup = base_wall / chunk_wall if chunk_wall > 0 else float("inf")
    pass_perf = speedup >= 2.0
    log(f"  GATE B (TTFT): {'PASS' if pass_perf else 'FAIL'} "
        f"(speedup={speedup:.2f}x, threshold=2.0x)")

    log("=" * 72)
    all_pass = pass_argmax and pass_cos and pass_perf
    log(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    log("=" * 72)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
