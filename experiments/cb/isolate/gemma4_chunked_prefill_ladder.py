#!/usr/bin/env python3
"""#290 P1 debug — teacher-forced per-layer + per-sub-op ladder for the
chunked-prefill correctness drift.

Ground truth: `step_forward_v031` already supports a `capture` dict that
records per-sub-op tensors at the current position. We run sequential L
times (pos 0..L-1), stack captures into an [L, ...] tensor per sub-op.

Probe: re-run the chunked forward instrumented to capture the SAME
sub-op tensors for the whole L at once. Compare cos per (position, sub-op).

The first (layer, position, sub-op) where cos < threshold IS the bug
locus. Forks the `_layer_pos0_sliding_paged` capture dict pattern
(server_gemma4_unified_ttnn.py:1496) verbatim.

Targets the layer-0 sliding path at L=4 (smallest useful L). Output
written to research/gm4_chunked_ladder_<ts>.json for inspection.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as srv  # noqa: E402

L = int(os.environ.get("GM4_CHUNKED_L", "4"))

HIDDEN = srv.HIDDEN
NCHIPS = srv.NCHIPS
EMBED_SCALE = srv.EMBED_SCALE
NQ_PER_CHIP = srv.NQ_PER_CHIP
NKV_PER_CHIP_SLIDING = srv.NKV_PER_CHIP_SLIDING
HEAD_DIM_SLIDING = srv.HEAD_DIM_SLIDING
EPS = srv.EPS
HIFI4 = srv.HIFI4

SUB_OP_KEYS = [
    "q_proj_out",     # [NCHIPS*NQ, HEAD_DIM]
    "k_proj_out",     # [NCHIPS*NKV, HEAD_DIM]
    "v_proj_out",     # [NCHIPS*NKV, HEAD_DIM]
    "q_norm_out",
    "k_norm_out",
    "v_norm_out",
    "q_rope_out",     # manual post-RoPE Q
    "k_rope_out",     # manual post-RoPE K
    "attn_out",       # post-SDPA, per-Q-head [NCHIPS*NQ, HEAD_DIM]
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── chunked layer 0 sub-op capture ──

def _capture_chunked_layer0_subops(state, token_ids):
    """Run the chunked forward at L=len(token_ids), capturing layer-0
    sub-op tensors via the same capture dict pattern. Returns dict of
    sub_op → [L, NCHIPS*nh, HD] np arrays.
    """
    Ltok = len(token_ids)
    w = state.per_layer_tt[0]

    # Embed + EMBED_SCALE.
    tok_tt = ttnn.from_torch(
        torch.tensor([list(token_ids)], dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    embed_rm = ttnn.embedding(tok_tt, state.embed_tt)
    ttnn.deallocate(tok_tt)
    embed_tile = ttnn.to_layout(embed_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(embed_rm)
    h = ttnn.multiply(embed_tile, EMBED_SCALE)  # [1, L, HIDDEN]
    ttnn.deallocate(embed_tile)

    # input_layernorm.
    h_norm = ttnn.rms_norm(h, weight=w["input_layernorm"], epsilon=EPS)
    ttnn.deallocate(h)

    # Q/K/V matmul.
    q = ttnn.matmul(h_norm, w["q_proj"], compute_kernel_config=HIFI4)
    k = ttnn.matmul(h_norm, w["k_proj"], compute_kernel_config=HIFI4)
    v = ttnn.matmul(h_norm, w["v_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(h_norm)

    # Per-head reshape: [1, L, NQ*HD] → [L, NQ, HD].
    q_h = ttnn.reshape(q, [Ltok, NQ_PER_CHIP, HEAD_DIM_SLIDING])
    ttnn.deallocate(q)
    k_h = ttnn.reshape(k, [Ltok, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING])
    ttnn.deallocate(k)
    v_h = ttnn.reshape(v, [Ltok, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING])
    ttnn.deallocate(v)

    # Capture each per-row q_h[p, :, :] as [NCHIPS*NQ, HD] using the
    # same path the existing _readback_sharded_head uses. We slice on
    # device per row, read back, then stack.
    def _readback_per_row(t, nh):
        """t shape [L, nh, HD] per chip. Slice row p, read sharded per-head."""
        rows = []
        for p in range(Ltok):
            row_slice = ttnn.slice(t, [p, 0, 0], [p + 1, nh, HEAD_DIM_SLIDING])
            # row_slice [1, nh, HD] per chip → reshape to [nh, HD] then use
            # the existing _readback_sharded_head pattern.
            row_2d = ttnn.reshape(row_slice, [nh, HEAD_DIM_SLIDING])
            arr = srv._readback_sharded_head(row_2d, state.mesh, nh,
                                              HEAD_DIM_SLIDING)
            rows.append(arr)
            # NOTE: don't deallocate row_slice (slice returns a view per
            # [[ttnn-slice-view-decay]]).
        return np.stack(rows, axis=0)  # [L, NCHIPS*nh, HD]

    captures = {}
    captures["q_proj_out"] = _readback_per_row(q_h, NQ_PER_CHIP)
    captures["k_proj_out"] = _readback_per_row(k_h, NKV_PER_CHIP_SLIDING)
    captures["v_proj_out"] = _readback_per_row(v_h, NKV_PER_CHIP_SLIDING)

    # Per-head norms.
    q_n_pre = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    k_n_pre = ttnn.rms_norm(k_h, weight=w["k_norm"], epsilon=EPS)
    v_n = ttnn.rms_norm(v_h, weight=state.ones_head_dim_sliding, epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h); ttnn.deallocate(v_h)
    captures["q_norm_out"] = _readback_per_row(q_n_pre, NQ_PER_CHIP)
    captures["k_norm_out"] = _readback_per_row(k_n_pre, NKV_PER_CHIP_SLIDING)
    captures["v_norm_out"] = _readback_per_row(v_n, NKV_PER_CHIP_SLIDING)
    # NOTE: v_n stays alive — needed for the SDPA pass below.

    # RoPE — multi-position. Forks _apply_full_rope_seq from the gate probe
    # (we don't import to keep the ladder self-contained on the cos/sin
    # contract; mirror exactly).
    pos_tt = ttnn.from_torch(
        torch.tensor([list(range(Ltok))], dtype=torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )

    def _lookup(table_tt, head_dim):
        row_rm = ttnn.embedding(pos_tt, table_tt)
        row_tile = ttnn.to_layout(row_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(row_rm)
        out = ttnn.reshape(row_tile, [Ltok, head_dim])
        return out

    cos_s = _lookup(state.cos_sliding_tt, HEAD_DIM_SLIDING)
    sin_s = _lookup(state.sin_sliding_tt, HEAD_DIM_SLIDING)
    ttnn.deallocate(pos_tt)

    def _apply_rope_seq(x_seq, n_heads, head_dim):
        half = head_dim // 2
        swapped = ttnn.roll(x_seq, shifts=half, dim=-1)
        cos_b = ttnn.reshape(cos_s, [Ltok, 1, head_dim])
        sin_b = ttnn.reshape(sin_s, [Ltok, 1, head_dim])
        x_cos = ttnn.mul(x_seq, cos_b)
        x_rope = ttnn.addcmul(x_cos, swapped, sin_b, value=1.0)
        ttnn.deallocate(x_cos); ttnn.deallocate(swapped)
        return x_rope

    q_n = _apply_rope_seq(q_n_pre, NQ_PER_CHIP, HEAD_DIM_SLIDING)
    k_n = _apply_rope_seq(k_n_pre, NKV_PER_CHIP_SLIDING, HEAD_DIM_SLIDING)
    ttnn.deallocate(q_n_pre); ttnn.deallocate(k_n_pre)
    ttnn.deallocate(cos_s); ttnn.deallocate(sin_s)

    captures["q_rope_out"] = _readback_per_row(q_n, NQ_PER_CHIP)
    captures["k_rope_out"] = _readback_per_row(k_n, NKV_PER_CHIP_SLIDING)

    # SDPA — mirror decode's per-KV-head split (2 SDPA calls, one per KV
    # head, with Q split into Q_HALF=2 groups). Decode does this at line
    # 1593 of server_gemma4_unified_ttnn.py. If the 1-call GQA SDPA on
    # NKV<NQ was the bug, mirroring decode here should make attn_out
    # cos=1.0 at every position.
    # q_n: [L, NQ, HD]. k_n / v_n: [L, NKV, HD]. NKV=2, NQ=4, Q_HALF=2.
    Q_HALF = NQ_PER_CHIP // NKV_PER_CHIP_SLIDING

    # Permute Q/K/V to head-leading so we can slice per-KV-head cleanly.
    q_perm = ttnn.permute(q_n, (1, 0, 2))  # [NQ, L, HD]
    ttnn.deallocate(q_n)
    k_perm = ttnn.permute(k_n, (1, 0, 2))  # [NKV, L, HD]
    ttnn.deallocate(k_n)
    v_perm = ttnn.permute(v_n, (1, 0, 2))
    ttnn.deallocate(v_n)

    attn_outs = []
    for kv_idx in range(NKV_PER_CHIP_SLIDING):
        # Q heads for this KV: [kv_idx*Q_HALF .. (kv_idx+1)*Q_HALF)
        q_half = ttnn.slice(q_perm, [kv_idx * Q_HALF, 0, 0],
                             [(kv_idx + 1) * Q_HALF, Ltok, HEAD_DIM_SLIDING])
        q_for = ttnn.reshape(q_half, [1, Q_HALF, Ltok, HEAD_DIM_SLIDING])
        # K, V for this KV head only.
        k_one = ttnn.slice(k_perm, [kv_idx, 0, 0],
                            [kv_idx + 1, Ltok, HEAD_DIM_SLIDING])
        k_for = ttnn.reshape(k_one, [1, 1, Ltok, HEAD_DIM_SLIDING])
        v_one = ttnn.slice(v_perm, [kv_idx, 0, 0],
                            [kv_idx + 1, Ltok, HEAD_DIM_SLIDING])
        v_for = ttnn.reshape(v_one, [1, 1, Ltok, HEAD_DIM_SLIDING])

        attn_i = ttnn.transformer.scaled_dot_product_attention(
            q_for, k_for, v_for,
            is_causal=True,
            scale=1.0,
            compute_kernel_config=state.sdpa_compute_kernel_config,
        )
        attn_outs.append(attn_i)
    ttnn.deallocate(q_perm); ttnn.deallocate(k_perm); ttnn.deallocate(v_perm)

    # Concat along Q-head axis (dim 1): [1, Q_HALF, L, HD] x NKV → [1, NQ, L, HD]
    attn_out = ttnn.concat(attn_outs, dim=1)
    for a in attn_outs:
        ttnn.deallocate(a)

    arr = ttnn.to_torch(attn_out, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    ttnn.deallocate(attn_out)
    arr = arr.float().cpu().numpy()                # [NCHIPS, NQ_PER_CHIP, L, HD]
    arr = arr.transpose(0, 2, 1, 3)                # [NCHIPS, L, NQ_PER_CHIP, HD]
    captures["attn_out"] = arr.transpose(1, 0, 2, 3).reshape(
        Ltok, NCHIPS * NQ_PER_CHIP, HEAD_DIM_SLIDING)

    return captures


def main():
    log("=" * 72)
    log(f"#290 P1 ladder — layer 0 sub-op cos vs sequential at L={L}")
    log("=" * 72)

    log("STAGE 1: bootstrap…")
    state = srv.State()
    srv.bootstrap(state, log=log)

    from gemma4_long_context_argmax_gate import build_prompt
    token_ids = build_prompt(state.tokenizer, target_len=L)
    log(f"  prompt: {len(token_ids)} tokens, first 6 = {token_ids[:6]}")

    log("STAGE 2: sequential ground truth — capture per-pos sub-ops")
    # _layer_pos0_sliding_paged accepts a capture dict but needs h_norm as
    # input. We build h_norm per position manually (embed → EMBED_SCALE →
    # input_layernorm), set_pos, then invoke the layer helper with capture.
    # The cache write inside the helper IS desirable here — that's the
    # ground-truth path (mirrors what step_forward_v031 does internally).
    w0 = state.per_layer_tt[0]
    seq_caps_per_pos = []
    for pos, tok in enumerate(token_ids):
        srv._set_pos(state, pos)
        tok_tt = ttnn.from_torch(
            torch.tensor([[int(tok)]], dtype=torch.int32),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        embed = ttnn.embedding(tok_tt, state.embed_tt)
        ttnn.deallocate(tok_tt)
        h = ttnn.multiply(ttnn.to_layout(embed, ttnn.TILE_LAYOUT), EMBED_SCALE)
        ttnn.deallocate(embed)
        h_norm = ttnn.rms_norm(h, weight=w0["input_layernorm"], epsilon=EPS)
        ttnn.deallocate(h)
        cap = {}
        _mixer = srv._layer_pos0_sliding_paged(state, h_norm, w0, 0, capture=cap)
        ttnn.deallocate(h_norm); ttnn.deallocate(_mixer)
        seq_caps_per_pos.append(cap)
        log(f"  pos {pos}: captured keys = {sorted(cap.keys())}")

    log("STAGE 3: chunked layer-0 sub-op capture")
    chunked_caps = _capture_chunked_layer0_subops(state, token_ids)
    for k, arr in chunked_caps.items():
        log(f"  chunked['{k}']: shape={arr.shape}")

    log("STAGE 4: per-(position, sub-op) cosine ladder")
    rows = []
    for sub_op in SUB_OP_KEYS:
        for pos in range(L):
            seq_v = seq_caps_per_pos[pos].get(sub_op)
            if seq_v is None:
                continue
            chunked_v = chunked_caps[sub_op][pos]
            c = cosine(seq_v, chunked_v)
            rows.append((sub_op, pos, c, seq_v.shape, chunked_v.shape))
            log(f"  [{sub_op:15s}] pos {pos}: cos = {c:.6f}  "
                f"(seq {seq_v.shape}, chunked {chunked_v.shape})")

    out_dir = PROJECT_ROOT / "research"
    out_path = out_dir / f"gm4_chunked_ladder_{int(time.time())}.json"
    out_path.write_text(json.dumps(
        [{"sub_op": s, "pos": p, "cos": c} for s, p, c, _, _ in rows],
        indent=2,
    ))
    log(f"saved ladder → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
