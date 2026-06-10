#!/usr/bin/env python3
"""#290 P4 trace root-cause — ladder probe.

The trace capture infrastructure (commit 856c1aa, rolled back in 06f9437)
produced garbled first tokens at HTTP request time. This probe localises
the divergence WITHOUT capturing a trace — by re-implementing the
"traced-friendly" inner forward eagerly and comparing per-stage against
the proven-correct `forward_prefill_chunked_tp` eager path.

  PATH A (ground truth):  forward_prefill_chunked_tp(token_ids)
                          slice last row → _lm_head_argmax (softcap + AR
                          + on-device argmax)
                          → argmax_A (int)

  PATH B (graph-only):    re-implement the matmul-over-all-L lm_head
                          (no softcap, host-side argmax on row L-1)
                          eagerly. NO trace capture.
                          → argmax_B (int)

If argmax_A == argmax_B → the alternative computation graph is correct;
the trace capture/replay path is to blame. Likely culprits then:
  • two-phase warmup (decode trace + prefill trace memory collision)
  • `state.prefill_tok_buf` not written before the captured replay
  • view-vs-storage of `final_2d = ttnn.reshape(final, [L, HIDDEN])`
    being invalidated by the surrounding deallocate

If argmax_A != argmax_B → the alternative graph itself is wrong.
Localise further by comparing intermediate tensors (final hidden,
gathered logits at row L-1).

Plus a hidden-state ladder: compare `final` after `rms_norm` at row L-1
across the two paths to catch divergence pre-lm_head.

Run on qb1 (per remote-only contract). Single bootstrap (~6 min) then
fast iteration via env-flag.

    ssh qb1 'cd ~/tt-xla && \\
        TT_GM4_TRACE_LADDER_L=128 \\
        .venv/bin/python -u \\
            experiments/cb/isolate/gemma4_chunked_prefill_trace_ladder.py'

REUSE: forks the bootstrap pattern from
`experiments/cb/isolate/gemma4_chunked_prefill_L128.py` (gate probe).
Forks the capture-comparison pattern from
`experiments/cb/isolate/gemma4_chunked_prefill_ladder.py` (P1 SDPA bug
finder).
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

L = int(os.environ.get("TT_GM4_TRACE_LADDER_L", "128"))
NUM_LAYERS = srv.NUM_LAYERS
HIDDEN = srv.HIDDEN
HEAD_DIM_SLIDING = srv.HEAD_DIM_SLIDING
HEAD_DIM_GLOBAL = srv.HEAD_DIM_GLOBAL
VOCAB = getattr(srv, "VOCAB", 262144)
EPS = srv.EPS
EMBED_SCALE = srv.EMBED_SCALE
HIFI4 = srv.HIFI4


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _readback(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).numpy()


# ── Path A: proven eager forward ──────────────────────────────────────

def _path_a_argmax_and_final(state, token_ids):
    """Run `forward_prefill_chunked_tp` instrumented to ALSO stash the
    pre-lm_head hidden state at row L-1, so we can compare to Path B's
    same row.

    We can't call srv.forward_prefill_chunked_tp directly because it
    deallocates `final` before we get a chance to read row L-1. Re-do
    the forward inline so we can capture the final tensor.
    """
    Ltok = len(token_ids)
    h, rope_seq = srv._embed_and_lookup_rope_seq(state, token_ids)
    for li in range(NUM_LAYERS):
        h_new = srv._layer_prefill(state, h, li, rope_seq, Ltok)
        ttnn.deallocate(h); h = h_new
    for t in rope_seq:
        ttnn.deallocate(t)

    final = ttnn.rms_norm(h, weight=state.final_norm_tt, epsilon=EPS)
    ttnn.deallocate(h)
    # Stash row L-1 of final BEFORE the lm_head slice eats it.
    last_row = ttnn.slice(final, [0, Ltok - 1, 0], [1, Ltok, HIDDEN])
    final_lastrow_np = _readback(last_row, state.mesh).astype(
        np.float32).reshape(HIDDEN)

    argmax_tt, _ = srv._lm_head_argmax(state, last_row, capture_logits=False)
    arr = ttnn.to_torch(argmax_tt,
                        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    ttnn.deallocate(argmax_tt)
    argmax = int(arr.reshape(-1)[0].item())
    return argmax, final_lastrow_np


# ── Path B: traced-style graph, run EAGERLY (no capture) ──────────────

def _path_b_argmax_and_final(state, token_ids):
    """Re-implements `forward_prefill_chunked_traced_inner` (the rolled-
    back P4 attempt) eagerly. Reads pos+tok from STATE BUFFERS to match
    the trace's input contract — but executes immediately, no
    begin_trace_capture.

    Two key differences from Path A's lm_head:
      • matmul runs on [L, HIDDEN] (all rows), NOT just last row
      • NO softcap (argmax-invariant — softcap is monotonic)
      • host-side argmax on row L-1 of the gathered logits

    If this matches Path A's argmax → the alternative graph is fine,
    the trace capture/replay is the bug.
    """
    Ltok = len(token_ids)
    # Write to pre-allocated buffers — same path the trace would have.
    # Lazy-allocate if the bootstrap didn't pre-allocate (current main
    # HEAD doesn't have prefill_tok_buf/prefill_pos_buf).
    if getattr(state, "prefill_tok_buf", None) is None:
        log("  lazy-allocating state.prefill_tok_buf / prefill_pos_buf "
            f"at chunk_size={Ltok}")
        state.prefill_chunk_size = Ltok
        state.prefill_tok_buf = ttnn.from_torch(
            torch.zeros((1, Ltok), dtype=torch.int32),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        state.prefill_pos_buf = ttnn.from_torch(
            torch.zeros((1, Ltok), dtype=torch.int32),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )

    # Write tok + pos via copy_host_to_device_tensor (matches trace path).
    tok_host = ttnn.from_torch(
        torch.tensor([list(token_ids)], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    ttnn.copy_host_to_device_tensor(tok_host, state.prefill_tok_buf)
    pos_host = ttnn.from_torch(
        torch.tensor([list(range(Ltok))], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    ttnn.copy_host_to_device_tensor(pos_host, state.prefill_pos_buf)

    # Embed from device buffer (no from_torch INSIDE the forward).
    embed_rm = ttnn.embedding(state.prefill_tok_buf, state.embed_tt)
    embed_tile = ttnn.to_layout(embed_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(embed_rm)
    h = ttnn.multiply(embed_tile, EMBED_SCALE)
    ttnn.deallocate(embed_tile)

    def _lookup(table_tt, head_dim):
        row_rm = ttnn.embedding(state.prefill_pos_buf, table_tt)
        row_tile = ttnn.to_layout(row_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(row_rm)
        return ttnn.reshape(row_tile, [Ltok, head_dim])

    cos_s = _lookup(state.cos_sliding_tt, HEAD_DIM_SLIDING)
    sin_s = _lookup(state.sin_sliding_tt, HEAD_DIM_SLIDING)
    cos_g = _lookup(state.cos_global_tt, HEAD_DIM_GLOBAL)
    sin_g = _lookup(state.sin_global_tt, HEAD_DIM_GLOBAL)
    rope_seq = (cos_s, sin_s, cos_g, sin_g)

    for li in range(NUM_LAYERS):
        h_new = srv._layer_prefill(state, h, li, rope_seq, Ltok)
        ttnn.deallocate(h); h = h_new
    for t in rope_seq:
        ttnn.deallocate(t)

    final = ttnn.rms_norm(h, weight=state.final_norm_tt, epsilon=EPS)
    ttnn.deallocate(h)

    # Read row L-1 of final FOR COMPARISON (separate from lm_head path).
    last_row_view = ttnn.slice(final, [0, Ltok - 1, 0], [1, Ltok, HIDDEN])
    final_lastrow_np = _readback(last_row_view, state.mesh).astype(
        np.float32).reshape(HIDDEN)
    ttnn.deallocate(last_row_view)

    # Traced-style lm_head: matmul over ALL L rows, no softcap.
    final_2d = ttnn.reshape(final, [Ltok, HIDDEN])
    sharded = ttnn.matmul(final_2d, state.lm_head_tt,
                          compute_kernel_config=HIFI4)
    ttnn.deallocate(final)  # final_2d is a view of final, dies with it
    gathered = ttnn.all_gather(sharded, dim=-1)
    ttnn.deallocate(sharded)
    # gshape may be [L, VOCAB] or [1, L, VOCAB] depending on path; build
    # slice indices off shape.
    gshape = list(gathered.shape)
    begins = [0] * len(gshape)
    ends = list(gshape)
    ends[-1] = VOCAB
    sliced = ttnn.slice(gathered, begins, ends)
    out = ttnn.untilize(sliced, use_multicore=True)
    # Read the [L, VOCAB] float32 tensor and take host-side argmax row L-1.
    full_np = _readback(out, state.mesh)
    ttnn.deallocate(gathered)  # owns sliced/out view chain
    full_np = full_np.astype(np.float32).reshape(-1, VOCAB)
    argmax_B = int(np.argmax(full_np[Ltok - 1]))
    return argmax_B, final_lastrow_np, full_np


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    log(f"trace-ladder L={L}")
    state = type("S", (), {})()
    srv.bootstrap(state, log=log)
    log("bootstrap done")

    # Deterministic prompt.
    tok = srv._build_tokenizer(state) if hasattr(srv, "_build_tokenizer") else None
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(srv.MODEL_ID)
    prompt = ("The history of computing spans many centuries, from "
              "abacuses to the first mechanical calculators, through "
              "vacuum tubes, transistors, and ultimately to today's ")
    ids = tok.encode(prompt, add_special_tokens=False)[:L]
    # Pad to L with the last token (deterministic).
    while len(ids) < L:
        ids.append(ids[-1])
    log(f"prompt L={len(ids)} ids[:8]={ids[:8]}")

    log("running Path A (eager known-good)…")
    t0 = time.time()
    argmax_A, final_A = _path_a_argmax_and_final(state, ids)
    log(f"  Path A done in {time.time() - t0:.1f}s  argmax={argmax_A}  "
        f"|final[L-1]|₂={np.linalg.norm(final_A):.4f}")

    log("running Path B (traced-style graph, eager exec)…")
    t0 = time.time()
    argmax_B, final_B, full_np = _path_b_argmax_and_final(state, ids)
    log(f"  Path B done in {time.time() - t0:.1f}s  argmax={argmax_B}  "
        f"|final[L-1]|₂={np.linalg.norm(final_B):.4f}")

    cos_final = cosine(final_A, final_B)
    print()
    print("══ Results " + "═" * 60)
    print(f"  argmax_A (eager known-good)       = {argmax_A}")
    print(f"  argmax_B (traced-style graph)     = {argmax_B}")
    print(f"  argmax match: {argmax_A == argmax_B}")
    print(f"  cos(final[L-1]_A, final[L-1]_B)   = {cos_final:.6f}")
    print()

    verdict = "UNKNOWN"
    diagnosis = "(no diagnosis derived)"
    if argmax_A == argmax_B and cos_final >= 0.9999:
        verdict = "GRAPH OK — bug is in trace capture/replay"
        diagnosis = (
            "Re-enable P4 with attention on:\n"
            "    1. two-phase warmup vs decode trace memory (TT_CB_USE_TRACE\n"
            "       interaction).\n"
            "    2. state.prefill_tok_buf write happening BEFORE the\n"
            "       captured replay (cb_scheduler must call\n"
            "       update_prefill_input_buffers before execute_trace).\n"
            "    3. final_2d = reshape(final, [L, HIDDEN]) being a view —\n"
            "       it must outlive the matmul's read."
        )
    elif argmax_A != argmax_B and cos_final >= 0.9999:
        verdict = "LM_HEAD-only divergence — softcap or row-indexing"
        diagnosis = (
            "Hidden state matches at row L-1, but argmax doesn't. The\n"
            "lm_head path differs. Suspects:\n"
            "    1. matmul-over-all-L row L-1 ≠ matmul-on-single-row\n"
            "       (sharding / all_gather layout difference).\n"
            "    2. NO softcap on Path B — should be argmax-invariant.\n"
            "       If not, softcap is non-monotonic in our impl.\n"
            "    3. host-side argmax on bf16-decoded float32 vs\n"
            "       on-device argmax tie-break (multicore=True race)."
        )
    else:
        verdict = "PRE-LM_HEAD divergence — graph itself is wrong"
        diagnosis = (
            "Hidden state diverges before lm_head. Path B's embed+rope+\n"
            "layer chain differs from Path A even though both run eagerly.\n"
            "Suspects:\n"
            "    1. prefill_tok_buf write timing vs ttnn.embedding read.\n"
            "    2. prefill_pos_buf positions wrong (off-by-one).\n"
            "    3. cache writes happening twice in Path B → SDPA reads\n"
            "       stale K/V at row L-1."
        )

    print(f"  VERDICT: {verdict}")
    print()
    print("  Diagnosis:")
    for line in diagnosis.splitlines():
        print(f"    {line}")
    print()

    # Persist results for post-session review.
    out_dir = PROJECT_ROOT / ".cache" / "gm4_trace_ladder"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"trace_ladder_L{L}_{ts}.json"
    out_path.write_text(json.dumps({
        "L": L,
        "argmax_A": argmax_A,
        "argmax_B": argmax_B,
        "cos_final_lastrow": cos_final,
        "verdict": verdict,
        "diagnosis": diagnosis,
        "ids_prefix": ids[:16],
    }, indent=2))
    log(f"results → {out_path}")

    return 0 if argmax_A == argmax_B else 1


if __name__ == "__main__":
    sys.exit(main())
