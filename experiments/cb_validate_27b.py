#!/usr/bin/env python3
"""CB validation ladder for the batched 27B forward (server_tp_cb.py).

Gate ladder (per the CB plan):
  3a. B=1 batched forward == production B=1 forward, bit-identical
      (next-token id match across a short prompt). KEY checkpoint — proves
      the batched plumbing matches the proven path before any B>1 risk.
  3b. B=B identical slots → all slots produce the same next-token as B=1.
  3c. B=B different slots → each slot matches its own B=1 reference.

Bootstraps the production 27B server (server_tp) once, runs both the
production B=1 forward and the batched forward, compares.

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb_validate_27b.py --max-pos 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base       # noqa: E402
import server_tp_cb as cb      # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prod_next_ids(state, prompt_ids):
    """Run production B=1 forward over the prompt; return next-token id at
    each position (the production reference sequence)."""
    import ttnn
    base.reset_kv_state(state) if hasattr(base, "reset_kv_state") else None
    ids = []
    for pos, tok in enumerate(prompt_ids):
        base.update_input_buffers(state, int(tok), pos)
        argmax_tt = base.forward_token_tp_inner(state)
        nid = int(ttnn.to_torch(
            argmax_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        ).flatten()[0].item())
        ttnn.deallocate(argmax_tt)
        ids.append(nid)
    return ids


def cb_next_ids(state, prompt_ids, B, slot_prompts=None):
    """Run the batched forward. If slot_prompts is None, all B slots get
    the same prompt_ids (3a/3b). Else slot_prompts is a list of B prompt-id
    lists (3c). Returns [B][positions] next-token ids per slot.

    Assumes equal-length prompts (static batch) — ragged scheduling is CB3.
    """
    import ttnn
    cb.setup_cb_state(state, B)
    cb.cb_reset_states(state)
    if slot_prompts is None:
        slot_prompts = [list(prompt_ids)] * B
    L = len(slot_prompts[0])
    out = [[] for _ in range(B)]
    for pos in range(L):
        toks = [slot_prompts[b][pos] for b in range(B)]
        cur = [pos] * B
        cb.update_input_buffers_batched(state, toks, cur)
        argmax_tt = cb.forward_batch_tp_inner(state)
        am = ttnn.to_torch(
            argmax_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        ).flatten().tolist()
        ttnn.deallocate(argmax_tt)
        for b in range(B):
            out[b].append(int(am[b]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pos", type=int, default=6, help="prompt length to teacher-force")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    tok = state.tokenizer
    prompt = "The capital of France is the city of"
    prompt_ids = tok.encode(prompt)[:args.max_pos]
    log(f"prompt_ids={prompt_ids}")

    log("=== production B=1 reference ===")
    ref = prod_next_ids(state, prompt_ids)
    log(f"  prod next_ids: {ref}")

    log(f"=== 3a: B=1 batched vs production ===")
    cb1 = cb_next_ids(state, prompt_ids, 1)
    match_3a = (cb1[0] == ref)
    log(f"  cb B=1 next_ids: {cb1[0]}")
    log(f"  3a {'PASS — bit-identical to production' if match_3a else 'FAIL'}")

    log(f"=== 3b: B={args.batch} identical slots vs B=1 ===")
    cbB = cb_next_ids(state, prompt_ids, args.batch)
    match_3b = all(cbB[b] == cb1[0] for b in range(args.batch))
    for b in range(args.batch):
        log(f"  slot {b}: {cbB[b]}  {'OK' if cbB[b] == cb1[0] else 'MISMATCH'}")
    log(f"  3b {'PASS' if match_3b else 'FAIL'}")

    ok = match_3a and match_3b
    log(f"\n=== verdict: {'PASS' if ok else 'FAIL'} ===")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
