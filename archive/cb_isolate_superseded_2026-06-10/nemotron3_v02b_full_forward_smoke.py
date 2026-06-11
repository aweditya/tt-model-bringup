#!/usr/bin/env python3
"""MM7 v0.2.b — full 52-layer forward + final_norm + lm_head + argmax.

Streams layers (upload → run → deallocate) to fit the 23 MoE EP layers
(~640 MB/chip each) inside the 8 GB P150 budget. Bootstrap loads only
top-level (embed + final_norm + lm_head + layer_types). Each layer is
uploaded right before its forward call and deallocated immediately
after — peak memory ≤ top-level + 1 layer.

Pipeline:
  embed(prompt_ids)
  → for L in 0..52: upload_one_layer(L); h = block(h, L); dealloc(L)
  → final_norm(h)
  → lm_head + argmax at last position
  → compare to HF predicted next token

Gate: TT argmax(logits[-1]) == HF argmax at the same position.

REUSE: forks v0.2.a (L0..L5 chain smoke); calls upload_one_layer +
deallocate_layer (added to server v0.2.b) for the streaming part.

Run on qb1 (default EP mode):
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/nemotron3_v02b_full_forward_smoke.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_nemotron3_nano"
COS_GATE = 0.99   # bf16 chain drift expected across 52 layers
N_LAYERS = 52


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_and_mad(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.mean(np.abs(a - b)))
    return cos, mad


def main() -> int:
    # IMPORTANT: don't pre-populate NEMOTRON3_UPLOAD_LAYERS — we stream.
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    log("loading HF oracle prompt_ids + last hidden_state + predicted token…")
    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")
    hf_logits_argmax = None
    hf_argmax_path = ORACLE_DIR / "argmax.npy"
    if hf_argmax_path.exists():
        hf_argmax_arr = np.load(hf_argmax_path)
        hf_logits_argmax = int(hf_argmax_arr.flatten()[-1])
    log(f"  prompt_ids: {prompt_ids.shape}  "
        f"hidden_states[-1] (post-final_norm-eq): {hidden_states[-1].shape}  "
        f"hf_argmax: {hf_logits_argmax}")

    log("bootstrapping (top-level only — streaming layers)…")
    import server_nemotron3_nano_ttnn as srv
    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"  bootstrap in {time.time() - t0:.1f}s")
    # Ensure per_layer_tt is sized for streaming (bootstrap leaves it
    # empty when no env layers requested).
    if not state.per_layer_tt:
        state.per_layer_tt = [None] * N_LAYERS

    try:
        # ── Embed ──────────────────────────────────────────────────
        log("embed lookup…")
        h_np = srv.embed_lookup(state, prompt_ids[None, :])  # [1, S, HIDDEN]
        emb_cos, emb_mad = cos_and_mad(h_np[0], hidden_states[0])
        log(f"  embed cos={emb_cos:.6f} mad={emb_mad:.4e} (vs HF hs[0])")

        # ── 52-layer stream ────────────────────────────────────────
        t_chain = time.time()
        gates_pass = 0
        for L in range(N_LAYERS):
            kind = state.layer_types[L]
            t_layer = time.time()
            # Upload
            state.per_layer_tt[L] = srv.upload_one_layer(state, L, log)
            t_upload = time.time() - t_layer
            # Forward
            t_fwd = time.time()
            if kind == "attention":
                res = srv.attn_block_eager(state, h_np, L)
            elif kind == "mamba2":
                res = srv.mamba2_block_eager(state, h_np, L)
            elif kind == "moe":
                res = srv.moe_block_eager_ep(state, h_np, L)
            else:
                raise NotImplementedError(f"L{L} kind={kind!r}")
            t_fwd = time.time() - t_fwd
            h_np = res["block_out"]
            if h_np.ndim == 2:
                h_np = h_np[None]
            # Per-layer gate vs HF (informational, not blocking)
            cos, mad = cos_and_mad(h_np[0], hidden_states[L + 1])
            if cos >= COS_GATE:
                gates_pass += 1
            # Deallocate
            srv.deallocate_layer(state, L)
            t_total = time.time() - t_layer
            log(f"L{L:>2d} ({kind:>9s}) cos={cos:.6f} mad={mad:.3e} "
                f"upload={t_upload:.1f}s fwd={t_fwd:.1f}s total={t_total:.1f}s")
        log(f"  52-layer stream in {time.time() - t_chain:.1f}s; "
            f"{gates_pass}/{N_LAYERS} layers ≥ cos {COS_GATE}")

        # ── Final norm + lm_head + argmax ──────────────────────────
        log("final_norm + lm_head + argmax…")
        h_final = srv.apply_final_norm(state, h_np)
        fn_cos, fn_mad = cos_and_mad(h_final[0], hidden_states[-1])
        log(f"  final_norm cos={fn_cos:.6f} mad={fn_mad:.4e} (vs HF hs[-1])")
        logits_np, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
        if argmax_np.ndim == 2:
            argmax_np = argmax_np[0]
        tt_last_argmax = int(argmax_np[-1])
        log(f"  TT argmax at last position: {tt_last_argmax}")

        if hf_logits_argmax is not None:
            ok = tt_last_argmax == hf_logits_argmax
            log(f"  HF argmax at last position: {hf_logits_argmax}")
            log(f"  Gate (argmax match): {'PASS ✓' if ok else 'FAIL ✗'}")
            return 0 if ok else 1
        else:
            log("  (no HF argmax oracle artifact present; only printed TT argmax)")
            return 0
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
