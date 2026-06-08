#!/usr/bin/env python3
"""Phase 2.A smoke — drafter forward fed by TARGET's exposed KV.

End-to-end validation that Phase 2.A target-side change
(`server_gemma4_unified_ttnn.read_shared_kv_for_drafter`) produces KV
the drafter (server_gemma4_12b_assistant_ttnn.drafter_forward) can
consume to match the HF oracle's argmax.

Two paths compared:
  (A) HF-KV path — feed HF oracle's `shared_kv_{sliding,full}_{K,V}.npy`
      directly into drafter_forward (Phase 1 already validated:
      `experiments/cb/isolate/gemma4_assistant_forward_smoke.py` 5/5 PASS).
  (B) TT-EXPOSED-KV path — prefill the prompt through the TARGET
      server (server_gemma4_unified_ttnn), call the new
      read_shared_kv_for_drafter helper, feed those numpy K/V into the
      same drafter forward.

PASS gate per Phase 2.A spec:
  - argmax(B) == argmax(A) (drafter still produces the SAME top-1 token
    when fed TT-exposed KV instead of HF KV)
  - logits cos(B, A) >= 0.95  (drafter SDPA tolerance to ~0.96 cos
    sliding KV input is the real question — the layout probe at
    `gemma4_target_kv_layout_probe.py` showed KV cos 0.96 vs HF;
    threshold 0.95 on drafter LOGITS gives ~2× safety margin).

Runs ONE prompt (prompt 0 = "The capital of France is") to keep wall
budget low — both bootstraps total ~2.5 min warm; full target boot is
~92s (per probe run3). Both target + drafter share the (1,4) qb1 mesh
so we open the mesh ONCE via the target bootstrap then load drafter
weights onto the same mesh.

Run on qb1:
    ssh qb1 'cd ~/tt-xla && TT_GEMMA4_VARIANT=it \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$HOME/tenstorrent/tt-metal/build_Release \\
        ARCH_NAME=blackhole \\
        PYTHONPATH=$HOME/tenstorrent/tt-metal/ttnn \\
        LD_LIBRARY_PATH=$HOME/tenstorrent/tt-metal/ttnn/ttnn:$HOME/tenstorrent/tt-metal/build_Release/ttnn:$HOME/tenstorrent/tt-metal/build_Release/lib \\
        .venv/bin/python -u experiments/cb/isolate/gemma4_target_kv_expose_smoke.py'
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401 — used inside ttnn from_torch round-trips

# HF oracle ran against IT — pin the target to IT variant so KV weights match.
os.environ.setdefault("TT_GEMMA4_VARIANT", "it")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as tgt  # noqa: E402
import server_gemma4_12b_assistant_ttnn as drf  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_assistant"
COS_THRESH_LOGITS = 0.95
PROMPT_IDX = 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(
        a.reshape(-1).astype(np.float64) - b.reshape(-1).astype(np.float64)
    ).max())


def load_prompt(i: int):
    base = ORACLE_DIR / f"prompt_{i}"
    return {
        "input_ids": np.load(base / "input_ids.npy"),
        "drafter_inputs_embeds": np.load(base / "drafter_inputs_embeds.npy"),
        "shared_kv_sliding_K": np.load(base / "shared_kv_sliding_K.npy"),
        "shared_kv_sliding_V": np.load(base / "shared_kv_sliding_V.npy"),
        "shared_kv_full_K": np.load(base / "shared_kv_full_K.npy"),
        "shared_kv_full_V": np.load(base / "shared_kv_full_V.npy"),
        "drafter_logits": np.load(base / "drafter_logits.npy"),
        "drafter_hidden": np.load(base / "drafter_hidden.npy"),
        "drafter_argmax": np.load(base / "drafter_argmax.npy"),
        "drafter_topk_ids": np.load(base / "drafter_topk_ids.npy"),
    }


def main() -> int:
    if not ORACLE_DIR.exists():
        log(f"FATAL: oracle missing at {ORACLE_DIR}")
        return 1

    d = load_prompt(PROMPT_IDX)
    L = int(d["input_ids"].shape[1])
    log(f"prompt {PROMPT_IDX} input_ids L={L} → {d['input_ids'].flatten().tolist()}")
    hf_argmax = int(d["drafter_argmax"].flatten().item())
    log(f"HF drafter argmax: {hf_argmax}, top8={d['drafter_topk_ids'].flatten().tolist()}")

    # Co-resident bootstrap: target opens the mesh; drafter reuses it.
    log("─" * 64)
    log("STAGE 1: bootstrap TARGET (Gemma 4 12B IT) on (1,4) qb1 mesh…")
    t0 = time.time()
    tgt_state = tgt.State()
    tgt.bootstrap(tgt_state, log=log)
    log(f"target bootstrap took {time.time()-t0:.1f}s")
    log(f"target.last_sliding_idx={tgt_state.last_sliding_idx} "
        f"last_full_idx={tgt_state.last_full_idx}")

    log("─" * 64)
    log("STAGE 2: bootstrap DRAFTER on the SAME mesh…")
    t0 = time.time()
    drf_state = drf.State()
    # Co-resident: reuse target's already-open mesh + already-initialised
    # fabric. The drafter's bootstrap unconditionally calls
    # ttnn.set_fabric_config + ttnn.open_mesh_device. Re-setting fabric
    # config after the target's first call destroys its fabric context
    # (TT_FATAL "Trying to get un-initialized fabric context" → IndexError:
    # map::at on the next all_reduce). Monkey-patch BOTH ttnn.set_fabric_config
    # and ttnn.open_mesh_device to no-ops for the duration of the drafter's
    # bootstrap; the target's mesh + fabric context stay live throughout.
    _orig_open = ttnn.open_mesh_device
    _orig_fab = ttnn.set_fabric_config
    ttnn.open_mesh_device = lambda *a, **kw: tgt_state.mesh
    ttnn.set_fabric_config = lambda *a, **kw: None
    try:
        drf.bootstrap(drf_state, log=log)
    finally:
        ttnn.open_mesh_device = _orig_open
        ttnn.set_fabric_config = _orig_fab
    # Ensure drafter sees the right mesh handle after bootstrap returns.
    drf_state.mesh = tgt_state.mesh
    log(f"drafter bootstrap took {time.time()-t0:.1f}s")

    log("─" * 64)
    log("STAGE 3: prefill the prompt through the TARGET (writes paged KV cache)")
    t_p = time.time()
    for pos in range(L):
        tok = int(d["input_ids"][0, pos])
        tt_argmax = tgt.step_forward_v031(tgt_state, tok_id=tok, pos=pos)
        log(f"  pos={pos} tok={tok} → target argmax={tt_argmax}")
    log(f"prefill took {time.time()-t_p:.1f}s")

    log("─" * 64)
    log("STAGE 4: read TARGET KV via Phase 2.A helper "
        "(read_shared_kv_for_drafter)…")
    t_r = time.time()
    tt_kv = tgt.read_shared_kv_for_drafter(tgt_state, L_kv=L)
    log(f"readback took {(time.time()-t_r)*1000:.1f} ms")
    K_sl_tt, V_sl_tt = tt_kv["sliding_attention"]
    K_fl_tt, V_fl_tt = tt_kv["full_attention"]
    log(f"  tt sliding K shape={K_sl_tt.shape}  V shape={V_sl_tt.shape}")
    log(f"  tt full    K shape={K_fl_tt.shape}  V shape={V_fl_tt.shape}")

    # Quick sanity vs HF KV (informational — not a gate).
    log(f"  sanity vs HF: sliding K cos={cos(K_sl_tt, d['shared_kv_sliding_K']):.4f} "
        f"mad={mad(K_sl_tt, d['shared_kv_sliding_K']):.4f}")
    log(f"  sanity vs HF: full    K cos={cos(K_fl_tt, d['shared_kv_full_K']):.4f} "
        f"mad={mad(K_fl_tt, d['shared_kv_full_K']):.4f}")

    log("─" * 64)
    log("STAGE 5: DRAFTER FORWARD (A) — with HF-KV (Phase 1 baseline, expect "
        f"argmax={hf_argmax})")
    shared_hf = {
        "sliding_attention": (d["shared_kv_sliding_K"], d["shared_kv_sliding_V"]),
        "full_attention":    (d["shared_kv_full_K"],    d["shared_kv_full_V"]),
    }
    t_a = time.time()
    out_A = drf.drafter_forward(drf_state, d["drafter_inputs_embeds"], shared_hf)
    log(f"  drafter (HF KV) wall: {(time.time()-t_a)*1000:.1f} ms")
    tt_argmax_A = int(out_A["argmax"].flatten().item())
    log(f"  drafter argmax (HF KV) = {tt_argmax_A} "
        f"(HF says {hf_argmax}; {'MATCH' if tt_argmax_A == hf_argmax else 'MISMATCH'})")
    log(f"  drafter logits cos vs HF = {cos(out_A['logits'], d['drafter_logits']):.6f}")

    log("─" * 64)
    log("STAGE 6: DRAFTER FORWARD (B) — with TT-EXPOSED KV (Phase 2.A end-to-end)")
    shared_tt = {
        "sliding_attention": (K_sl_tt, V_sl_tt),
        "full_attention":    (K_fl_tt, V_fl_tt),
    }
    t_b = time.time()
    out_B = drf.drafter_forward(drf_state, d["drafter_inputs_embeds"], shared_tt)
    log(f"  drafter (TT KV) wall: {(time.time()-t_b)*1000:.1f} ms")
    tt_argmax_B = int(out_B["argmax"].flatten().item())
    log(f"  drafter argmax (TT KV) = {tt_argmax_B}")
    log(f"  drafter logits cos vs HF = {cos(out_B['logits'], d['drafter_logits']):.6f}")

    log("─" * 64)
    log("PHASE 2.A SMOKE GATE — drafter(TT KV) vs drafter(HF KV)")
    log("─" * 64)
    c_logits = cos(out_B["logits"], out_A["logits"])
    m_logits = mad(out_B["logits"], out_A["logits"])
    log(f"  argmax: HF-KV path = {tt_argmax_A}  TT-KV path = {tt_argmax_B}  "
        f"{'EXACT' if tt_argmax_A == tt_argmax_B else 'DIFFER'}")
    log(f"  logits cos(B vs A) = {c_logits:.6f}  mad={m_logits:.4e}")
    # Also report top-K overlap.
    topkA = np.argpartition(out_A["logits"].reshape(-1), -8)[-8:]
    topkB = np.argpartition(out_B["logits"].reshape(-1), -8)[-8:]
    overlap = len(set(topkA.tolist()) & set(topkB.tolist()))
    log(f"  top-8 overlap (A∩B): {overlap}/8")

    in_hf_topk = tt_argmax_B in d["drafter_topk_ids"].flatten().tolist()
    log(f"  TT-KV argmax in HF top-8: {'YES' if in_hf_topk else 'NO'}")

    # Gates.
    pass_argmax = (tt_argmax_A == tt_argmax_B)
    pass_logits = (c_logits >= COS_THRESH_LOGITS)
    overall = pass_argmax and pass_logits
    log("")
    log(f"VERDICT: {'PASS' if overall else 'FAIL'}  "
        f"(argmax={'OK' if pass_argmax else 'NO'}, "
        f"logits_cos>={COS_THRESH_LOGITS}={'OK' if pass_logits else 'NO'})")

    ttnn.close_mesh_device(tgt_state.mesh)
    log("mesh closed.")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
