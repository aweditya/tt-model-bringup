#!/usr/bin/env python3
"""Drafter v0.1 — embed lookup + pre_projection smoke vs HF oracle.

Pattern forks `experiments/cb/isolate/gm4_v01_L0_cos.py` (target's v0.1
cosine validator). Runs `server_gemma4_12b_assistant_ttnn.bootstrap` then
exercises two helpers:

  - `embed_lookup_tt(state, input_ids)`  vs numpy reference
       state.embed_w_np[input_ids]
  - `pre_projection_tt(state, inputs_embeds)`  vs numpy reference
       inputs_embeds @ W.T  (W = pre_projection.weight loaded from HF)

Loads the HF oracle artifacts produced by
`experiments/utils/hf_oracle_gemma4_assistant.py` at
`.cache/hf_oracle_gemma4_12b_assistant/`. Five prompts; we test prompt_0
only here for speed.

Gate:
  - embed cos ≥ 0.999 vs numpy reference (bit-perfect modulo bf16
    quantization).
  - pre_projection cos ≥ 0.999 vs numpy reference.

If pass: drafter bootstrap is correct — weights loaded, mesh + projections
ready for v0.2 layer forward.

Run on qb2:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python -u \\
        experiments/cb/isolate/gemma4_assistant_embed_smoke.py'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_12b_assistant_ttnn as srv  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_assistant"
PASS_THRESH = 0.999


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def mad(a, b):
    return float(np.abs(
        a.reshape(-1).astype(np.float64) - b.reshape(-1).astype(np.float64)
    ).max())


def main() -> int:
    if not ORACLE_DIR.exists():
        log(f"FATAL: oracle missing at {ORACLE_DIR}")
        log("Run experiments/utils/hf_oracle_gemma4_assistant.py first.")
        return 1
    prompt_dir = ORACLE_DIR / "prompt_0"
    if not prompt_dir.exists():
        log(f"FATAL: prompt_0 dir missing at {prompt_dir}")
        return 1

    input_ids = np.load(prompt_dir / "input_ids.npy")  # [1, L]
    drafter_inputs_embeds_hf = np.load(prompt_dir / "drafter_inputs_embeds.npy")  # [1,1,7680]
    log(f"oracle: input_ids shape={input_ids.shape} values={input_ids.flatten().tolist()}")
    log(f"oracle: drafter_inputs_embeds shape={drafter_inputs_embeds_hf.shape}")

    log("bootstrapping drafter (~30 sec)…")
    t0 = time.time()
    state = srv.State()
    srv.bootstrap(state, log=log)
    log(f"bootstrap took {time.time()-t0:.1f}s")

    # === Test 1: embed lookup ===
    log("=" * 64)
    log("Test 1 — embed lookup")
    log("=" * 64)
    log(f"running embed_lookup_tt on input_ids[0,:] ({input_ids.shape[1]} tokens)…")
    embed_tt_np = srv.embed_lookup_tt(state, input_ids)  # [1, L, HIDDEN]
    log(f"  TT embed output shape: {embed_tt_np.shape}")

    # Reference: pure numpy table lookup at the same indices.
    embed_ref = state.embed_w_np[input_ids[0], :]  # [L, HIDDEN]
    embed_ref = embed_ref[None, :, :]  # [1, L, HIDDEN]
    log(f"  numpy ref shape: {embed_ref.shape}")

    c_embed = cos(embed_tt_np, embed_ref)
    m_embed = mad(embed_tt_np, embed_ref)
    log(f"  cos = {c_embed:.7f}   mad = {m_embed:.6f}")
    embed_pass = c_embed >= PASS_THRESH

    # === Test 2: pre_projection ===
    log("=" * 64)
    log("Test 2 — pre_projection")
    log("=" * 64)
    # Pass the HF oracle's drafter_inputs_embeds (the actual model input).
    log(f"running pre_projection_tt on drafter_inputs_embeds ({drafter_inputs_embeds_hf.shape})…")
    pre_proj_tt_np = srv.pre_projection_tt(state, drafter_inputs_embeds_hf)
    log(f"  TT output shape: {pre_proj_tt_np.shape}")

    # Reference: numpy fp32 matmul vs the same HF weight.
    # The HF weight on disk is [out=1024, in=7680]; matmul: x @ W.T = x @ in,out.
    # Load it fresh as float32 from the same safetensors as a sanity check.
    from safetensors import safe_open
    key_to_shard = srv.build_key_to_shard()
    with safe_open(key_to_shard["pre_projection.weight"], framework="pt") as f:
        W_hf = f.get_tensor("pre_projection.weight").float().numpy()  # [1024, 7680]
    log(f"  HF weight (re-loaded) shape: {W_hf.shape}")
    pre_proj_ref = drafter_inputs_embeds_hf.astype(np.float32) @ W_hf.T.astype(np.float32)
    log(f"  numpy ref shape: {pre_proj_ref.shape}")

    c_pre = cos(pre_proj_tt_np, pre_proj_ref)
    m_pre = mad(pre_proj_tt_np, pre_proj_ref)
    log(f"  cos = {c_pre:.7f}   mad = {m_pre:.6f}")
    pre_pass = c_pre >= PASS_THRESH

    # === Summary ===
    log("=" * 64)
    log(f"v0.1 SUMMARY (gate: cos ≥ {PASS_THRESH})")
    log("=" * 64)
    log(f"  embed_lookup:    cos={c_embed:.6f}  mad={m_embed:.6f}  "
        f"[{'PASS' if embed_pass else 'FAIL'}]")
    log(f"  pre_projection:  cos={c_pre:.6f}  mad={m_pre:.6f}  "
        f"[{'PASS' if pre_pass else 'FAIL'}]")
    overall = embed_pass and pre_pass
    log(f"  overall: {'PASS' if overall else 'FAIL'}")

    # Close mesh cleanly.
    import ttnn
    ttnn.close_mesh_device(state.mesh)
    log("mesh closed.")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
