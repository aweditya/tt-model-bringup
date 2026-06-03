#!/usr/bin/env python3
"""v0.1.0 cosine validator — embed_scaled + L0 input_layernorm vs HF oracle.

Fork base: shape from `experiments/cb/dev/cb35_drift_ladder.py` (per
REUSE MANDATE) — cosine ladder + headline metric pattern.

Loads `.cache/hf_oracle_gemma4_12b/` (`hf_reference_gemma4_12b.py`
output) and runs `server_gemma4_unified_ttnn.step_forward_v01` on the
prompt's first token. Compares:
  - TT `embed_scaled` vs HF `hidden_states[0, pos=0, :]` (post-embed,
    post-sqrt-3840 scale).
  - TT `in_norm`       vs HF `L0_in_norm[pos=0, :]`.

Gate: per-sub-step cos ≥ 0.999. Both PASS → bootstrap + embed scale +
RMSNorm `w` convention are correct. Headline printed at end.

Run (qb1):
    bash scripts/run_remote.sh experiments/cb/isolate/gm4_v01_L0_cos.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as srv  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b"
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
    return float(np.abs(a.reshape(-1).astype(np.float64) - b.reshape(-1).astype(np.float64)).max())


def main():
    if not ORACLE_DIR.exists():
        log(f"FATAL: oracle missing at {ORACLE_DIR}; run hf_reference_gemma4_12b.py first")
        return 1

    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")  # [49, seq, 3840]
    hf_embed_scaled = hidden_states[0, 0, :]  # post-embed-scale at pos 0
    hf_in_norm = np.load(ORACLE_DIR / "L0_in_norm.npy")  # [seq, 3840]
    hf_in_norm_pos0 = hf_in_norm[0, :]

    # v0.1.1 sub-step refs: require --hook-attn-layer 0 oracle re-run.
    attn_refs = {}
    optional = ["q_proj", "k_proj", "v_proj", "q_norm", "k_norm"]
    for sub in optional:
        p = ORACLE_DIR / f"L0_attn_L0_{sub}.npy"
        if p.exists():
            attn_refs[sub] = np.load(p)

    log(f"oracle: prompt_ids={prompt_ids.tolist()} hidden_shape={hidden_states.shape}")
    log(f"oracle: v0.1.1 attn sub-step refs: {sorted(attn_refs.keys())}")

    log("bootstrapping Gemma 4 12B server (~85 sec)…")
    t0 = time.time()
    state = srv.State()
    srv.bootstrap(state, log=log)
    log(f"bootstrap took {time.time()-t0:.1f}s")

    log("running v0.1.1 forward at pos 0 (tok_id = prompt_ids[0])…")
    cap = {}
    tok_id0 = int(prompt_ids[0])
    srv.step_forward_v01(state, tok_id=tok_id0, capture=cap)

    log("=" * 64)
    log(f"Gemma 4 12B v0.1.1 cosine ladder vs HF oracle (gate: cos ≥ {PASS_THRESH})")
    log("=" * 64)

    results = []

    # v0.1.0 gates (already PASSing as of commit b9f3c35).
    c_embed = cos(cap["embed_scaled"], hf_embed_scaled)
    c_in    = cos(cap["in_norm"],      hf_in_norm_pos0)
    m_embed = mad(cap["embed_scaled"], hf_embed_scaled)
    m_in    = mad(cap["in_norm"],      hf_in_norm_pos0)
    results.append(("embed_scaled", c_embed, m_embed))
    results.append(("in_norm", c_in, m_in))

    # v0.1.1 gates: per-head comparison [NUM_HEADS, HEAD_DIM] vs HF pos 0.
    # HF q_proj shape: [seq, NUM_Q * head_dim] = [6, 4096] (pre-view).
    # HF q_norm shape: [seq, NUM_Q, head_dim] = [6, 16, 256] (post-view).
    # TT q_proj_out / q_norm_out: [NUM_Q, head_dim] = [16, 256].
    def _hf_proj_pos0_to_head(arr_pos, n_heads, head_dim):
        # arr_pos shape can be [HEAD*DIM] (proj hook output) OR
        # [HEAD, DIM] (norm hook output after view). Reshape to [HEAD, DIM].
        a = arr_pos.reshape(-1)
        return a.reshape(n_heads, head_dim)

    if "q_proj" in attn_refs:
        hf_q = attn_refs["q_proj"][0]  # [4096] flat
        tt_q = cap["q_proj_out"]       # [16, 256]
        c = cos(tt_q, _hf_proj_pos0_to_head(hf_q, 16, 256))
        results.append(("q_proj_out", c, mad(tt_q, _hf_proj_pos0_to_head(hf_q, 16, 256))))
    if "k_proj" in attn_refs:
        hf_k = attn_refs["k_proj"][0]  # [2048] flat
        tt_k = cap["k_proj_out"]       # [8, 256]
        c = cos(tt_k, _hf_proj_pos0_to_head(hf_k, 8, 256))
        results.append(("k_proj_out", c, mad(tt_k, _hf_proj_pos0_to_head(hf_k, 8, 256))))
    if "v_proj" in attn_refs:
        hf_v = attn_refs["v_proj"][0]
        tt_v = cap["v_proj_out"]
        c = cos(tt_v, _hf_proj_pos0_to_head(hf_v, 8, 256))
        results.append(("v_proj_out", c, mad(tt_v, _hf_proj_pos0_to_head(hf_v, 8, 256))))
    if "q_norm" in attn_refs:
        hf_qn = attn_refs["q_norm"][0]  # already [16, 256] post-view
        tt_qn = cap["q_norm_out"]
        c = cos(tt_qn, hf_qn)
        results.append(("q_norm_out", c, mad(tt_qn, hf_qn)))
    if "k_norm" in attn_refs:
        hf_kn = attn_refs["k_norm"][0]
        tt_kn = cap["k_norm_out"]
        c = cos(tt_kn, hf_kn)
        results.append(("k_norm_out", c, mad(tt_kn, hf_kn)))

    all_pass = True
    for name, c, m in results:
        status = "PASS" if c >= PASS_THRESH else "FAIL"
        if c < PASS_THRESH:
            all_pass = False
        log(f"  {name:14s}: cos={c:.6f}  mad={m:.4e}  [{status}]")
    log("=" * 64)
    log(f"VERDICT: {'PASS' if all_pass else 'FAIL'} ({len(results)} sub-steps checked)")

    import ttnn
    ttnn.close_device(state.mesh)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
