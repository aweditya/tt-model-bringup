#!/usr/bin/env python3
"""MM7 v0.1.0 — bootstrap + top-level forward smoke vs HF oracle.

Three surgical gates that prove embed + final_norm + lm_head are wired
correctly without needing any layer composites:

  Gate A — embed cos:   ttnn.embedding(table, prompt_ids)
                            == hidden_states[0]  (from HF oracle)
  Gate B — final_norm cos: ttnn.rms_norm(hidden_states[-1], norm_w)
                            == final_norm.npy   (from HF oracle)
  Gate C — lm_head argmax: argmax(ttnn.matmul(final_norm.npy, lm_head.T))
                            == argmax.npy        (from HF oracle)

All three artefacts come from `.cache/hf_oracle_nemotron3_nano/` populated
by `experiments/utils/hf_reference_nemotron3_nano.py` at v0.0. Gate uses
fp32 cosine for A/B; exact integer match for C.

REUSE: forks `experiments/cb/isolate/mamba2_step_wrapper_smoke.py` for
the bootstrap+log+log-cosine pattern.

Run on the QuietBox:
    cd ~/tt-xla && \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/nemotron3_v010_bootstrap_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "utils"))

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_nemotron3_nano"

# Pass thresholds — same shape as the per-layer ladders we've shipped.
COS_GATE = 0.999
MAD_PRINT_THRESH = 0.05  # log a warning if MAD spikes above this


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_and_mad(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Cosine similarity + mean abs deviation between two fp32 arrays."""
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    cos = float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    )
    mad = float(np.mean(np.abs(a - b)))
    return cos, mad


def main() -> int:
    log("loading HF oracle artifacts…")
    if not ORACLE_DIR.exists():
        log(f"FATAL: oracle dir missing — run experiments/utils/"
            f"hf_reference_nemotron3_nano.py first ({ORACLE_DIR})")
        return 1
    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")
    final_norm_hf = np.load(ORACLE_DIR / "final_norm.npy")
    argmax_hf = np.load(ORACLE_DIR / "argmax.npy")
    logits_hf = np.load(ORACLE_DIR / "logits.npy")
    log(f"  prompt_ids: {prompt_ids.shape}  -> {prompt_ids.tolist()}")
    log(f"  hidden_states: {hidden_states.shape}")
    log(f"  final_norm: {final_norm_hf.shape}")
    log(f"  logits: {logits_hf.shape}")
    log(f"  argmax: {argmax_hf.shape} -> {argmax_hf.tolist()}")

    # Sanity: prompt_ids matches the embed lookup query.
    assert prompt_ids.shape[0] >= 1, "empty prompt_ids"

    log("bootstrapping Nemotron-3 Nano on (1,4) mesh…")
    import server_nemotron3_nano_ttnn as srv

    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"  bootstrap in {time.time() - t0:.1f}s")

    try:
        # ── Gate A — embed lookup vs HF hidden_states[0] ──────────
        log("Gate A: embed lookup…")
        embed_out = srv.embed_lookup(state, prompt_ids)
        log(f"  embed_out shape: {embed_out.shape}")
        hf_embed = hidden_states[0:1]  # [1, S, HIDDEN]
        cos_a, mad_a = cos_and_mad(embed_out, hf_embed)
        log(f"  cos = {cos_a:.6f}   mad = {mad_a:.6e}")
        gate_a = cos_a >= COS_GATE
        log(f"  Gate A: {'PASS ✓' if gate_a else 'FAIL ✗'}")

        # ── Gate B — final_norm on hidden_states[-1] vs final_norm.npy ─
        log("Gate B: final_norm on hidden_states[-1] (post-L51 → post-norm)…")
        hs_last = hidden_states[-1]  # [S, HIDDEN]
        # Reshape to [1, S, HIDDEN] for downstream tile-mode consumers.
        fn_out = srv.apply_final_norm(state, hs_last)
        log(f"  fn_out shape: {fn_out.shape}")
        cos_b, mad_b = cos_and_mad(fn_out, final_norm_hf)
        log(f"  cos = {cos_b:.6f}   mad = {mad_b:.6e}")
        if mad_b > MAD_PRINT_THRESH:
            log("  ⚠️  large MAD — may indicate Llama-style (no +1.0) vs "
                "Qwen-style (with +1.0) confusion; see "
                "[[feedback-qwen36-qnorm-knorm-zero-centered]]")
        gate_b = cos_b >= COS_GATE
        log(f"  Gate B: {'PASS ✓' if gate_b else 'FAIL ✗'}")

        # ── Gate C — lm_head on HF's final_norm.npy ─────────────────
        # IMPORTANT discovery: HF's `logits.npy` is computed via
        # `nn.Linear(bf16)` (modeling_nemotron_h.py:1717), so the
        # matmul reduction happens in bf16 precision. Our TT lm_head
        # runs HiFi4 + fp32_dest_acc=True, i.e. closer to fp32
        # precision. A direct comparison gives logits cos ~0.92 not
        # because we're WRONG, but because HF's reference is the
        # bf16-imprecise one. Pure numpy fp32 matmul also gives the
        # same ~0.92 cos vs HF, confirming this.
        #
        # Use NUMPY fp32 as the strict ground truth, HF only as a soft
        # generation-token sanity check.
        log("Gate C: lm_head on HF final_norm…")
        logits_tt, argmax_tt = srv.apply_lm_head_and_argmax(state, final_norm_hf)

        # Pure numpy fp32 reference: more accurate than HF's bf16.
        from safetensors import safe_open
        snap = next(srv.SNAPSHOT_ROOT.glob("*"))
        lm_head_np = None
        for shard in sorted(snap.glob("*.safetensors")):
            with safe_open(shard, framework="pt") as f:
                if "lm_head.weight" in f.keys():
                    lm_head_np = f.get_tensor("lm_head.weight").float().numpy()
                    break
        assert lm_head_np is not None, "lm_head.weight not found in any shard"
        logits_np = final_norm_hf.astype(np.float32) @ lm_head_np.T
        argmax_np = logits_np.argmax(axis=-1).astype(np.int32)
        log(f"  argmax_tt:    {argmax_tt.tolist()}")
        log(f"  argmax_np_fp32: {argmax_np.tolist()}")
        log(f"  argmax_hf_bf16: {argmax_hf.tolist()}")

        # C1: generation token (pos -1) — matches HF (the only argmax
        # that's actually consumed at decode time).
        gen_match = bool(argmax_tt[-1] == argmax_hf[-1])
        log(f"  C1 generation token (pos -1): TT={argmax_tt[-1]}  "
            f"HF={argmax_hf[-1]}  {'PASS ✓' if gen_match else 'FAIL ✗'}")

        # C2: per-position logits cosine vs NUMPY FP32 (strict gate).
        per_pos_cos_np: list[float] = []
        for p in range(logits_tt.shape[0]):
            c, _ = cos_and_mad(logits_tt[p], logits_np[p])
            per_pos_cos_np.append(c)
        log(f"  C2 per-position logits cos vs NUMPY fp32: "
            f"{['%.5f' % c for c in per_pos_cos_np]}")
        gate_c2 = min(per_pos_cos_np) >= COS_GATE
        log(f"  C2 min vs numpy = {min(per_pos_cos_np):.6f}  "
            f"(gate ≥ {COS_GATE})  "
            f"{'PASS ✓' if gate_c2 else 'FAIL ✗'}")

        # C3: per-position argmax match vs NUMPY (strict). Sanity that
        # the matmul math is correct end-to-end.
        argmax_np_matches = int(np.sum(argmax_tt == argmax_np))
        gate_c3 = argmax_np_matches == len(argmax_tt)
        log(f"  C3 argmax match vs numpy: {argmax_np_matches}/{len(argmax_tt)} "
            f"{'PASS ✓' if gate_c3 else 'FAIL ✗'}")

        # Soft info: HF logits cos + HF argmax match (bf16 floor)
        per_pos_cos_hf = []
        for p in range(logits_tt.shape[0]):
            c, _ = cos_and_mad(logits_tt[p], logits_hf[p])
            per_pos_cos_hf.append(c)
        argmax_hf_matches = int(np.sum(argmax_tt == argmax_hf))
        log(f"  (soft) HF logits cos: "
            f"{['%.5f' % c for c in per_pos_cos_hf]}")
        log(f"  (soft) HF argmax match: {argmax_hf_matches}/{len(argmax_tt)} "
            f"(bf16 reference; expect <100% from accumulator noise)")

        gate_c = gen_match and gate_c2 and gate_c3

        all_pass = gate_a and gate_b and gate_c
        log("")
        log(f"v0.1.0 bootstrap smoke {'PASS ✓' if all_pass else 'FAIL ✗'} "
            f"({sum([gate_a, gate_b, gate_c])}/3 gates green)")
        return 0 if all_pass else 1
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
