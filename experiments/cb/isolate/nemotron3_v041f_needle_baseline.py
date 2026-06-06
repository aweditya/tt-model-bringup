#!/usr/bin/env python3
"""MM7 v0.4.1.f — Nemotron-3 needle-in-haystack baseline.

Long-context correctness gate. PASS = the model retrieves the magic
password buried in the haystack.

We test the CURRENT trace-prereq integrated path (post v0.4.1.e: pure-
ttnn embed/lm_head/argmax, on-device reduce_scatter for MoE shard,
on-device matmul-fold conv1d, pure-state Mamba2 SSD). This establishes
the baseline against which any future precision-reducing change
(e.g. on-device router with cos=0.9997 tie-break drift) gets compared.

Follows the Gemma 4 Round 9 lesson: needle-haystack failures can be
PROMPT-shape artifacts (the IT model echoing its instruction), not
precision floors. So we test multiple prompt formats: raw text and
short instruct.

Run via the nm3 dev harness:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v041f_needle_baseline'

Reuses the 35B needle pattern at experiments/utils/needle_haystack_35b_ttnn.py
plus the dev-harness forward loop pattern at v033_nstep_chain_smoke.

Tested at L=128, 512 with 2 trials each (time-budget ~10 min total).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

N_LAYERS = 52
MAX_NEW = int(os.environ.get("NM3_NEEDLE_MAX_NEW", "24"))
RNG_SEED = 1337


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_needle(seed):
    """Build an 8-char alphanumeric needle."""
    rng = np.random.default_rng(seed)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(list(alphabet), size=8))


def build_prompt(tok, target_tokens, frac, needle):
    """Build a haystack prompt with needle inserted at fractional position.

    Args:
      target_tokens: target prompt length (tokens, not chars)
      frac: 0..1 — fraction along the haystack where needle sits
      needle: 8-char string to retrieve

    Returns:
      (prompt_text, token_count)
    """
    needle_sentence = (
        f"REMEMBER THIS: The magic password is {needle}. END REMEMBER."
    )
    # Use a long, neutral filler text that doesn't interfere semantically.
    filler_sentence = (
        "The library was quiet. Books lined the walls. The reader sat "
        "down and opened a volume of poetry from the seventeenth century. "
        "The lamp shed a warm yellow light over the page. "
    )
    question = (
        f" What is the magic password? Answer: "
    )

    def count(text):
        return len(tok.encode(text, add_special_tokens=False))

    # Binary-search a count of filler_sentence repeats to hit target_tokens.
    base_overhead = count(needle_sentence) + count(question)
    target_filler = max(1, target_tokens - base_overhead)
    fillers_total = max(1, target_filler // count(filler_sentence) + 1)
    haystack_words = filler_sentence * fillers_total
    needle_pos = int(len(haystack_words) * frac)
    haystack_with_needle = (
        haystack_words[:needle_pos]
        + needle_sentence + " "
        + haystack_words[needle_pos:]
    )
    prompt = haystack_with_needle + question
    n_prompt = count(prompt)
    return prompt, n_prompt


def score(text, needle):
    """Y if full needle present, P if 4+ contiguous chars, N otherwise."""
    if needle in text:
        return "Y"
    for k in range(len(needle) - 3):
        if needle[k:k+4] in text:
            return "P"
    return "N"


def _forward_layers(state, h_tt, srv, ttnn, *, attn_fn_name: str):
    attn_fn = getattr(srv, attn_fn_name)
    for L in range(N_LAYERS):
        kind = state.layer_types[L]
        if kind == "attention":
            h_next = attn_fn(state, h_tt, L)
        elif kind == "mamba2":
            h_next = srv.mamba2_block_eager_tt(state, h_tt, L)
        elif kind == "moe":
            h_next = srv.moe_block_eager_ep_tt(state, h_tt, L)
        ttnn.deallocate(h_tt)
        h_tt = h_next
    return h_tt


def generate_one(state, srv, ttnn, prompt_ids, max_new):
    """Run full generation: prefill prompt + decode max_new tokens.

    Returns (generated_text, decode_ms_per_token).
    """
    srv.reset_decode_state(state, B=1, log=lambda m: None)

    # PREFILL via eager path (no trace yet).
    h_np = srv.embed_lookup(state, prompt_ids[None, :])
    h_tt = ttnn.from_torch(
        torch.from_numpy(h_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    h_tt = _forward_layers(state, h_tt, srv, ttnn,
                           attn_fn_name="attn_prefill_tt")
    h_np = ttnn.to_torch(
        h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )[:1].float().numpy()
    ttnn.deallocate(h_tt)
    h_final = srv.apply_final_norm(state, h_np)
    _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
    prev_token = int(argmax_np.flatten()[-1])

    state.cur_pos = len(prompt_ids)
    generated = [prev_token]

    # DECODE loop.
    decode_times = []
    for step in range(max_new - 1):
        t0 = time.time()
        h_np = srv.embed_lookup(
            state, np.asarray([[prev_token]], dtype=np.int64),
        )
        h_tt = ttnn.from_torch(
            torch.from_numpy(h_np.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        h_tt = _forward_layers(state, h_tt, srv, ttnn,
                               attn_fn_name="attn_decode_step_tt")
        h_np = ttnn.to_torch(
            h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )[:1].float().numpy()
        ttnn.deallocate(h_tt)
        h_final = srv.apply_final_norm(state, h_np)
        _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
        prev_token = int(argmax_np.flatten()[-1])
        decode_times.append(time.time() - t0)
        state.cur_pos += 1
        generated.append(prev_token)
        if prev_token == state.tokenizer.eos_token_id:
            break

    text = state.tokenizer.decode(generated, skip_special_tokens=True)
    decode_mean_ms = (sum(decode_times) / len(decode_times) * 1000.0
                      if decode_times else 0.0)
    return text, decode_mean_ms


def main(state=None) -> int:
    os.environ.setdefault("NEMOTRON3_UPLOAD_LAYERS", "all")
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    import server_nemotron3_nano_ttnn as srv
    import ttnn

    if state is None:
        log("bootstrap…")
        state = srv.State()
        srv.bootstrap(state, log)
    else:
        log("[harness] reusing live state ✓")

    tok = state.tokenizer

    # Configuration: 2 lengths × 2 trials = 4 total runs ≈ ~10-15 min
    # at current 0.2s/step warm + prefill costs.
    LENGTHS = [128, 512]
    FRACS = [0.5]
    TRIALS = 2

    results = []
    for L_target in LENGTHS:
        for f in FRACS:
            for t in range(TRIALS):
                seed = RNG_SEED + 1000 * L_target + 100 * int(f * 100) + t
                needle = make_needle(seed)
                prompt, n_prompt = build_prompt(tok, L_target, f, needle)
                log(f"L={L_target} frac={f} trial={t} needle={needle} "
                    f"n_prompt={n_prompt}")

                prompt_ids = np.asarray(
                    tok.encode(prompt, add_special_tokens=False),
                    dtype=np.int64,
                )

                t0 = time.time()
                text, decode_ms = generate_one(
                    state, srv, ttnn, prompt_ids, MAX_NEW,
                )
                gen_time = time.time() - t0
                s = score(text, needle)
                log(f"  result={s}  decode={decode_ms:.0f} ms/tok  "
                    f"total={gen_time:.1f}s  text={text[:80]!r}")
                results.append({
                    "L": L_target, "frac": f, "trial": t,
                    "needle": needle, "n_prompt": n_prompt,
                    "result": s, "decode_ms": decode_ms,
                    "gen_time_s": gen_time, "text": text,
                })

    # ── REPORT ──────────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("REPORT — Nemotron-3 needle baseline (post v0.4.1.e)")
    log("=" * 60)
    for L_target in LENGTHS:
        for f in FRACS:
            rows = [r for r in results if r["L"] == L_target and r["frac"] == f]
            ys = sum(1 for r in rows if r["result"] == "Y")
            ps = sum(1 for r in rows if r["result"] == "P")
            ns = sum(1 for r in rows if r["result"] == "N")
            mean_dec = (sum(r["decode_ms"] for r in rows) / len(rows)
                        if rows else 0.0)
            log(f"  L={L_target:>4d}  frac={f}  "
                f"Y={ys}/{len(rows)}  P={ps}/{len(rows)}  N={ns}/{len(rows)}  "
                f"mean decode={mean_dec:.0f} ms/tok")
    log("")
    all_y = all(r["result"] == "Y" for r in results)
    any_y = any(r["result"] == "Y" for r in results)
    if all_y:
        log("v0.4.1.f BASELINE PASS — long-context retrieval clean.")
        log("Safe to layer in further precision changes.")
    elif any_y:
        log("v0.4.1.f BASELINE PARTIAL — some L pass, some don't. Use as")
        log("the bar for any future change; track deltas, not absolutes.")
    else:
        log("v0.4.1.f BASELINE FAIL — no retrieval at any L. May be a")
        log("prompt-shape issue (per Gemma 4 Round 9 lesson) OR a real")
        log("model issue. Try varying the prompt format before declaring")
        log("the model broken.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
