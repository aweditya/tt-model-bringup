#!/usr/bin/env python3
"""Real-world spec-dec perf bench — realistic chat prompts, tok/s breakdown.

After F-1 RoPE fix (commit 86528f6), the spec-dec algorithm is bit-exact
to HF. This bench measures actual generation tok/s on longer realistic
prompts and compares to baseline B=1 traced decode. Per-stage breakdown
(drafter eager × K, verify traced, accept walk, target B=1 cache-write
advance) shows where to focus next perf work.

The 5-prompt v0.0c smoke uses 5-10-token oracle prompts and only emits
~10 tokens per prompt — not representative of real chat workloads where
generation length matters more than prefill length.

Bench design:
- Bootstrap target + drafter ONCE (~90s).
- For each prompt: prefill, then spec-dec rounds until N_GEN tokens
  emitted. Also run a BASELINE pass: prefill same prompt, run target B=1
  traced for N_GEN steps. Compare ms/tok.
- Report breakdown: drafter / verify / cache-advance / accept walk per
  round, and aggregate tok/s.

Run on qb2:
  ssh qb2 'cd ~/tt-xla && tt-smi -r 0,1,2,3 >/dev/null 2>&1 && \\
    TT_GEMMA4_VARIANT=it TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$HOME/tenstorrent/tt-metal/build_Release \\
    ARCH_NAME=blackhole PYTHONPATH=$HOME/tenstorrent/tt-metal/ttnn \\
    LD_LIBRARY_PATH=...:...:... \\
    .venv/bin/python -u experiments/cb/isolate/gemma4_spec_dec_realworld_bench.py'

The script picks the IT variant by default (TT_GEMMA4_VARIANT=it). It
needs both the target (12B-it) and the drafter (12B-it-assistant) cached.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as tgt  # noqa: E402
import server_gemma4_12b_assistant_ttnn as drf  # noqa: E402
import spec_dec_scheduler as sched  # noqa: E402

# Spec-dec config.
K = 3
N_GEN = 24  # tokens to emit per prompt (long enough to amortize one-time costs)

# Realistic prompts — chat-style for IT model. Encoded at runtime via
# drafter's tokenizer (shared vocab with target).
PROMPTS = [
    # Q&A — short prompt + medium-length expected answer
    "Question: What is the capital of France, and what is its main river?\nAnswer:",
    # Story continuation — natural-language gen, drafter likely OK on common words
    "Once upon a time, in a small village nestled between two mountains, there lived a baker who",
    # Code — drafter may struggle (different distribution)
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def co_load():
    log("STAGE 1: bootstrap target (~90s)…")
    tgt_state = tgt.State()
    t0 = time.time()
    tgt.bootstrap(tgt_state, log=log)
    log(f"  target bootstrap took {time.time()-t0:.1f}s")
    log("STAGE 1: bootstrap drafter co-loaded (~8s)…")
    drf_state = drf.State()
    drf_state.mesh = tgt_state.mesh
    _orig_open = ttnn.open_mesh_device
    _orig_fab = ttnn.set_fabric_config
    ttnn.open_mesh_device = lambda *a, **kw: tgt_state.mesh
    ttnn.set_fabric_config = lambda *a, **kw: None
    t0 = time.time()
    try:
        drf.bootstrap(drf_state, log=log)
    finally:
        ttnn.open_mesh_device = _orig_open
        ttnn.set_fabric_config = _orig_fab
    drf_state.mesh = tgt_state.mesh
    log(f"  drafter bootstrap took {time.time()-t0:.1f}s")
    return tgt_state, drf_state


def tokenize(drf_state, text: str) -> list:
    """BPE-encode via drafter's tokenizer; prepend BOS for target prefill."""
    raw = drf_state.tokenizer.encode(text, add_special_tokens=False)
    return [2] + list(raw)  # BOS = 2 for Gemma 4


def prefill_prompt(tgt_state, prompt_ids: list):
    """Prefill target with prompt_ids starting at position 0. Sets
    tgt_state.last_target_hidden_{prev,cur} via step_forward_v031's hook.
    """
    for pos, tok in enumerate(prompt_ids):
        tgt.step_forward_v031(tgt_state, tok_id=int(tok), pos=pos)


def run_baseline(tgt_state, prompt_ids: list, n_gen: int) -> dict:
    """Baseline: target B=1 traced decode for n_gen tokens. Returns
    metrics (ms/tok, total ms, emitted text)."""
    prefill_prompt(tgt_state, prompt_ids)
    # Ensure decode trace exists (idempotent).
    tgt.ensure_decode_trace(tgt_state, log=log)

    cur_pos = int(ttnn.to_torch(
        tgt_state.cur_pos_buf,
        mesh_composer=ttnn.ConcatMeshToTensor(tgt_state.mesh, dim=0),
    ).flatten()[0].item())
    # Last predicted token from prefill = first token to emit.
    last_argmax = int(ttnn.to_torch(
        tgt_state.tok_buf,
        mesh_composer=ttnn.ConcatMeshToTensor(tgt_state.mesh, dim=0),
    ).flatten()[0].item()) if hasattr(tgt_state, "tok_buf") else None

    # Step traced for n_gen tokens.
    emitted = []
    t_start = time.time()
    feed_tok = prompt_ids[-1]
    feed_pos = cur_pos + 1
    for _ in range(n_gen):
        argmax = tgt.step_forward_traced(tgt_state, token_id=feed_tok,
                                          cur_pos=feed_pos)
        emitted.append(argmax)
        feed_tok = argmax
        feed_pos += 1
    elapsed = time.time() - t_start
    ms_per_tok = elapsed * 1000 / n_gen
    try:
        text = tgt_state.tok.decode(emitted)
    except Exception as e:
        text = f"[decode error: {e}]"
    return {
        "n": n_gen,
        "wall_s": elapsed,
        "ms_per_tok": ms_per_tok,
        "tok_per_s": n_gen / elapsed,
        "text": text,
        "emitted": emitted,
    }


def run_spec_dec(tgt_state, drf_state, prompt_ids: list, n_gen: int,
                  K_val: int = K) -> dict:
    """Spec-dec: prefill + rounds until ≥ n_gen tokens emitted. Returns
    per-round breakdown."""
    prefill_prompt(tgt_state, prompt_ids)

    # One-time verify-trace capture (idempotent at same K).
    tgt.setup_verify_kp1_state(tgt_state, K=K_val, log=log)
    cur_pos_init = int(ttnn.to_torch(
        tgt_state.cur_pos_buf,
        mesh_composer=ttnn.ConcatMeshToTensor(tgt_state.mesh, dim=0),
    ).flatten()[0].item())
    tgt.update_verify_inputs(tgt_state, current_pos=cur_pos_init,
                              candidate_token_ids=[0] * (K_val + 1))
    tgt.ensure_verify_trace_kp1(tgt_state, log=log)

    cfg = sched.SpecDecConfig(K=K_val, max_new=n_gen + K_val)
    scheduler = sched.SpecDecScheduler(target_state=tgt_state,
                                         drafter_state=drf_state, config=cfg)
    base_token = int(prompt_ids[-1])
    emitted = []
    per_round_ms = []
    per_round_breakdown = []
    t_start = time.time()
    r = 0
    while len(emitted) < n_gen:
        cur_pos = int(ttnn.to_torch(
            tgt_state.cur_pos_buf,
            mesh_composer=ttnn.ConcatMeshToTensor(tgt_state.mesh, dim=0),
        ).flatten()[0].item())
        round_t = time.time()
        result = scheduler.step(
            base_token=base_token,
            target_h_prev_np=tgt_state.last_target_hidden_prev,
            target_h_last_np=tgt_state.last_target_hidden_cur,
            cur_pos=cur_pos,
        )
        round_ms = (time.time() - round_t) * 1000
        per_round_ms.append(round_ms)
        per_round_breakdown.append({
            "drafter_ms": result.drafter_step_ms,
            "verify_ms": result.verify_step_ms,
            "walk_ms": result.host_walk_ms,
            "target_advance_ms": result.target_step_ms,
            "accept": result.accept_count,
            "emitted_n": len(result.accepted_tokens),
        })
        emitted.extend(result.accepted_tokens)
        base_token = result.accepted_tokens[-1]
        r += 1
        if r > n_gen * 2:  # safety bound
            log(f"  ⚠ safety bound hit at round {r}")
            break
    elapsed = time.time() - t_start
    ms_per_tok = elapsed * 1000 / len(emitted)
    # Aggregate per-stage averages.
    n_r = len(per_round_breakdown)
    avg = {k: sum(b[k] for b in per_round_breakdown) / n_r
           for k in ["drafter_ms", "verify_ms", "walk_ms",
                     "target_advance_ms"]}
    avg["round_total_ms"] = sum(per_round_ms) / n_r
    avg_alpha = sum(b["accept"] for b in per_round_breakdown) / (
        n_r * K_val) if n_r else 0.0
    try:
        text = tgt_state.tok.decode(emitted[:n_gen])
    except Exception as e:
        text = f"[decode error: {e}]"
    return {
        "n": len(emitted),
        "rounds": n_r,
        "wall_s": elapsed,
        "ms_per_tok": ms_per_tok,
        "tok_per_s": len(emitted) / elapsed,
        "avg_alpha": avg_alpha,
        "avg_breakdown_ms": avg,
        "per_round_ms": per_round_ms,
        "text": text,
        "emitted": emitted,
    }


def main():
    log("=" * 72)
    log(f"Real-world spec-dec perf bench (K={K}, N_GEN={N_GEN} tokens, "
        f"{len(PROMPTS)} prompts)")
    log("=" * 72)

    tgt_state, drf_state = co_load()

    log("STAGE 2: dry-run target traced decode capture (~5s)…")
    # Trigger ensure_decode_trace + verify_trace + drafter once on a
    # short prompt so JIT/capture costs don't taint per-prompt timing.
    warmup_ids = tokenize(drf_state, "Hello world")
    prefill_prompt(tgt_state, warmup_ids)
    tgt.ensure_decode_trace(tgt_state, log=log)

    rows = []
    for p_i, ptext in enumerate(PROMPTS):
        ids = tokenize(drf_state, ptext)
        L = len(ids)
        log(f"━━━ PROMPT {p_i}: {ptext[:60]!r}{'…' if len(ptext) > 60 else ''}")
        log(f"  encoded L={L} tokens (first 8 = {ids[:8]})")

        # Baseline pass.
        log(f"  → baseline pass: target B=1 traced × {N_GEN}…")
        try:
            base = run_baseline(tgt_state, ids, N_GEN)
            log(f"    BASELINE: {N_GEN} tok in {base['wall_s']:.1f}s = "
                f"{base['ms_per_tok']:.1f} ms/tok ({base['tok_per_s']:.2f} tok/s)")
            log(f"    TEXT: {base['text']!r}")
        except Exception as e:
            log(f"    ✗ baseline crashed: {e!r}")
            base = None

        # Spec-dec pass.
        log(f"  → spec-dec pass: K={K} rounds until ≥{N_GEN} emitted…")
        try:
            spec = run_spec_dec(tgt_state, drf_state, ids, N_GEN, K_val=K)
            log(f"    SPEC-DEC: {spec['n']} tok in {spec['wall_s']:.1f}s = "
                f"{spec['ms_per_tok']:.1f} ms/tok ({spec['tok_per_s']:.2f} tok/s)")
            log(f"    avg α = {spec['avg_alpha']:.3f} over {spec['rounds']} rounds")
            b = spec["avg_breakdown_ms"]
            log(f"    avg/round: drafter={b['drafter_ms']:.1f}ms "
                f"verify={b['verify_ms']:.1f}ms walk={b['walk_ms']:.1f}ms "
                f"target_adv={b['target_advance_ms']:.1f}ms "
                f"TOTAL={b['round_total_ms']:.1f}ms")
            log(f"    TEXT: {spec['text']!r}")
        except Exception as e:
            log(f"    ✗ spec-dec crashed: {e!r}")
            import traceback; traceback.print_exc()
            spec = None

        rows.append({
            "prompt_idx": p_i,
            "prompt_preview": ptext[:40],
            "L": L,
            "baseline": base,
            "spec": spec,
        })

    # Summary table.
    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)
    log(f"{'prompt':6s} {'L':4s} {'base ms/tok':12s} {'spec ms/tok':12s} "
        f"{'speedup':10s} {'α':6s}")
    log("-" * 72)
    for r in rows:
        b = r["baseline"]; s = r["spec"]
        if b is None or s is None:
            log(f"  {r['prompt_idx']} L={r['L']:3d} [crash]")
            continue
        sup = b["ms_per_tok"] / s["ms_per_tok"]
        log(f"  {r['prompt_idx']:4d} {r['L']:4d} {b['ms_per_tok']:11.1f}  "
            f"{s['ms_per_tok']:11.1f}  {sup:7.2f}×    {s['avg_alpha']:.2f}")
    log("=" * 72)
    log("VERDICT")
    valid = [r for r in rows if r["baseline"] and r["spec"]]
    if not valid:
        log("  ✗ no valid runs")
        return 1
    mean_spec_ms = sum(r["spec"]["ms_per_tok"] for r in valid) / len(valid)
    mean_base_ms = sum(r["baseline"]["ms_per_tok"] for r in valid) / len(valid)
    speedup = mean_base_ms / mean_spec_ms
    log(f"  mean baseline: {mean_base_ms:.1f} ms/tok")
    log(f"  mean spec-dec: {mean_spec_ms:.1f} ms/tok")
    log(f"  net speedup:  {speedup:.2f}× (>1.0 means spec-dec wins)")
    log("=" * 72)
    ttnn.close_mesh_device(tgt_state.mesh)
    return 0 if speedup > 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
