#!/usr/bin/env python3
"""Long-context argmax regression gate for Gemma 4.

Purpose: detect precision-drift regressions at long context. Captures
a small fixed set of "fingerprints" (first N decode argmaxes at each
length L) under the CURRENT implementation, then verifies future
implementations still produce the same fingerprints.

Sensitive enough that any bf16 chain-drift change, fp32_dest_acc
toggle, or Typecast elimination will surface here as a mismatch
within the first few decoded tokens (the model has done L × 48 layers
of arithmetic before the first decode argmax). Cheap: ~5 minutes
per run vs the existing needle-haystack test (~30 min).

Mode A (capture, default):
  Run the full forward at each L ∈ {128, 512, 1024, 2048}, decode N=8
  greedy tokens, save the argmax fingerprint to
  research/gemma4_long_context_baseline.json. Commit that file.

Mode B (verify, --verify):
  Re-run the same prompts and compare argmax sequences to the saved
  baseline. Exit non-zero if any L mismatches.

Run on qb1 (or qb2 — model variant via TT_GEMMA4_VARIANT). DO NOT use
GM4_NUM_LAYERS_OVERRIDE — this gate needs the full 48-layer forward
to actually test long-context precision behaviour.

  ssh qb1 'cd ~/tt-xla && tt-smi -r 0,1,2,3 >/dev/null 2>&1 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=... \\
    .venv/bin/python -u experiments/cb/isolate/gemma4_long_context_argmax_gate.py [--verify]'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as srv  # noqa: E402

# Fixed seeds so the prompts are reproducible across runs.
BASELINE_PATH = (PROJECT_ROOT / "research" /
                  "gemma4_long_context_baseline.json")

# 2026-06-09 #289 Step 2: L=4032 (NOT 4096) added to exercise the
# K = sequence_length attention S·V contraction near the cap. The
# server's MAX_KV=4096; prefill writes positions 0..L-1 and we then
# decode N_DECODE=8 more — so L+N_DECODE must be ≤ MAX_KV. With L=4032
# we have 64 positions of slack (4032 + 8 = 4040 < 4096), which leaves
# room for the decode trace warmup writes too. L=4096 itself crashed
# silently inside the kernel because of this out-of-bounds.
LENGTHS = [128, 512, 1024, 2048, 4032]
N_DECODE = 8        # tokens to sample after prefill (greedy argmax)
BOS = 2

# A long, fixed paragraph we can splice/repeat to reach each L. Picked
# so it never hits a special-token (the encoded ids stay in the normal
# vocab range), and so the same byte content reproduces across runs.
SEED_PARAGRAPH = (
    "The history of computing spans many centuries from the abacus "
    "to modern silicon chips. Early mechanical calculators like the "
    "Pascaline gave way to electromechanical machines and eventually "
    "to fully electronic computers. The transistor revolutionized the "
    "field in the late 1940s enabling much smaller and faster devices. "
    "Integrated circuits packed thousands then millions of transistors "
    "onto a single chip. Today modern processors contain billions of "
    "transistors and execute instructions in parallel across many cores. "
    "Memory hierarchies range from registers to caches to main memory "
    "and finally to disk and network storage with each level trading "
    "speed for capacity. Operating systems mediate access between "
    "hardware resources and the user-space programs that depend on them. "
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cur_pos(state):
    return int(ttnn.to_torch(
        state.cur_pos_buf,
        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    ).flatten()[0].item())


def build_prompt(tok, target_len: int) -> list[int]:
    """Repeat SEED_PARAGRAPH until token count >= target_len, trim to
    exactly target_len. BOS prepended."""
    text = SEED_PARAGRAPH
    while True:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) >= target_len - 1:  # -1 for BOS
            break
        text = text + SEED_PARAGRAPH
    ids = ids[: target_len - 1]
    return [BOS] + ids


def capture_fingerprint(state, prompt_ids: list[int],
                         n_decode: int) -> list[int]:
    """Prefill prompt_ids (overwriting cache positions 0..L-1), then
    decode n_decode greedy steps. Returns the decode argmax sequence.
    """
    # Prefill: sequential decode-style (our only prefill today).
    last_argmax = None
    for pos, tok in enumerate(prompt_ids):
        last_argmax = srv.step_forward_v031(state, tok_id=int(tok), pos=pos)
    decoded = [int(last_argmax)]
    # Continue decoding using the traced path (faster than eager).
    srv.ensure_decode_trace(state, log=lambda *a, **k: None)
    feed_tok = int(last_argmax)
    feed_pos = _cur_pos(state) + 1
    for _ in range(n_decode - 1):
        argmax = srv.step_forward_traced(state, token_id=feed_tok,
                                          cur_pos=feed_pos)
        decoded.append(int(argmax))
        feed_tok = int(argmax)
        feed_pos += 1
    return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                         help="compare against saved baseline (default: capture)")
    args = parser.parse_args()

    if args.verify and not BASELINE_PATH.exists():
        print(f"ERROR: --verify requires {BASELINE_PATH}; capture first",
              file=sys.stderr)
        return 2

    log("STAGE 1: bootstrap target (~90s)…")
    state = srv.State()
    srv.bootstrap(state, log=log)

    tok = state.tokenizer

    results: dict[str, list[int]] = {}
    for L in LENGTHS:
        prompt_ids = build_prompt(tok, target_len=L)
        log(f"L={L}: prompt has {len(prompt_ids)} tokens "
            f"(first 6={prompt_ids[:6]})")
        t0 = time.time()
        fp = capture_fingerprint(state, prompt_ids, n_decode=N_DECODE)
        log(f"L={L}: decoded {N_DECODE} argmaxes in {time.time()-t0:.1f}s")
        log(f"L={L}: fingerprint = {fp}")
        results[str(L)] = fp

    if args.verify:
        baseline = json.loads(BASELINE_PATH.read_text())
        log("=" * 72)
        log("VERIFY")
        log("=" * 72)
        all_pass = True
        for L in LENGTHS:
            key = str(L)
            saved = baseline.get(key, [])
            now = results[key]
            n_match = sum(1 for a, b in zip(saved, now) if a == b)
            ok = (saved == now)
            log(f"  L={L:5d}: saved={saved}")
            log(f"            now  ={now}")
            log(f"            match={n_match}/{N_DECODE} "
                f"{'✓ PASS' if ok else '✗ FAIL'}")
            if not ok:
                all_pass = False
        log("=" * 72)
        if all_pass:
            log("VERDICT: ✓ ALL L PASS — precision invariant holds")
            return 0
        log("VERDICT: ✗ FAIL — precision regression detected. Do NOT ship.")
        return 1

    # Capture mode.
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(results, indent=2) + "\n")
    log(f"baseline captured → {BASELINE_PATH}")
    log("Commit this file. Future precision-changing PRs must run with --verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
