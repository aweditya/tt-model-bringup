#!/usr/bin/env python3
"""#290 P5 smoke — chunked prefill + decode handoff via server-level API.

Bootstraps Gemma 4, tokenizes a long-context chat prompt, calls
`server_gemma4_unified_ttnn.forward_prefill_chunked_tp` (server-level
chunked prefill), then continues with N sequential decode steps. Prints
TTFT (prefill wall) + tok/s decode + the generated text.

Validates the integration path for cb_engine / chat scripts: any caller
that needs "ingest a long prompt then start generating" can now use the
server-level chunked entry instead of N × step_forward_v031.

Run:
  scripts/run_remote.sh experiments/cb/isolate/gemma4_chunked_prefill_chat_smoke.py
  scripts/run_remote.sh experiments/cb/isolate/gemma4_chunked_prefill_chat_smoke.py 2048
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as srv  # noqa: E402

L = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N_DECODE = 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 72)
    log(f"#290 P5 chat smoke — prefill L={L} + decode {N_DECODE} tokens")
    log("=" * 72)

    log("STAGE 1: bootstrap…")
    state = srv.State()
    srv.bootstrap(state, log=log)

    log(f"STAGE 2: build {L}-token prompt")
    from gemma4_long_context_argmax_gate import build_prompt
    token_ids = build_prompt(state.tokenizer, target_len=L)
    log(f"  first 6 tokens: {token_ids[:6]}")

    log(f"STAGE 3: chunked prefill ({L} tokens, parallel)…")
    t0 = time.time()
    first_tok = srv.forward_prefill_chunked_tp(state, token_ids)
    prefill_wall = time.time() - t0
    log(f"  TTFT = {prefill_wall:.2f}s")
    log(f"  first generated token id = {first_tok}")

    log(f"STAGE 4: decode {N_DECODE} tokens sequentially…")
    generated = [first_tok]
    t0 = time.time()
    for i in range(1, N_DECODE):
        next_tok = srv.step_forward_v031(state, tok_id=int(generated[-1]),
                                          pos=L + i - 1)
        generated.append(int(next_tok))
    decode_wall = time.time() - t0
    decode_tok_s = (N_DECODE - 1) / decode_wall if decode_wall > 0 else float("inf")
    log(f"  decode wall = {decode_wall:.2f}s, throughput = {decode_tok_s:.1f} tok/s")

    log("STAGE 5: detokenize")
    text = state.tokenizer.decode(generated)
    log(f"  generated text: {text!r}")

    log("=" * 72)
    log(f"OVERALL: TTFT {prefill_wall:.2f}s, {decode_tok_s:.1f} tok/s decode")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
