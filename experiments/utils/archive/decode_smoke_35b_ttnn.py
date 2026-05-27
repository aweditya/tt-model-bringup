#!/usr/bin/env python3
"""B16-decode — autoregressive greedy decode smoke for server_35b_ttnn.

Prompt "The capital of France is", greedy decode N tokens, print the
generated text. Validates that state evolution (conv1d + DN recurrent +
KV cache placeholder) works across positions when used autoregressively
(not just teacher-forced).

Run (qb1):
  cd ~/tt-xla && tt-smi -r && \
    export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \
    export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \
    export ARCH_NAME=blackhole && \
    export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \
    export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \
    .venv/bin/python -u experiments/utils/decode_smoke_35b_ttnn.py
"""
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402

import argparse

PROMPT = "The capital of France is"
MAX_NEW = 24  # overridden via --max-new


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    global MAX_NEW
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--fused-qk-norm", choices=["default", "true", "false"], default="default",
                     help="Override state.dn_fused_qk_norm (default keeps State() value).")
    args = ap.parse_args()
    MAX_NEW = args.max_new
    log(f"bootstrap… (max_new={MAX_NEW}, prompt={args.prompt!r})")
    state = srv.State()
    if args.fused_qk_norm != "default":
        state.dn_fused_qk_norm = (args.fused_qk_norm == "true")
        log(f"OVERRIDE dn_fused_qk_norm = {state.dn_fused_qk_norm}")
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    prompt_ids = state.tokenizer.encode(PROMPT)
    log(f"prompt_ids={prompt_ids}")

    # Prefill (teacher-force the prompt; capture argmax at last position)
    log("prefill prompt…")
    t0 = time.time()
    last_argmax = None
    for p, tid in enumerate(prompt_ids):
        last_argmax = srv.step_forward_ttnn(state, tid, p)
    prefill_ms = (time.time() - t0) * 1000
    log(f"prefill {len(prompt_ids)} tok in {prefill_ms:.0f} ms "
        f"({prefill_ms / len(prompt_ids):.0f} ms/tok)")
    log(f"first predicted token (after prompt): id={last_argmax} "
        f"text={state.tokenizer.decode([last_argmax])!r}")

    # Autoregressive decode
    log(f"\ndecoding {MAX_NEW} tokens…")
    generated = [last_argmax]
    pos = len(prompt_ids)
    t0 = time.time()
    for step in range(MAX_NEW - 1):
        next_id = srv.step_forward_ttnn(state, generated[-1], pos)
        generated.append(next_id)
        pos += 1
    decode_wall = time.time() - t0
    log(f"decoded {MAX_NEW - 1} more tok in {decode_wall:.1f}s "
        f"({decode_wall * 1000 / (MAX_NEW - 1):.0f} ms/tok)")

    full_text = state.tokenizer.decode(prompt_ids + generated)
    log(f"\n=== generated text ===")
    log(full_text)
    log(f"=== token ids ===")
    log(str(generated))


if __name__ == "__main__":
    main()
