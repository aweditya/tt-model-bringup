#!/usr/bin/env python3
"""Real correctness gate for owned_gdn integration: prefill + greedy generate.

Runs the same prompt through 35B both ways (manual DN recurrence vs the fused
qwen36_gdn_decode_owned kernel) and prints the decoded text. Definitive gate
for the multi-step bug we suspected — if owned_gdn produces coherent text,
the kernel is fine; if it produces garbage, the mesh write-back bug is real.

Two bootstraps (one per mode), each ~4 min. Total ~10 min.

Run (qb1):
  .venv/bin/python -u experiments/utils/test_owned_gdn_greedy_generation.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402
import ttnn  # noqa: E402

PROMPT = "The capital of France is"
N_GENERATE = 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one(mode_label, owned_gdn):
    log(f"=== mode = {mode_label} ===")
    state = srv.State()
    state.dn_owned_gdn = owned_gdn
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    prompt_ids = state.tokenizer.encode(PROMPT)
    log(f"prompt: {PROMPT!r}  ids={prompt_ids}")

    # Prefill: feed prompt tokens at positions 0..n-1; track per-position next_id.
    next_id = None
    for pos, tok in enumerate(prompt_ids):
        next_id = srv.step_forward_ttnn(state, tok, pos)
    log(f"prefill done; last next_id = {next_id}")

    # Greedy decode N tokens, feeding each predicted token at the next position.
    generated = []
    cur_pos = len(prompt_ids)
    for _ in range(N_GENERATE):
        generated.append(next_id)
        next_id = srv.step_forward_ttnn(state, generated[-1], cur_pos)
        cur_pos += 1
    log(f"generated ids = {generated}")
    log(f"decoded text = {state.tokenizer.decode(prompt_ids + generated)!r}")

    ttnn.close_mesh_device(state.mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return generated


def main():
    manual_ids = run_one("manual", owned_gdn=False)
    log("")
    owned_ids = run_one("owned_gdn", owned_gdn=True)
    log("")
    log("=== COMPARISON ===")
    log(f"manual    : {manual_ids}")
    log(f"owned_gdn : {owned_ids}")
    n_match = sum(1 for a, b in zip(manual_ids, owned_ids) if a == b)
    log(f"matching positions: {n_match}/{len(manual_ids)}")


if __name__ == "__main__":
    main()
