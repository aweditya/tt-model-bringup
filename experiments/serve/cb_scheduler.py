#!/usr/bin/env python3
"""CB3 — Orca-style iteration-level continuous-batching scheduler for 27B.

Sits on the validated device primitives (server_tp_cb): batched forward (CB1),
per-slot ragged positions + cb_reset_slots admission (CB2). This is pure Python
orchestration — no new device risk. Design = vLLM/Orca, adopted as-is:

  - Fixed B slots (one TILE width; the trace captures at fixed B — CB4).
  - A waiting queue of requests; iteration-level scheduling: every forward step
    admits waiting requests into FREE slots, advances all active slots one
    token, and evicts finished ones (EOS / max_new) — admission/eviction happen
    BETWEEN steps, not at sequence boundaries (the Orca insight).
  - DeltaNet recurrent state = Mamba-style per-slot state slot; admission calls
    cb_reset_slots([slot]) so the new sequence starts fresh. KV self-overwrites
    (per-slot cur_pos bounds the SDPA read).
  - Prefill is done one token/step through the decode path (simple + correct;
    chunked prefill is a later efficiency item). FREE slots are parked at
    cur_pos=0 with a dummy token — fully isolated (validated), output ignored.

Greedy decode (argmax). Validation main: K requests through B slots; each
request's generation must equal its standalone B=1 greedy reference.

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/serve/cb_scheduler.py --slots 2 --max-new 8
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base       # noqa: E402
import server_tp_cb as cb      # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DUMMY_TOK = 0  # token fed to parked/FREE slots (output ignored)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _batched_step(state, toks, curs):
    """One batched forward; returns per-slot argmax (list[int] len B)."""
    import ttnn
    cb.update_input_buffers_batched(state, toks, curs)
    am = cb.forward_batch_tp_inner(state)
    vals = ttnn.to_torch(
        am, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)).flatten().tolist()
    ttnn.deallocate(am)
    return [int(v) for v in vals[:len(toks)]]


def greedy_ref(state, prompt, max_new, eos_id):
    """Standalone B=1 greedy reference: prefill one token/step, then argmax-feed.
    Mirrors the scheduler's per-slot logic exactly so outputs are comparable."""
    cb.setup_cb_state(state, 1)
    cb.cb_reset_states(state)
    cur, tok, gen = 0, int(prompt[0]), []
    last = len(prompt) - 1
    while True:
        o = _batched_step(state, [tok], [cur])[0]
        if cur < last:
            cur += 1; tok = int(prompt[cur])
        else:
            gen.append(o)
            if len(gen) >= max_new or o == eos_id:
                break
            cur += 1; tok = o
    return gen


class Scheduler:
    """Orca iteration-level scheduler over a fixed pool of B slots."""

    def __init__(self, state, B, max_new, eos_id, use_trace=False):
        self.state = state
        self.B = B
        self.max_new = max_new
        self.eos_id = eos_id
        self.use_trace = use_trace
        self._trace_id = None
        self._argmax_handle = None
        cb.setup_cb_state(state, B)
        cb.cb_reset_states(state)
        if use_trace:
            self._capture_trace()
        self.slots = [None] * B          # slot -> request id or None
        self.waiting = deque()           # request ids
        self.reqs = {}                   # id -> request dict
        self._next_id = 0

    def _capture_trace(self):
        """Capture the batched forward once (CB4 pattern). step() then replays
        via execute_trace at ~compute speed. Admission (cb_reset_slots) and
        update_input_buffers run eager BETWEEN execute_trace calls — they mutate
        the persistent buffers in-place, which the next replay reads."""
        import ttnn
        st = self.state
        for i in range(2):  # JIT warmup (capture-during-JIT hangs on Blackhole)
            cb.update_input_buffers_batched(st, [DUMMY_TOK] * self.B, [i] * self.B)
            am = cb.forward_batch_tp_inner(st); ttnn.deallocate(am)
        ttnn.synchronize_device(st.mesh)
        cb.update_input_buffers_batched(st, [DUMMY_TOK] * self.B, [2] * self.B)
        self._trace_id = ttnn.begin_trace_capture(st.mesh, cq_id=0)
        self._argmax_handle = cb.forward_batch_tp_inner(st)
        ttnn.end_trace_capture(st.mesh, self._trace_id, cq_id=0)
        cb.cb_reset_states(st)  # clean slate after warmup dirtied the state

    def release(self):
        if self._trace_id is not None:
            import ttnn
            ttnn.release_trace(self.state.mesh, self._trace_id)
            self._trace_id = None

    def submit(self, prompt):
        rid = self._next_id; self._next_id += 1
        self.reqs[rid] = {
            'id': rid, 'prompt': [int(t) for t in prompt], 'gen': [],
            'cur_pos': 0, 'next_tok': int(prompt[0]), 'status': 'WAIT', 'slot': None,
        }
        self.waiting.append(rid)
        return rid

    def _admit(self):
        admitted = []
        for s in range(self.B):
            if self.slots[s] is None and self.waiting:
                rid = self.waiting.popleft()
                r = self.reqs[rid]
                self.slots[s] = rid
                r['slot'] = s; r['cur_pos'] = 0
                r['next_tok'] = r['prompt'][0]; r['status'] = 'PREFILL'
                admitted.append(s)
        if admitted:
            cb.cb_reset_slots(self.state, admitted)  # fresh DN state for new seqs
        return admitted

    def _finish(self, r, s, last_out):
        done = (len(r['gen']) >= self.max_new) or (last_out == self.eos_id)
        if done:
            r['status'] = 'DONE'
            self.slots[s] = None
        return done

    def step(self):
        """One scheduler iteration = one batched forward. Returns #active slots."""
        self._admit()
        toks, curs = [DUMMY_TOK] * self.B, [0] * self.B
        for s in range(self.B):
            rid = self.slots[s]
            if rid is not None:
                r = self.reqs[rid]
                toks[s] = r['next_tok']; curs[s] = r['cur_pos']
        if self.use_trace:
            import ttnn
            cb.update_input_buffers_batched(self.state, toks, curs)
            ttnn.execute_trace(self.state.mesh, self._trace_id, cq_id=0, blocking=False)
            vals = ttnn.to_torch(self._argmax_handle,
                                 mesh_composer=ttnn.ConcatMeshToTensor(self.state.mesh, dim=0)
                                 ).flatten().tolist()
            out = [int(v) for v in vals[:self.B]]
        else:
            out = _batched_step(self.state, toks, curs)
        active = 0
        for s in range(self.B):
            rid = self.slots[s]
            if rid is None:
                continue
            active += 1
            r = self.reqs[rid]; o = out[s]
            last_prompt = len(r['prompt']) - 1
            if r['status'] == 'PREFILL' and r['cur_pos'] < last_prompt:
                r['cur_pos'] += 1
                r['next_tok'] = r['prompt'][r['cur_pos']]
            else:  # last prefill token OR a decode step → emit a generated token
                r['gen'].append(o)
                r['status'] = 'DECODE'
                if not self._finish(r, s, o):
                    r['cur_pos'] += 1
                    r['next_tok'] = o
        return active

    def run(self, max_iters=10000):
        it = 0
        while it < max_iters and (self.waiting or any(s is not None for s in self.slots)):
            self.step(); it += 1
        return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--trace", action="store_true",
                    help="run the scheduler via execute_trace (production speed)")
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    state.deltanet_recurrence_mode = "manual"
    state.deltanet_decay_gate_mode = "manual"
    state.deltanet_decay_mode = "native_softplus"
    tok = state.tok
    eos_id = getattr(tok, 'eos_token_id', None)
    eos_id = int(eos_id) if eos_id is not None else -1

    prompts = [
        "The capital of France is the city of",
        "Once upon a time there lived a young",
        "The largest planet in our solar system is",
        "Water boils at a temperature of one hundred",
        "The quick brown fox jumps over the",
    ]
    pid = [tok.encode(p) for p in prompts]

    log(f"=== standalone greedy references (max_new={args.max_new}) ===")
    refs = []
    for i, p in enumerate(pid):
        g = greedy_ref(state, p, args.max_new, eos_id)
        refs.append(g)
        log(f"  req {i}: {g}")

    log(f"=== scheduler ({'TRACED' if args.trace else 'eager'}): "
        f"{len(prompts)} requests through {args.slots} slots ===")
    sched = Scheduler(state, args.slots, args.max_new, eos_id, use_trace=args.trace)
    for p in pid:
        sched.submit(p)
    t0 = time.perf_counter()
    iters = sched.run()
    dt = time.perf_counter() - t0
    gen_tokens = sum(len(sched.reqs[i]['gen']) for i in range(len(pid)))
    log(f"  completed in {iters} scheduler iterations, {dt:.2f}s "
        f"({iters/dt:.1f} iters/s, {gen_tokens/dt:.1f} generated tok/s)")
    sched.release()

    ok = True
    for i in range(len(pid)):
        g = sched.reqs[i]['gen']
        match = (g == refs[i])
        ok = ok and match
        log(f"  req {i}: {g}  {'OK' if match else 'MISMATCH vs ' + str(refs[i])}")

    log(f"\n=== verdict: {'PASS' if ok else 'FAIL'} ===")
    log("  each request's continuous-batched output == its standalone greedy ref"
        if ok else "  scheduler output diverged from standalone references")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
