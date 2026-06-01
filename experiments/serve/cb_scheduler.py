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


def _sample_from_topk(values, indices, sp, rng):
    """Per-slot sample from a sorted top-K row (W2). `values`/`indices` are
    1-D [K] arrays (largest first; ttnn.topk with sorted=True). Applies
    user-requested top_k clip + top_p nucleus truncation in K-space, then
    samples. Returns the vocab-space token id."""
    import numpy as np
    k_req = int(sp.get("top_k", 0) or 0)
    k_eff = len(values) if k_req <= 0 else min(k_req, len(values))
    v = values[:k_eff].astype(np.float64) / max(float(sp.get("temperature", 1.0)), 1e-6)
    v -= v.max()
    p = np.exp(v); p /= p.sum()
    top_p = float(sp.get("top_p", 1.0))
    if 0.0 < top_p < 1.0:
        # values are already sorted desc → cumsum directly is the nucleus.
        cum = np.cumsum(p)
        keep = int(np.searchsorted(cum, top_p)) + 1
        p = p[:keep]; p /= p.sum()
    pick = int(rng.choice(len(p), p=p))
    return int(indices[pick])


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

    def __init__(self, state, B, max_new, eos_id, use_trace=False, sampling=False,
                 topk_k=None, chunked_prefill=False):
        self.state = state
        self.B = B
        self.max_new = max_new
        self.eos_id = eos_id
        self.use_trace = use_trace
        # chunked_prefill: alternating PREFILL_ONLY / DECODE_ONLY scheduler pattern
        # (TT vLLM convention; research/27b_chunked_prefill_prior_art.md). When a
        # new request is admitted, the next step runs S1a chunked prefill on the
        # production state + cb_prefill_transplant into the slot — one step
        # regardless of L, instead of L decode steps. Other slots are paused
        # during that one PREFILL step (matches TT vLLM "no mixed batches").
        # Default OFF (existing 1-tok/iter decode-loop prefill behaviour).
        self.chunked_prefill = bool(chunked_prefill)
        if self.chunked_prefill:
            state.cb_conv_mode = 'kdim'                  # bit-identical to prod
            state.cb_dn_recurrence_mode = 'manual'       # ditto
        # sampling mode: per-slot host temp/top-p/top-k each step.
        #   sampling=False, use_trace=True             → argmax trace (P0 fast path).
        #   sampling=True,  use_trace=True, topk_k=N   → topk trace (W2). N ~128
        #     amortizes well at B>=16; ~6× total step at B=32 vs the logits
        #     trace, but adds ~100ms of fixed device cost that HURTS at low B
        #     (e.g. B=4 step grew 131→232ms in measurement).
        #   sampling=True,  use_trace=True, topk_k=None → logits trace (P1/P3.5).
        #     Per-slot numpy sample over full [B, vocab] readback. Best at low
        #     B (solo chat); host loop dominates at high B.
        #   sampling=True,  use_trace=False            → eager logits forward.
        # The trace modes never coexist (one engine = one mode), so there's no
        # "mixing eager forwards with execute_trace" hazard either way.
        self.sampling = sampling
        self.topk_k = int(topk_k) if topk_k else None
        self._trace_id = None
        self._argmax_handle = None
        self._logits_handle = None
        self._topk_values_handle = None
        self._topk_indices_handle = None
        self._prefill_trace_id = None
        self._prefill_trace_out = None
        cb.setup_cb_state(state, B)
        cb.cb_reset_states(state)
        # Capture prefill trace BEFORE the decode trace. Each trace reserves
        # device scratch addresses; capturing prefill first means decode's
        # allocations won't overlap with prefill's pre-baked addresses.
        if self.chunked_prefill:
            self._capture_prefill_trace()
        if use_trace:
            self._capture_trace()
        self.slots = [None] * B          # slot -> request id or None
        self.waiting = deque()           # request ids
        self.reqs = {}                   # id -> request dict
        self._next_id = 0

    def _capture_trace(self):
        """Capture the batched forward once. step() then replays via
        execute_trace at compute speed. Admission (cb_reset_slots) and
        update_input_buffers run eager BETWEEN replays — they mutate the
        persistent input buffers, which the next replay reads.

        Three trace tails (one per engine mode):
          sampling=False             → ttnn.argmax (returns [B,1]).
          sampling=True,  topk_k=N   → ttnn.topk    (returns ([B,K], [B,K])).
          sampling=True,  topk_k=None → logits      (returns [B, vocab])."""
        import ttnn
        st = self.state
        if self.sampling and self.topk_k is not None:
            kw = {"return_topk": self.topk_k}
        elif self.sampling:
            kw = {"return_logits": True}
        else:
            kw = {}
        for i in range(2):  # JIT warmup (capture-during-JIT hangs on Blackhole)
            cb.update_input_buffers_batched(st, [DUMMY_TOK] * self.B, [i] * self.B)
            out = cb.forward_batch_tp_inner(st, **kw)
            if isinstance(out, tuple):
                for h in out: ttnn.deallocate(h)
            else:
                ttnn.deallocate(out)
        ttnn.synchronize_device(st.mesh)
        cb.update_input_buffers_batched(st, [DUMMY_TOK] * self.B, [2] * self.B)
        self._trace_id = ttnn.begin_trace_capture(st.mesh, cq_id=0)
        handle = cb.forward_batch_tp_inner(st, **kw)
        ttnn.end_trace_capture(st.mesh, self._trace_id, cq_id=0)
        if self.sampling and self.topk_k is not None:
            self._topk_values_handle, self._topk_indices_handle = handle
        elif self.sampling:
            self._logits_handle = handle
        else:
            self._argmax_handle = handle
        cb.cb_reset_states(st)  # clean slate after warmup dirtied the state

    def release(self):
        import ttnn
        if self._trace_id is not None:
            ttnn.release_trace(self.state.mesh, self._trace_id)
            self._trace_id = None
        if self._prefill_trace_id is not None:
            ttnn.release_trace(self.state.mesh, self._prefill_trace_id)
            self._prefill_trace_id = None

    def submit(self, prompt, sampling=None):
        """sampling: None → greedy (argmax); else a dict
        {temperature, top_p, top_k, seed}. Only honoured in sampling mode."""
        rid = self._next_id; self._next_id += 1
        rng = None
        if sampling is not None:
            import numpy as np
            rng = np.random.default_rng(sampling.get('seed'))
        self.reqs[rid] = {
            'id': rid, 'prompt': [int(t) for t in prompt], 'gen': [],
            'cur_pos': 0, 'next_tok': int(prompt[0]), 'status': 'WAIT', 'slot': None,
            'sampling': sampling, 'rng': rng,
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

    def _capture_prefill_trace(self):
        """Capture forward_prefill_chunked_traced_inner as a trace. One-shot;
        replayed thereafter via trace_id. Mutates production state buffers
        during warmup + capture — call BEFORE any in-flight requests."""
        import ttnn
        st = self.state
        if not hasattr(st, 'prefill_chunk_size') or not hasattr(st, 'dn_chunked_q'):
            return None  # prefill trace prerequisites not in this build
        # JIT warmup
        dummy = [0] * st.prefill_chunk_size
        base.update_prefill_input_buffers(st, dummy)
        for _ in range(2):
            base._reset_state_buffers(st)
            out = base.forward_prefill_chunked_traced_inner(st)
            ttnn.synchronize_device(st.mesh)
            ttnn.deallocate(out)
        # Capture
        base._reset_state_buffers(st)
        base.update_prefill_input_buffers(st, dummy)
        self._prefill_trace_id = ttnn.begin_trace_capture(st.mesh, cq_id=0)
        self._prefill_trace_out = base.forward_prefill_chunked_traced_inner(st)
        ttnn.end_trace_capture(st.mesh, self._prefill_trace_id, cq_id=0)
        return self._prefill_trace_id

    def _step_prefill_chunked(self):
        """PREFILL_ONLY step. For L <= chunk_size with a captured trace: replay
        the trace (3.3s constant, ~10x speedup at L=128). Otherwise fall back
        to eager forward_prefill_chunked_tp (legacy path, still functional).
        Both finish with cb_prefill_transplant into the CB slot."""
        rid = None; slot = None
        for s in range(self.B):
            if self.slots[s] is None and self.waiting:
                rid = self.waiting.popleft(); slot = s
                break
        if rid is None:
            return False
        r = self.reqs[rid]
        import numpy as np, ttnn
        cb.cb_reset_slots(self.state, [slot])
        base._reset_state_buffers(self.state)
        prompt = r['prompt']
        L = len(prompt)
        traced = (getattr(self, '_prefill_trace_id', None) is not None
                   and L <= self.state.prefill_chunk_size)
        if traced:
            padded = list(prompt) + [0] * (self.state.prefill_chunk_size - L)
            base.update_prefill_input_buffers(self.state, padded)
            ttnn.execute_trace(self.state.mesh, self._prefill_trace_id,
                                cq_id=0, blocking=False)
            ttnn.synchronize_device(self.state.mesh)
            full = ttnn.to_torch(self._prefill_trace_out,
                mesh_composer=ttnn.ConcatMeshToTensor(self.state.mesh, dim=0)
            ).float().numpy()[:self.state.prefill_chunk_size, :self.state.vocab_size]
            first_tok = int(np.argmax(full[L - 1]))
        else:
            cap = base.forward_prefill_chunked_tp(self.state, prompt, capture_logits=True)
            ttnn.synchronize_device(self.state.mesh)
            first_tok = int(np.argmax(cap[-1]))
        cb.cb_prefill_transplant(self.state, slot, L)
        ttnn.synchronize_device(self.state.mesh)
        self.slots[slot] = rid
        r['slot'] = slot
        r['cur_pos'] = L
        r['next_tok'] = first_tok
        r['gen'] = [first_tok]
        r['status'] = 'DECODE'
        self._finish(r, slot, first_tok)
        return True

    def _finish(self, r, s, last_out):
        done = (len(r['gen']) >= self.max_new) or (last_out == self.eos_id)
        if done:
            r['status'] = 'DONE'
            self.slots[s] = None
        return done

    def cancel(self, rid):
        """Evict a request mid-flight. Frees its slot (the next _admit calls
        cb_reset_slots → fresh DN state for the new occupant; KV self-overwrites,
        cur_pos-bounded) or drops it from the waiting queue. Same eviction
        mechanism as a finished request — no device op here. Returns True if it
        was live."""
        r = self.reqs.get(rid)
        if r is None or r['status'] in ('DONE', 'CANCELLED'):
            return False
        s = r['slot']
        if s is not None and self.slots[s] == rid:
            self.slots[s] = None
        if rid in self.waiting:
            self.waiting.remove(rid)
        r['status'] = 'CANCELLED'
        return True

    def step(self):
        """One scheduler iteration. Returns #active slots.

        Two modes:
         - chunked_prefill=True: when any request is waiting AND any slot is
           free, run a PREFILL_ONLY step (S1a chunked prefill + transplant for
           ONE request). Otherwise run a DECODE_ONLY step on the active slots.
           Matches TT vLLM scheduler (research/27b_chunked_prefill_prior_art.md).
         - chunked_prefill=False: original behaviour — admit into slots and
           advance ALL slots one token per iteration through the decode forward
           (the waiting request's prompt is consumed one tok/iter through the
           same forward; the slot transitions to DECODE on the last prompt
           token).
        """
        if self.chunked_prefill:
            if self.waiting and any(s is None for s in self.slots):
                self._step_prefill_chunked()
                return sum(1 for s in self.slots if s is not None)
        self._admit()
        toks, curs = [DUMMY_TOK] * self.B, [0] * self.B
        for s in range(self.B):
            rid = self.slots[s]
            if rid is not None:
                r = self.reqs[rid]
                toks[s] = r['next_tok']; curs[s] = r['cur_pos']
        if self.sampling:
            out = self._step_sampled(toks, curs)
        elif self.use_trace:
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

    def _step_sampled(self, toks, curs):
        """Sampling-mode step. Dispatches to the topk path (W2 — opt-in via
        topk_k) or the logits path (P3.5 default).

        Times two sub-segments (cb_metrics histograms): device = execute_trace
        + to_torch + upcast (includes device sync); sample = the per-slot host
        sample loop. Lets us answer host-loop-vs-device at any B."""
        if self.topk_k is not None:
            return self._step_sampled_topk(toks, curs)
        return self._step_sampled_logits(toks, curs)

    def _step_sampled_logits(self, toks, curs):
        """Logits trace + per-slot full-vocab sample. Best at low B (solo chat)."""
        import time
        import ttnn
        cb.update_input_buffers_batched(self.state, toks, curs)
        t0 = time.perf_counter()
        if self.use_trace:
            ttnn.execute_trace(self.state.mesh, self._trace_id, cq_id=0, blocking=False)
            rm = self._logits_handle
        else:
            rm = cb.forward_batch_tp_inner(self.state, return_logits=True)
        t = ttnn.to_torch(rm, mesh_composer=ttnn.ConcatMeshToTensor(self.state.mesh, dim=0))
        if not self.use_trace:
            ttnn.deallocate(rm)
        logits = t[:self.B].float().numpy()
        t1 = time.perf_counter()
        argmax_all = logits.argmax(axis=-1)  # W1: one vectorised call for greedy slots
        out = [DUMMY_TOK] * self.B
        for s in range(self.B):
            rid = self.slots[s]
            if rid is None:
                continue
            sp = self.reqs[rid]['sampling']
            if sp is None:
                out[s] = int(argmax_all[s])
            else:
                out[s] = base._sample_from_logits(
                    logits[s], sp.get('temperature', 1.0), sp.get('top_p', 1.0),
                    sp.get('top_k', 0), self.reqs[rid]['rng'])
        t2 = time.perf_counter()
        if hasattr(self, 'm_device'):
            self.m_device.observe(t1 - t0); self.m_sample.observe(t2 - t1)
        return out

    def _step_sampled_topk(self, toks, curs):
        """W2: topk trace + per-slot sample over K. ~6× total step at B=32 vs
        the logits path; HURTS at low B (~75% slower at B=4 — the topk op has
        fixed device cost that only amortises at large B). Opt-in via topk_k."""
        import time
        import ttnn
        cb.update_input_buffers_batched(self.state, toks, curs)
        t0 = time.perf_counter()
        if self.use_trace:
            ttnn.execute_trace(self.state.mesh, self._trace_id, cq_id=0, blocking=False)
            vals_h = self._topk_values_handle
            idxs_h = self._topk_indices_handle
        else:
            vals_h, idxs_h = cb.forward_batch_tp_inner(self.state, return_topk=self.topk_k)
        composer = ttnn.ConcatMeshToTensor(self.state.mesh, dim=0)
        vals_t = ttnn.to_torch(vals_h, mesh_composer=composer)
        idxs_t = ttnn.to_torch(idxs_h, mesh_composer=composer)
        if not self.use_trace:
            ttnn.deallocate(vals_h); ttnn.deallocate(idxs_h)
        vals = vals_t[:self.B].float().numpy()    # [B, K]
        idxs = idxs_t[:self.B].long().numpy()     # [B, K]
        t1 = time.perf_counter()
        out = [DUMMY_TOK] * self.B
        for s in range(self.B):
            rid = self.slots[s]
            if rid is None:
                continue
            sp = self.reqs[rid]['sampling']
            if sp is None:
                out[s] = int(idxs[s, 0])  # topk is sorted; index 0 == argmax
            else:
                out[s] = _sample_from_topk(vals[s], idxs[s], sp, self.reqs[rid]['rng'])
        t2 = time.perf_counter()
        if hasattr(self, 'm_device'):
            self.m_device.observe(t1 - t0); self.m_sample.observe(t2 - t1)
        return out

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
