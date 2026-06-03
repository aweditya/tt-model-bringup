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
import os
import sys
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

# Backend dispatch: TT_BACKEND selects which model module to use as
# `base` (single-stream State + bootstrap) and `cb` (CB forward wrapper).
# Mirrors cb_api.py:BACKENDS. Without this dispatch cb_scheduler ALWAYS
# bound to 27B even when cb_api loaded 35B (caught 2026-06-02 — server
# loaded 35B weights but the scheduler ran 27B forward, producing
# coherent-but-wrong-model responses).
import importlib  # noqa: E402
_BACKEND_MODULES = {
    "27b":   ("server_tp",        "server_tp_cb"),
    "35b":   ("server_35b_ttnn",  "server_35b_cb"),
}
_TT_BACKEND = os.environ.get("TT_BACKEND", "27b")
if _TT_BACKEND not in _BACKEND_MODULES:
    raise ValueError(
        f"unknown TT_BACKEND={_TT_BACKEND!r}; valid: {sorted(_BACKEND_MODULES)}")
_base_mod, _cb_mod = _BACKEND_MODULES[_TT_BACKEND]
base = importlib.import_module(_base_mod)
cb = importlib.import_module(_cb_mod)
from live_slot_store import LiveSlotStore  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DUMMY_TOK = 0  # token fed to parked/FREE slots (output ignored)
PREFIX_CACHE_MIN_MATCH = 16  # tokens; below this don't bother caching


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
                 topk_k=None, chunked_prefill=False, prefix_cache=False):
        self.state = state
        self.B = B
        self.max_new = max_new
        self.eos_id = eos_id
        self.use_trace = use_trace
        # prefix_cache: slot-level content-keyed prefix caching. When a request
        # completes, its slot is kept "live" indexed by hash(tokens_so_far)
        # instead of being torn down. On the next admit, if the new prompt has
        # the cached prefix, we reclaim the live slot at cur_pos=n_matched and
        # only decode the new suffix (no re-prefill). Plan: research/27b_prefix_caching_plan.md.
        # Default OFF — bit-identical to today.
        self.prefix_cache = bool(prefix_cache)
        self.live_slots = LiveSlotStore(min_match_tokens=PREFIX_CACHE_MIN_MATCH) \
            if self.prefix_cache else None
        # Prefix-cache metrics (incremented by P3+P6; safe to expose now).
        self.pc_hits = 0
        self.pc_misses = 0
        self.pc_evictions = 0
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
        # Two-phase warmup for multi-trace coexistence (vLLM #352): compile
        # ALL paths first (enable_trace=False), THEN capture all back-to-back.
        # If we interleave warmup+capture, decode JIT compilation between
        # captures allocates buffers that can land on prefill trace's reserved
        # memory → 99% CPU hang on second capture (T5a-T5f all failed this way).
        if self.chunked_prefill:
            self._warmup_prefill()
        if use_trace:
            self._warmup_decode()
        if self.chunked_prefill or use_trace:
            import ttnn
            ttnn.synchronize_device(state.mesh)
        if self.chunked_prefill:
            self._capture_prefill_trace_only()
        if use_trace:
            self._capture_decode_trace_only()
        self.slots = [None] * B          # slot -> request id or None
        self.waiting = deque()           # request ids
        self.reqs = {}                   # id -> request dict
        self._next_id = 0

    def _decode_kw(self):
        """Trace-tail kwargs for the decode forward (one per engine mode):
          sampling=False              → ttnn.argmax [B,1]
          sampling=True,  topk_k=N    → ttnn.topk ([B,K], [B,K])
          sampling=True,  topk_k=None → logits [B, vocab]"""
        if self.sampling and self.topk_k is not None:
            return {"return_topk": self.topk_k}
        elif self.sampling:
            return {"return_logits": True}
        return {}

    def _warmup_decode(self):
        """Phase 1 of two-phase warmup — eager forward to populate program
        cache. NO trace capture; allocator stays in 'normal' mode."""
        import ttnn
        st = self.state
        kw = self._decode_kw()
        for i in range(2):
            cb.update_input_buffers_batched(st, [DUMMY_TOK] * self.B, [i] * self.B)
            out = cb.forward_batch_tp_inner(st, **kw)
            if isinstance(out, tuple):
                for h in out: ttnn.deallocate(h)
            else:
                ttnn.deallocate(out)

    def _capture_decode_trace_only(self):
        """Phase 2 — capture (assumes _warmup_decode has been called +
        synchronize_device fired). No JIT, no allocations during capture."""
        import ttnn
        st = self.state
        kw = self._decode_kw()
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
        """Admit waiting requests into free slots.

        With prefix_cache=True, slot allocation prefers slots NOT currently in
        the live cache (so we never evict a useful prefix unnecessarily). Only
        when every free slot is also a live-cached slot do we LRU-evict one.
        Reclaimed slots are removed from the cache *before* cb_reset_slots so
        the cache entry never points at stale state.

        With prefix_cache=False this is the original first-free-wins behavior.
        """
        admitted = []
        for s in self._slot_alloc_order():
            if self.slots[s] is None and self.waiting:
                rid = self.waiting.popleft()
                r = self.reqs[rid]
                self.slots[s] = rid
                r['slot'] = s; r['cur_pos'] = 0
                r['next_tok'] = r['prompt'][0]; r['status'] = 'PREFILL'
                if self.prefix_cache and s in self.live_slots:
                    # We're about to wipe this slot's state via cb_reset_slots.
                    # Drop its prefix cache entry first so it never points at
                    # zeroed state.
                    self.live_slots.reclaim(s)
                    self.pc_evictions += 1
                admitted.append(s)
        if admitted:
            cb.cb_reset_slots(self.state, admitted)  # fresh DN state for new seqs
        return admitted

    def _admit_from_cache(self):
        """For each waiting request, check the live-slot cache for a prefix
        match. On hit, reclaim the cached slot at cur_pos = n_matched and
        transition the request to PREFILL state (so the existing per-step loop
        feeds prompt[n_matched..L-1] through the decode trace, then switches
        to DECODE — same mechanism as cold prefill, just starting partway
        through). On miss, leave in waiting for normal admit.

        Degenerate case: if n_matched == len(prompt), the new prompt is exactly
        the cached prefix (no new user content). Skip — let it fall through to
        normal admit and cold-prefill again. Loses the cache benefit for this
        rare case but keeps state transitions consistent.

        Idempotent and safe to call every step; only processes waiting requests.
        """
        if not self.waiting:
            return
        # Iterate over a snapshot so we can mutate self.waiting (remove hits).
        for rid in list(self.waiting):
            r = self.reqs[rid]
            slot_id, n = self.live_slots.find_longest_match(r['prompt'])
            if slot_id is None:
                self.pc_misses += 1
                continue  # leave in waiting for normal admit
            if n >= len(r['prompt']):
                # Degenerate exact-match: no new suffix. Drop to normal admit
                # so the request goes through cold prefill (rare; chat clients
                # always append a new user message on turn N).
                self.pc_misses += 1
                continue
            # Cache hit — reclaim the slot. Note: slots in live_slots have
            # self.slots[s] is None by invariant (mark_live only fires after
            # _finish flips it to None).
            assert self.slots[slot_id] is None, \
                f"prefix cache invariant: cached slot {slot_id} should be free"
            self.live_slots.reclaim(slot_id)
            self.slots[slot_id] = rid
            r['slot'] = slot_id
            r['cur_pos'] = n
            r['next_tok'] = r['prompt'][n]
            r['status'] = 'PREFILL'  # existing PREFILL loop consumes the suffix
            # DO NOT call cb_reset_slots — the slot's DN+KV state IS the cached
            # prefix's state, which is exactly what we want.
            self.waiting.remove(rid)
            self.pc_hits += 1

    def _slot_alloc_order(self):
        """Iteration order for slot allocation: non-cached free slots first,
        then cached free slots in LRU order (oldest first → evict-friendly).
        Without prefix caching, returns range(self.B) (original behavior)."""
        if not self.prefix_cache:
            return range(self.B)
        non_cached = [s for s in range(self.B)
                      if self.slots[s] is None and s not in self.live_slots]
        # live_slots.slot_ids() is LRU order (oldest first) — perfect for eviction
        cached_free = [s for s in self.live_slots.slot_ids()
                       if self.slots[s] is None]
        return non_cached + cached_free

    def _warmup_prefill(self):
        """Phase 1 — eager forward_prefill_chunked_traced_inner to populate
        program cache. NO capture. Returns False if prerequisites missing."""
        import ttnn
        st = self.state
        if not hasattr(st, 'prefill_chunk_size') or not hasattr(st, 'dn_chunked_q'):
            return False
        dummy = [0] * st.prefill_chunk_size
        base.update_prefill_input_buffers(st, dummy)
        for _ in range(2):
            base._reset_state_buffers(st)
            out = base.forward_prefill_chunked_traced_inner(st)
            ttnn.synchronize_device(st.mesh)
            ttnn.deallocate(out)
        return True

    def _capture_prefill_trace_only(self):
        """Phase 2 — capture forward_prefill_chunked_traced_inner as a trace.
        Assumes _warmup_prefill ran + synchronize_device fired."""
        import ttnn
        st = self.state
        if not hasattr(st, 'prefill_chunk_size') or not hasattr(st, 'dn_chunked_q'):
            return None
        dummy = [0] * st.prefill_chunk_size
        base._reset_state_buffers(st)
        base.update_prefill_input_buffers(st, dummy)
        self._prefill_trace_id = ttnn.begin_trace_capture(st.mesh, cq_id=0)
        self._prefill_trace_out = base.forward_prefill_chunked_traced_inner(st)
        ttnn.end_trace_capture(st.mesh, self._prefill_trace_id, cq_id=0)
        ttnn.synchronize_device(st.mesh)
        return self._prefill_trace_id

    def _step_prefill_chunked(self):
        """PREFILL_ONLY step. For L <= chunk_size with a captured trace: replay
        the trace (3.3s constant, ~10x speedup at L=128). Otherwise fall back
        to eager forward_prefill_chunked_tp (legacy path, still functional).
        Both finish with cb_prefill_transplant into the CB slot."""
        rid = None; slot = None
        for s in self._slot_alloc_order():
            if self.slots[s] is None and self.waiting:
                rid = self.waiting.popleft(); slot = s
                break
        if rid is None:
            return False
        r = self.reqs[rid]
        import numpy as np, ttnn
        if self.prefix_cache and slot in self.live_slots:
            # Evicting a cached slot for a new prefill — drop the stale entry
            # before cb_reset_slots wipes the state it pointed at.
            self.live_slots.reclaim(slot)
            self.pc_evictions += 1
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
            # Prefix cache: keep the slot's DN+KV state alive under the request's
            # full token sequence so a returning chat can resume from cur_pos
            # without re-prefill.
            #
            # IMPORTANT: do NOT strip the trailing EOS. For Qwen3.6 chat the
            # "EOS" detected here is usually <|im_end|>, which the chat
            # template ALSO emits between turns. If we strip it, turn 2's
            # tokenization (which includes <|im_end|> after assistant_1) won't
            # match the cached prefix. The slot's DN+KV state already processed
            # that token, so storing it in tokens_so_far is correct.
            if self.prefix_cache:
                tokens_so_far = list(r['prompt']) + list(r['gen'])
                if len(tokens_so_far) >= PREFIX_CACHE_MIN_MATCH:
                    self.live_slots.mark_live(s, tokens_so_far)
        return done

    def cancel(self, rid, mark_live=False):
        """Evict a request mid-flight. Frees its slot (the next _admit calls
        cb_reset_slots → fresh DN state for the new occupant; KV self-overwrites,
        cur_pos-bounded) or drops it from the waiting queue. Same eviction
        mechanism as a finished request — no device op here. Returns True if it
        was live.

        mark_live=True: caller is signaling "this request hit a per-request cap
        (e.g. max_tokens), and r['gen'] is the COMPLETE generated response — the
        slot's state is a valid cache prefix." mark_live=False (default): user
        cancel / engine-side abort; the partial state is not useful, drop it.
        """
        r = self.reqs.get(rid)
        if r is None or r['status'] in ('DONE', 'CANCELLED'):
            return False
        s = r['slot']
        if s is not None and self.slots[s] == rid:
            self.slots[s] = None
            if mark_live and self.prefix_cache:
                tokens_so_far = list(r['prompt']) + list(r['gen'])
                if len(tokens_so_far) >= PREFIX_CACHE_MIN_MATCH:
                    self.live_slots.mark_live(s, tokens_so_far)
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

        With prefix_cache=True, BEFORE either mode runs, waiting requests are
        checked against the live-slot cache. Cache hits reclaim their cached
        slot directly, bypassing prefill — the new suffix gets consumed via the
        normal PREFILL state path (one tok/iter through the decode trace),
        starting at cur_pos = n_matched.
        """
        # Prefix-cache fast path: admit any waiting requests whose prompts match
        # a cached prefix. This must run BEFORE chunked_prefill / _admit so that
        # cache-hit admits skip the expensive prefill paths entirely.
        if self.prefix_cache:
            self._admit_from_cache()
        if self.chunked_prefill:
            if self.waiting and any(s is None for s in self.slots):
                # CW1 fix: only take the chunked-prefill traced path when the
                # request fits in chunk_size. For L > chunk_size, the existing
                # _step_prefill_chunked falls back to eager forward_prefill_
                # chunked_tp, which allocates device tensors at runtime
                # (ttnn.from_torch + every op output). With both prefill and
                # decode traces captured, those allocations collide with
                # trace-reserved memory → wedge after a couple of requests.
                #
                # The L>chunk_size case is uncommon in chat (only first-turn
                # prompts > 32 tokens; turn-2+ hits prefix cache). Falling
                # through to _admit + 1-tok/iter through the decode trace is
                # slower but allocation-free, so it's safe alongside captured
                # traces.
                next_rid = self.waiting[0]
                next_L = len(self.reqs[next_rid]['prompt'])
                if next_L <= self.state.prefill_chunk_size:
                    self._step_prefill_chunked()
                    return sum(1 for s in self.slots if s is not None)
        # Idle-step guard: if prefix caching is enabled, never run a forward
        # when there's nothing to do — the (DUMMY_TOK, cur_pos=0) feed for FREE
        # slots mutates DN state, which would corrupt any live-cached slot's
        # DN state. With this guard, between turns of a chat (no active +
        # waiting), step() returns 0 without touching the device → cache pristine.
        # (Pre-prefix-cache behavior preserved when prefix_cache=False.)
        if self.prefix_cache:
            has_active = any(s is not None for s in self.slots)
            has_admissible = bool(self.waiting) and any(s is None for s in self.slots)
            if not has_active and not has_admissible:
                return 0
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
        vals = vals_t[:self.B].float().numpy()    # [B, K] (or [B, 1, K] on 35B)
        idxs = idxs_t[:self.B].long().numpy()     # [B, K] (or [B, 1, K] on 35B)
        # 35B's hidden activations carry a seq dim, so its topk output is 3-D
        # ([B, 1, K] post-mesh-concat-slice) where 27B is 2-D ([B, K]). Squeeze
        # the dangling seq dim so `idxs[s, 0]` is a scalar (not a length-K row).
        if vals.ndim == 3 and vals.shape[1] == 1:
            vals = vals.squeeze(1)
            idxs = idxs.squeeze(1)
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
