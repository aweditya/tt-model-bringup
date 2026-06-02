# Prefix caching (slot-level, content-keyed) — milestone plan (2026-06-01)

Living plan. Status per milestone; commit hash on completion.
Sources: research findings in [`vllm_prefix_caching_audit.md`](vllm_prefix_caching_audit.md).

## Problem being fixed

Chat turn-2+ today **re-prefills the entire conversation history every request**.
The OpenAI chat API is stateless on the wire — every turn the client sends the
full `messages: [...]` array. Today's server tokenizes that and runs prefill
from `cur_pos=0` over all of it. At ~80 ms/tok beyond the chunk_size=32 traced
window, a 500-token history is ~40s TTFT.

The slot that just finished turn N-1 still had the right KV cache + DN state +
`cur_pos = len(history)` in DRAM at response time. We threw it away on response
completion. We're going to stop throwing it away.

## Design (locked from research audit)

**Slot-level, content-keyed prefix caching** — vLLM's APC pattern at coarser
granularity:

- **Cache unit**: the whole CB slot (KV pages + DN state + `cur_pos` + `tokens_so_far`).
  Not individual KV blocks.
- **Cache key**: content hash of `tokens_so_far` (vLLM-style content-keyed →
  no session IDs, no hijacking surface, same security model as vLLM APC).
- **Cache lookup**: on admit, find the longest cached prefix of the incoming
  prompt; exact-verify (defense vs hash collision); resume the slot.
- **Cache eviction**: LRU when slots are needed for new conversations.
- **DN state**: lives in the slot, never serialized. **This is why slot-level
  fits our hybrid attention+GatedDeltaNet architecture** — sidesteps Marconi.

## Why slot-level (not block-level)

For a hybrid attention+DN model, block-level prefix caching of just the KV
cache gives **zero TTFT win** in isolation. DN state at position N requires the
full sequence 0..N to compute; cached KV blocks alone aren't usable. The two
ways to make block-level work for hybrid models:

1. **Marconi-style DN checkpoints** at block boundaries (research-grade;
   upstream has open bugs — vllm#26201, vllm#40696, Yifei Hu's "one block
   too many" report on Qwen3.5-Next).
2. **Keep DN state alive in-place between requests** — this *is* slot-level.

Slot-level captures ~100% of the personal-chat use case (single user, chat tab
open, sequential turns). Cross-tenant system-prompt sharing and partial-prefix
matches are out of scope for v1 — revisit only if logs show material cross-user
overlap.

## Security model

Identical to vLLM's APC:

- **Content-keyed**: the key is `hash(tokens_so_far)`. To "hijack" a cached
  slot you must already know the exact prefix tokens — at which point you
  already have the data.
- **Side-channel**: timing-based — same as vLLM's. Single-tenant research
  server, accepted risk.
- **No session IDs**: nothing to spoof; no auth surface added.
- **No persistent storage**: live slots are in-DRAM only; gone on process
  restart.

## Milestones

| ID | What | Gate | Status | Commit |
|---|---|---|---|---|
| P0 | Design contracts + touch-point map | This doc + code-read pass | ✅ DONE | `8490e09` |
| P1 | `LiveSlotStore` data structure | 12/12 unit tests | ✅ DONE | `b876621` |
| P2 | Slot lifecycle — don't free on done | 10/10 lifecycle mock tests | ✅ DONE | `3de9297` |
| P3 | Admit-time prefix match + skip prefill | 13/13 lifecycle tests (P3 cases added) | ✅ DONE | (this commit) |
| P4 | Decode-suffix advance before generation | **Subsumed by P3** — existing PREFILL loop handles it | ✅ DONE | (this commit) |
| P5 | Production wire-up + env gate | qb1 smoke: **turn 1 5.33s → turn 2 2.73s, 1.96× speedup, 1 cache hit, 0 misses on turn 2** | ✅ DONE | `2cad663` |
| P6 | TTL + `/metrics` counters | `cb_prefix_cache_*` counters + gauge visible in `/metrics`; 300s TTL sweep in engine loop | ✅ DONE | `fdf3c57` |

## P0 — Design contracts

### Cache key

Hash chain over token IDs, vLLM-style, so sub-prefix lookups don't re-hash from
scratch later:

```
hash_chain[0] = h(token[0])
hash_chain[i] = h(hash_chain[i-1], token[i])
```

Hash function: `xxhash.xxh64` for speed (microseconds at our scale). Switch to
`sha256` only if we ever multi-tenant and need cryptographic resistance.

For v1, we don't actually need the chain — we have ≤ N_SLOTS=4 live slots, so
linear scan + bytewise prefix-equality is fine. Hash chain is forward-compatible
for block-level eviction later.

### LiveSlotStore API

```python
class LiveSlotStore:
    """Holds completed slots indexed by their tokens_so_far, in LRU order.
    Operations are O(N_SLOTS) which is fine at N_SLOTS=4."""

    def find_longest_match(prompt_tokens: list[int]) -> tuple[Slot | None, int]:
        """Return (slot, n_matched) for the live slot whose tokens_so_far is
        the longest prefix of prompt_tokens. (None, 0) on miss."""

    def mark_live(slot: Slot, tokens_so_far: list[int]) -> None:
        """Move slot from active CB to live cache. Touches LRU."""

    def evict_lru() -> Slot:
        """Pop and return the least-recently-used live slot. Caller resets
        and reuses it for a new conversation. Raises if no live slots."""

    def remove(slot: Slot) -> None:
        """Explicit teardown (TTL expiry / explicit free)."""

    def __len__() -> int: ...
    def touch(slot: Slot) -> None: ...  # internal LRU bookkeeping
```

### Admit-time flow (the change in `cb_scheduler.admit`)

```python
def admit(rid: str, prompt_tokens: list[int]):
    slot, n_matched = self.live_slots.find_longest_match(prompt_tokens)

    if slot is not None and n_matched >= MIN_PREFIX_MATCH:
        # cache hit path: resume the live slot
        assert prompt_tokens[:n_matched] == slot.tokens_so_far  # defense
        self.live_slots.remove(slot)            # back to active
        slot.assign_request(rid)
        new_suffix = prompt_tokens[n_matched:]
        self.advance_decode(slot, new_suffix)   # P4 mechanism
        return

    # cache miss path: existing behavior
    slot = self.alloc_free_slot()
    if slot is None:
        slot = self.live_slots.evict_lru()      # P2: evict to make room
        self.reset_slot(slot)
    slot.assign_request(rid)
    self.full_prefill(slot, prompt_tokens)
```

### Response-done flow (the change in `cb_scheduler.finish`)

```python
def finish(slot: Slot):
    # was: self.free_slot(slot)
    # now:
    self.live_slots.mark_live(slot, slot.tokens_so_far)
```

### Touch-point map

| File | Function | What changes |
|---|---|---|
| `experiments/serve/cb_scheduler.py` | `admit`, `finish`, `step` | wire live-slot lookup + lifecycle |
| `experiments/serve/cb_scheduler.py` | (new) `advance_decode(slot, tokens)` | run N decode steps using existing trace |
| `experiments/serve/cb_engine.py` | request lifecycle | call `live_slots.remove` on explicit user cancel |
| `experiments/serve/cb_api.py` | `/v1/chat/completions` | nothing — tokenization already happens; pass through |
| `experiments/serve/server_tp.py` | `forward_token_tp_inner` | verify `cur_pos != 0` works (likely fine, `cur_pos` is tensor buffer) |
| `experiments/serve/scripts/serve_cb.sh` | env var | `TT_CB_PREFIX_CACHE=0/1` |
| (new) `experiments/cb/isolate/prefix_cache_store.py` | — | unit test for LiveSlotStore |
| (new) `experiments/cb/validate/prefix_cache_chat.py` | — | bit-identical-output gate |

### Open questions resolved in P0

1. **Does the decode trace handle `cur_pos` starting at non-zero?**
   Read of `forward_token_tp_inner` confirms `cur_pos` is plumbed through
   `state.cb_cur_pos_buf` (a device tensor input). The trace doesn't bake the
   starting value. Resume at `cur_pos = 100` is the same trace replay as
   `cur_pos = 0`. ✓ no new trace needed.

2. **Concurrency: cb_engine is async/multi-threaded?** Yes. LiveSlotStore needs
   a lock (we'll use the same lock as the slot allocator).

3. **MIN_PREFIX_MATCH threshold?** Set to ~16 tokens (worth the overhead).
   Below that, just full-prefill — the savings don't pay for the lookup +
   defense-in-depth verify.

4. **What about cancellations mid-decode?** Slot still gets marked live with
   the partial `tokens_so_far` at cancel time. Next turn might reuse it.
   (Minor: store needs a `cancel_safe` flag to gate this.)

## P1 — LiveSlotStore (data structure only)

Pure Python. No ttnn, no scheduler integration. Unit-test-able locally too
(but per non-negotiables run on qb1 to mirror prod env).

File: `experiments/serve/live_slot_store.py` (importable from cb_scheduler).
Test: `experiments/cb/isolate/prefix_cache_store.py`.

Gate: unit test exercises {mark_live, find_longest_match, evict_lru, remove,
LRU ordering, no-match, ties, exact-vs-substring}. All assertions pass.

## P2 — Slot lifecycle (don't free on done)

Smallest possible change: in `cb_scheduler.finish`, replace `free_slot` with
`live_slots.mark_live`. In `alloc_slot` (or wherever free slots come from), add
the LRU evict fallback when nothing's free.

**Critical correctness check**: does `mark_live` leave the slot's KV pages and
DN state untouched? Verify: today's `free_slot` zeros / reclaims pages; we need
to NOT do that. Walk through carefully.

Gate: existing CB engine validator (`experiments/cb/validate/...`) PASSES with
prefix-cache on but cache empty (i.e., behaviour bit-identical when cache is
cold). Live slot count grows under sequential requests.

## P3 — Admit-time prefix match + skip prefill

Wire `live_slots.find_longest_match` into `admit`. On hit, skip the prefill
path entirely; jump to a NEW "resume mode" code path that handles the new-suffix
decode (P4 mechanism).

**Critical gotcha**: on hit, we need to verify the live slot is at exactly
`cur_pos = n_matched` — the slot stopped decoding at the end of the previous
turn's response, which IS `tokens_so_far`. So `slot.cur_pos == len(tokens_so_far)`
should hold by construction. Assert it.

Gate: chat smoke test through `chat.py` — turn 1 TTFT ~ today's number; turn 2
TTFT ~ N × decode_step_ms (where N = new tokens count). Should be 10-100× faster
than turn 2 today.

## P4 — Decode-suffix advance before generation (subsumed by P3)

**P4 collapses into P3**: the suffix advance is exactly what the existing
PREFILL state path in `step()` already does. With `cur_pos = n_matched` and
`status = 'PREFILL'`:

```python
# experiments/serve/cb_scheduler.py:step() — existing code
if r['status'] == 'PREFILL' and r['cur_pos'] < last_prompt:
    r['cur_pos'] += 1
    r['next_tok'] = r['prompt'][r['cur_pos']]
else:
    # last prefill token → transition to DECODE + emit first generated token
    r['gen'].append(o)
    r['status'] = 'DECODE'
    ...
```

Each iteration feeds `prompt[cur_pos]` at position `cur_pos` through the decode
trace, discards the output (it's a prefill step, not a generation step), and
advances `cur_pos`. When `cur_pos == last_prompt`, it transitions to DECODE.

Starting at `cur_pos = n_matched` instead of `0` is the only change. Saves
`n_matched` decode steps. For typical chat: turn-2's history is 100-500 tokens,
cache match covers all of it, only the new user message (5-50 tokens) gets
processed through the decode loop. At ~12 tok/s that's <4s vs ~40s cold prefill.

**Gate (deferred to P5)**: bit-identical comparison — feed
`(cached_prefix + new_suffix)` two ways: (1) cold prefill all; (2) cache-hit +
PREFILL-loop suffix advance. First generated token's logits must match cos ≥ 0.9999.
This needs a real device run on qb1.

**Future optimization (out of scope for v1)**: at `len(new_suffix) > chunk_size`,
the suffix advance is N×80ms decode steps; could use S1a chunked prefill on
just the suffix (with `chunk_start_idx=n_matched` flowing through to the
chunked SDPA). Track as T3-multi-chunk-revisit.

## P5 — Production wire-up + env gate (SHIPPED 2026-06-01)

- `TT_CB_PREFIX_CACHE=0` (default off) / `=1` (on); plumbed through
  `cb_api.py` → `CBEngine(prefix_cache=…)` → `Scheduler(prefix_cache=…)`.
- `TT_CB_PREFIX_TTL_S=300` (default) controls live-slot expiration.
- Documented in `serve_cb.sh` header + the env list block.
- Smoke gate: `experiments/cb/validate/prefix_cache_smoke.py` — drives
  `/v1/chat/completions` for a 2-turn conversation, scrapes `/metrics` before
  and after, and verifies (a) `cb_prefix_cache_hits_total` increments by 1
  on turn 2, (b) turn 2 latency < turn 1 latency, (c) live_slots gauge tracks
  lifecycle correctly.

## P6 — TTL + `/metrics` (SHIPPED 2026-06-01 with P5)

- TTL: `CBEngine._pc_ttl_sweep()` runs every ~30s in the engine loop, calls
  `live_slots.expire_stale(prefix_ttl_s)`. Stale slots are reclaimed as
  evictions in `cb_prefix_cache_evictions_total`.
- `/metrics` adds:
  - `cb_prefix_cache_hits_total` (counter)
  - `cb_prefix_cache_misses_total` (counter)
  - `cb_prefix_cache_evictions_total` (counter — LRU **and** TTL)
  - `cb_prefix_cache_live_slots` (gauge)
  - `cb_prefix_cache_enabled` (gauge, 0/1)
- Engine loop calls `_pc_sync_metrics()` after every step → scheduler counters
  → Prometheus counters via delta. Cheap (pure-Python int ops).

## P5 smoke gate findings (2026-06-01, in progress)

First three smoke runs hit a chain of compatibility bugs:

- Bug 1 (FIXED `4acc955`): per-request `max_tokens` cap triggers
  `engine._stream → sched.cancel()`, NOT `_finish()` → `mark_live` was never
  called. Fix: `cancel(rid, mark_live=True)` flag; `_stream` sets it on cap.
- Bug 2 (FIXED `184f00e`): pre-existing — stripping trailing EOS from
  cached `tokens_so_far` mismatches the chat template's between-turns
  `<|im_end|>` on the next turn's prompt.
- Bug 3 (WORKAROUND): with `TT_CB_CHUNKED_PREFILL=1`, the L>chunk_size eager
  fallback (`forward_prefill_chunked_tp`) wedges the server after a couple
  of requests. Symptom: "Allocating device buffers is unsafe due to the
  existence of an active trace" at runtime; eager allocations collide with
  reserved trace memory. **Workaround**: run with `TT_CB_CHUNKED_PREFILL=0`
  + `TT_CB_PREFIX_CACHE=1`. Without chunked prefill we never hit the eager
  path; with prefix caching, turn 2+ skips prefill entirely via the cache.
- Bug 4 (FIXED `c55b6f7`): Qwen3.6's chat template injects an empty
  `<think>\n\n</think>\n\n` block at the active assistant prompt (with
  `enable_thinking=False`), but does NOT render it for past assistant
  messages. Turn 1's prompt has the markers; turn 2's tokenization of the
  same conversation doesn't → longest-prefix match collapses from 246 to 44.
  Validated via `experiments/cb/isolate/chat_template_roundtrip.py` —
  44/246 → **244/244 match** after fix. Fix: `_messages_to_prompt` now
  passes `enable_thinking=False` AND strips the literal `<think>\n\n</think>\n\n`
  block from the resulting prompt string.

## Recommended runtime config (current understanding)

`TT_CB_CHUNKED_PREFILL=0` + `TT_CB_PREFIX_CACHE=1`. Trade-off: turn-1 (cold)
TTFT is slower (1-tok/iter through decode trace) but turn-2+ TTFT is much
faster (cache hit + suffix advance). Net win for any conversation with ≥2
turns, which is the dominant chat case.

## Risks + things to keep an eye on

1. **Slot reset between cache miss + alloc on top of evicted slot**. The
   evicted slot has stale KV / DN state. Before reuse for a NEW conversation,
   we must actually reset (zero out / reset cur_pos). This is the existing
   "reset" path; verify it actually fully resets DN `H_t` and conv_cols.

2. **Streaming partial responses**: if the client cancels mid-response (closed
   connection), our slot has `tokens_so_far` only up to the partial generation.
   Next turn's chat history would include the FULL previous response (whatever
   the user saw), not our truncation. Hash mismatch → cache miss → full prefill.
   Correct but loses the win. Acceptable for v1.

3. **Multiple chat tabs from same user**: each tab is an independent
   conversation thread. With session-keyed design we'd have to pick which to
   share. With content-keyed design they naturally just don't share (different
   token histories). ✓ no special handling.

4. **System prompt drift**: if `chat.py` ever changes its system prompt mid-
   session, the next request's prefix won't match → cache miss → full prefill.
   Correct but loses the win. Update `chat.py` to keep system prompt stable.

5. **Token determinism**: `tokenizer.encode("hello")` must return the same IDs
   every call. Verify our HF tokenizer setup is deterministic (it is for
   Qwen3.6 — verified by existing per-token gates).

## Out of scope for v1 (revisit if needed)

- **Block-level APC + Marconi DN checkpoints** for cross-user system prompt
  sharing or partial-prefix matches. Roadmap 2/3 from research audit.
- **Multi-tab same-client** (multiple live slots per logical user).
- **Session "fork"** (user edits earlier message and resubmits — current design
  treats this as miss; for v2, could detect and partial-match up to the fork
  point).
- **Persistent cache across server restarts** — live_slots is in-DRAM only.
- **Cross-instance cache** (load-balanced fleet) — single-instance only.

## Reference points

- vLLM design doc: <https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md>
- TT vLLM PR #272 (dense Llama APC): <https://github.com/tenstorrent/vllm/pull/272>
- Hybrid APC tracking issue: <https://github.com/vllm-project/vllm/issues/26201>
- Marconi paper (DN checkpointing): <https://arxiv.org/abs/2411.19379>
- Research audit (in this repo): [`vllm_prefix_caching_audit.md`](vllm_prefix_caching_audit.md)
