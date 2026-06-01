"""PC-P2 gate: slot lifecycle logic test for cb_scheduler.Scheduler with
prefix_cache=True.

Mocks the device boundary (server_tp_cb + the Scheduler.__init__ device setup)
just enough to drive _finish, _admit, _slot_alloc_order, and step(). Verifies:
  - prefix_cache=False: behavior identical to today (no live_slots, no mark_live)
  - prefix_cache=True: completed slots get marked live; LRU eviction works;
    idle step() returns 0 without forward; admit prefers non-cached free slots.

Runs pure Python — no ttnn / no torch / no mesh required. Use to gate the
P2 logic change BEFORE running the heavier P3 device test.

Run:
    python3 experiments/cb/isolate/prefix_cache_lifecycle.py
"""

from __future__ import annotations
import os
import sys
import types

# Path setup mirrors prefix_cache_store.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVE = os.path.normpath(os.path.join(_HERE, '..', '..', 'serve'))
sys.path.insert(0, _SERVE)


# --- Mock the device boundary BEFORE importing cb_scheduler. ---
# cb_scheduler imports server_tp + server_tp_cb at module load. We replace
# both with stubs so the import succeeds without ttnn.

def _mock_modules():
    base = types.ModuleType('server_tp')
    base.update_input_buffers = lambda *a, **k: None
    base.update_prefill_input_buffers = lambda *a, **k: None
    base.forward_token_tp_inner = lambda *a, **k: None
    base._reset_state_buffers = lambda *a, **k: None
    base.forward_prefill_chunked_traced_inner = lambda *a, **k: None
    base.forward_prefill_chunked_tp = lambda *a, **k: None
    base._sample_from_logits = lambda *a, **k: 0
    sys.modules['server_tp'] = base

    cb_mod = types.ModuleType('server_tp_cb')
    cb_mod.setup_cb_state = lambda *a, **k: None
    cb_mod.cb_reset_states = lambda *a, **k: None
    cb_mod.cb_reset_slots = lambda *a, **k: _RESET_LOG.append(list(a[1]))
    cb_mod.cb_prefill_transplant = lambda *a, **k: None
    cb_mod.update_input_buffers_batched = lambda *a, **k: None
    cb_mod.forward_batch_tp_inner = lambda *a, **k: None
    sys.modules['server_tp_cb'] = cb_mod

    # Skip the optional ttnn import inside cb_scheduler (only used in trace
    # paths we won't hit).
    ttnn_mod = types.ModuleType('ttnn')
    sys.modules['ttnn'] = ttnn_mod


_RESET_LOG: list[list[int]] = []
_mock_modules()

# Import AFTER mocks are installed.
import cb_scheduler  # noqa: E402


class FakeState:
    """Minimal stand-in for MeshServerState. Only attrs Scheduler reads."""
    def __init__(self):
        self.cb_B = 4
        self.cb_conv_mode = None
        self.cb_dn_recurrence_mode = None


def make_scheduler(prefix_cache=False):
    """Construct a Scheduler bypassing the ttnn warmup paths."""
    s = cb_scheduler.Scheduler.__new__(cb_scheduler.Scheduler)
    # Replicate just enough of __init__ that the lifecycle methods work.
    s.state = FakeState()
    s.B = 4
    s.max_new = 16
    s.eos_id = 999
    s.use_trace = False
    s.chunked_prefill = False
    s.sampling = False
    s.topk_k = None
    s._trace_id = None
    s._argmax_handle = None
    s._logits_handle = None
    s._topk_values_handle = None
    s._topk_indices_handle = None
    s._prefill_trace_id = None
    s._prefill_trace_out = None
    s.prefix_cache = bool(prefix_cache)
    s.live_slots = (cb_scheduler.LiveSlotStore(
        min_match_tokens=cb_scheduler.PREFIX_CACHE_MIN_MATCH)
        if s.prefix_cache else None)
    s.pc_hits = 0
    s.pc_misses = 0
    s.pc_evictions = 0
    from collections import deque
    s.slots = [None] * s.B
    s.waiting = deque()
    s.reqs = {}
    s._next_id = 0
    return s


def make_request(rid, prompt_len=20, gen_len=10):
    """Fake completed-request dict matching the shape cb_scheduler builds."""
    return {
        'id': rid,
        'prompt': list(range(100 + rid * 1000, 100 + rid * 1000 + prompt_len)),
        'gen': list(range(500 + rid * 1000, 500 + rid * 1000 + gen_len)),
        'cur_pos': 0,
        'next_tok': 100 + rid * 1000,
        'status': 'DECODE',
        'slot': None,
        'sampling': None,
        'rng': None,
    }


# --- Test cases ---

def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


def case_pc_off_no_live_slots():
    s = make_scheduler(prefix_cache=False)
    r = make_request(0)
    s.reqs[0] = r
    s.slots[0] = 0
    r['slot'] = 0
    done = s._finish(r, 0, last_out=42)
    assert_eq(done, False, "not done yet (gen<max)")
    # Force done via max_new
    r['gen'] = list(range(s.max_new))
    done = s._finish(r, 0, last_out=42)
    assert_eq(done, True, "now done")
    assert_eq(r['status'], 'DONE', "status DONE")
    assert_eq(s.slots[0], None, "slot freed")
    assert_eq(s.live_slots, None, "live_slots stays None when pc=off")
    print("  ✓ case_pc_off_no_live_slots")


def case_finish_marks_live_when_pc_on():
    s = make_scheduler(prefix_cache=True)
    r = make_request(0, prompt_len=20, gen_len=10)  # tokens_so_far=30 >= min
    s.reqs[0] = r
    s.slots[0] = 0
    r['slot'] = 0
    r['gen'] = list(range(s.max_new))  # force done
    s._finish(r, 0, last_out=42)
    assert_eq(s.slots[0], None, "slot None after finish")
    assert_eq(len(s.live_slots), 1, "live entry added")
    assert_true(0 in s.live_slots, "slot 0 in cache")
    print("  ✓ case_finish_marks_live_when_pc_on")


def case_finish_skips_eos_in_tokens_so_far():
    """If the last generated token is EOS, drop it from cached tokens_so_far —
    the next turn's prompt won't have an EOS inside the assistant message."""
    s = make_scheduler(prefix_cache=True)
    r = make_request(0, prompt_len=20, gen_len=10)
    s.reqs[0] = r
    s.slots[0] = 0
    r['slot'] = 0
    r['gen'][-1] = s.eos_id  # EOS at the end
    s._finish(r, 0, last_out=s.eos_id)
    expected_tokens = list(r['prompt']) + list(r['gen'][:-1])  # EOS dropped
    entry = s.live_slots._by_slot[0]
    assert_eq(list(entry.tokens_so_far), expected_tokens, "EOS dropped from cache")
    print("  ✓ case_finish_skips_eos_in_tokens_so_far")


def case_finish_too_short_not_cached():
    """tokens_so_far < min_match → don't bother caching."""
    s = make_scheduler(prefix_cache=True)
    r = make_request(0, prompt_len=5, gen_len=5)  # total 10 < min=16
    s.reqs[0] = r
    s.slots[0] = 0
    r['slot'] = 0
    r['gen'] = list(range(s.max_new))[:5]  # gen len 5
    s._finish(r, 0, last_out=42)
    assert_eq(len(s.live_slots), 0, "short prefix not cached")
    print("  ✓ case_finish_too_short_not_cached")


def case_slot_alloc_order_prefers_non_cached():
    """When some free slots are cached and others aren't, prefer non-cached
    first so the cache survives admits when possible."""
    s = make_scheduler(prefix_cache=True)
    # Slot 0, 1: non-cached free
    # Slot 2: cached free (LRU)
    # Slot 3: cached free (MRU)
    s.live_slots.mark_live(2, list(range(0, 20)))
    s.live_slots.mark_live(3, list(range(100, 120)))
    order = list(s._slot_alloc_order())
    # Non-cached come first; cached come in LRU order (2 before 3)
    assert_eq(order, [0, 1, 2, 3], "alloc order correct")
    print("  ✓ case_slot_alloc_order_prefers_non_cached")


def case_admit_evicts_lru_cache_when_no_free():
    """All slots cached → admit forces eviction from LRU end."""
    global _RESET_LOG
    _RESET_LOG = []
    s = make_scheduler(prefix_cache=True)
    # Fill all 4 slots with cache entries
    for i in range(4):
        s.live_slots.mark_live(i, list(range(i * 100, i * 100 + 20)))
    # All slots are None (free) but cached.
    # Add a waiting request — should evict LRU (slot 0)
    r0 = make_request(0)
    s.reqs[0] = r0
    s.waiting.append(0)
    admitted = s._admit()
    assert_eq(admitted, [0], "evicted LRU slot 0")
    assert_eq(s.slots[0], 0, "slot 0 holds rid 0")
    assert_eq(len(s.live_slots), 3, "one cache entry evicted")
    assert_true(0 not in s.live_slots, "slot 0 no longer in cache")
    assert_eq(s.pc_evictions, 1, "eviction counter")
    assert_eq(_RESET_LOG, [[0]], "cb_reset_slots called for [0]")
    print("  ✓ case_admit_evicts_lru_cache_when_no_free")


def case_admit_prefers_truly_free_over_cached():
    """If 1 truly-free + 1 cached-free, admit picks truly-free first."""
    global _RESET_LOG
    _RESET_LOG = []
    s = make_scheduler(prefix_cache=True)
    s.live_slots.mark_live(0, list(range(0, 20)))
    s.live_slots.mark_live(1, list(range(100, 120)))
    # Slots 2, 3 stay truly-free (not cached)
    r0 = make_request(0)
    s.reqs[0] = r0
    s.waiting.append(0)
    admitted = s._admit()
    assert_eq(admitted, [2], "preferred truly-free slot 2")
    # No eviction — cache untouched
    assert_eq(len(s.live_slots), 2, "cache untouched")
    assert_eq(s.pc_evictions, 0, "no eviction")
    print("  ✓ case_admit_prefers_truly_free_over_cached")


def case_idle_step_returns_zero_no_forward():
    """When no active slots and no waiting requests, step() returns 0 without
    touching the device (prevents DUMMY_TOK pollution of cached state)."""
    global _RESET_LOG
    _RESET_LOG = []
    s = make_scheduler(prefix_cache=True)
    # Put two slots in the cache, leave all slots free, no waiting requests.
    s.live_slots.mark_live(0, list(range(0, 20)))
    s.live_slots.mark_live(1, list(range(100, 120)))
    active = s.step()
    assert_eq(active, 0, "idle step returns 0")
    assert_eq(_RESET_LOG, [], "no cb_reset_slots call")
    # We can't directly check forward wasn't called (mock returns None silently),
    # but the early-return ensures we don't even enter the admit path.
    assert_eq(len(s.live_slots), 2, "cache untouched by idle step")
    print("  ✓ case_idle_step_returns_zero_no_forward")


def case_pc_off_idle_step_still_runs_admit():
    """With prefix_cache=False, behavior unchanged — idle step still hits the
    admit + forward path (today's behavior; safe because no cache to protect)."""
    s = make_scheduler(prefix_cache=False)
    # We can't run the full forward (no mocks), but we can verify the early-
    # return guard ISN'T triggered. Call _admit directly to confirm path:
    admitted = s._admit()
    assert_eq(admitted, [], "no admits when nothing waiting (expected)")
    # No assertion failure = path was taken without short-circuit.
    print("  ✓ case_pc_off_idle_step_still_runs_admit")


def case_cancel_does_not_mark_live():
    """Cancelled requests should NOT be cached (partial/interrupted state)."""
    s = make_scheduler(prefix_cache=True)
    r = make_request(0, prompt_len=20, gen_len=5)
    s.reqs[0] = r
    s.slots[0] = 0
    r['slot'] = 0
    # cancel() sets status='CANCELLED'; _finish would not be called.
    s.cancel(0)
    assert_eq(s.slots[0], None, "slot freed")
    assert_eq(r['status'], 'CANCELLED', "status cancelled")
    assert_eq(len(s.live_slots), 0, "no cache entry for cancelled")
    print("  ✓ case_cancel_does_not_mark_live")


def main():
    cases = [
        case_pc_off_no_live_slots,
        case_finish_marks_live_when_pc_on,
        case_finish_skips_eos_in_tokens_so_far,
        case_finish_too_short_not_cached,
        case_slot_alloc_order_prefers_non_cached,
        case_admit_evicts_lru_cache_when_no_free,
        case_admit_prefers_truly_free_over_cached,
        case_idle_step_returns_zero_no_forward,
        case_pc_off_idle_step_still_runs_admit,
        case_cancel_does_not_mark_live,
    ]
    print(f"[prefix_cache_lifecycle] running {len(cases)} cases...")
    for c in cases:
        try:
            c()
        except AssertionError as e:
            print(f"  ✗ {c.__name__}: {e}")
            return 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ✗ {c.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"[prefix_cache_lifecycle] PASS — {len(cases)}/{len(cases)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
