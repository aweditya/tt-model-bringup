"""Unit test for experiments/serve/live_slot_store.py.

PC-P1 gate. Pure Python — no ttnn, no torch. Runs locally or on qb1.

Run:
    python experiments/cb/isolate/prefix_cache_store.py

Exits 0 on success, 1 on any assertion failure.
"""

from __future__ import annotations
import sys
import time
import os

# Make the serve package importable from this file's location:
#   .../experiments/cb/isolate/prefix_cache_store.py
#   .../experiments/serve/live_slot_store.py
_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../cb/isolate
_SERVE = os.path.normpath(os.path.join(_HERE, '..', '..', 'serve'))  # .../experiments/serve
sys.path.insert(0, _SERVE)

from live_slot_store import LiveSlotStore


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


def case_empty_store():
    s = LiveSlotStore(min_match_tokens=4)
    assert_eq(len(s), 0, "empty len")
    slot, n = s.find_longest_match([1, 2, 3, 4, 5])
    assert_eq(slot, None, "empty find returns None")
    assert_eq(n, 0, "empty find n=0")
    try:
        s.evict_lru()
        raise AssertionError("evict_lru on empty should raise")
    except LookupError:
        pass
    print("  ✓ case_empty_store")


def case_mark_live_then_find():
    s = LiveSlotStore(min_match_tokens=4)
    s.mark_live(0, [10, 20, 30, 40, 50])
    s.mark_live(1, [10, 20, 30, 40, 60, 70])
    assert_eq(len(s), 2, "len=2")
    # Prompt matches slot 1 exactly (6 toks), slot 0 differs at pos 4
    slot, n = s.find_longest_match([10, 20, 30, 40, 60, 70, 80])
    assert_eq(slot, 1, "longest match slot")
    assert_eq(n, 6, "longest match n")
    # Prompt matches slot 0 (5 toks); slot 1 doesn't match
    slot, n = s.find_longest_match([10, 20, 30, 40, 50, 99])
    assert_eq(slot, 0, "slot 0 match")
    assert_eq(n, 5, "slot 0 n")
    print("  ✓ case_mark_live_then_find")


def case_min_match_threshold():
    s = LiveSlotStore(min_match_tokens=8)
    s.mark_live(0, [1, 2, 3, 4])  # 4 tokens; below threshold
    slot, n = s.find_longest_match([1, 2, 3, 4, 5, 6, 7, 8])
    assert_eq(slot, None, "below threshold not matched")
    assert_eq(n, 0, "below threshold n=0")
    print("  ✓ case_min_match_threshold")


def case_no_match():
    s = LiveSlotStore(min_match_tokens=4)
    s.mark_live(0, [10, 20, 30, 40, 50])
    slot, n = s.find_longest_match([99, 98, 97, 96, 95])
    assert_eq(slot, None, "no match")
    assert_eq(n, 0, "no match n=0")
    print("  ✓ case_no_match")


def case_prompt_shorter_than_entry():
    """Cache holds 10 tokens; prompt is only 5. Slot's tokens_so_far is NOT a
    prefix of prompt (it's longer than prompt). Must be a miss — we can't
    resume the slot at a position past what the user asked for."""
    s = LiveSlotStore(min_match_tokens=4)
    s.mark_live(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    slot, n = s.find_longest_match([1, 2, 3, 4, 5])
    assert_eq(slot, None, "prompt-shorter is miss")
    assert_eq(n, 0, "prompt-shorter n=0")
    print("  ✓ case_prompt_shorter_than_entry")


def case_mru_tie_break():
    """Two entries with same length. MRU wins."""
    s = LiveSlotStore(min_match_tokens=4)
    s.mark_live(0, [1, 2, 3, 4, 5])  # added first
    s.mark_live(1, [1, 2, 3, 4, 5])  # added second (more recent)
    slot, n = s.find_longest_match([1, 2, 3, 4, 5, 99])
    assert_eq(slot, 1, "MRU wins tie")
    assert_eq(n, 5, "MRU tie n")
    # Touch slot 0 → now it's MRU
    s.touch(0)
    slot, n = s.find_longest_match([1, 2, 3, 4, 5, 99])
    assert_eq(slot, 0, "after touch, slot 0 is MRU")
    print("  ✓ case_mru_tie_break")


def case_longest_beats_mru():
    """If a less-recent slot has a longer prefix, longest wins over MRU."""
    s = LiveSlotStore(min_match_tokens=4)
    s.mark_live(0, [1, 2, 3, 4, 5, 6, 7, 8])  # length 8, older
    s.mark_live(1, [1, 2, 3, 4])               # length 4, newer
    slot, n = s.find_longest_match([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert_eq(slot, 0, "longest beats MRU")
    assert_eq(n, 8, "longest n")
    print("  ✓ case_longest_beats_mru")


def case_evict_lru():
    s = LiveSlotStore(min_match_tokens=4)
    s.mark_live(0, [10, 20, 30, 40])
    s.mark_live(1, [50, 60, 70, 80])
    s.mark_live(2, [90, 91, 92, 93])
    # LRU = slot 0
    victim = s.evict_lru()
    assert_eq(victim, 0, "first evict = slot 0")
    assert_eq(len(s), 2, "len after evict")
    # Touch slot 1 (now MRU); next LRU eviction should be slot 2
    s.touch(1)
    victim = s.evict_lru()
    assert_eq(victim, 2, "after touch slot 1, next evict = slot 2")
    print("  ✓ case_evict_lru")


def case_reclaim():
    s = LiveSlotStore(min_match_tokens=4)
    s.mark_live(7, [1, 2, 3, 4, 5])
    entry = s.reclaim(7)
    assert_eq(entry.slot_id, 7, "reclaim returns entry")
    assert_eq(entry.tokens_so_far, (1, 2, 3, 4, 5), "reclaim tokens")
    assert_eq(len(s), 0, "reclaim removes")
    try:
        s.reclaim(7)
        raise AssertionError("double-reclaim should raise")
    except KeyError:
        pass
    print("  ✓ case_reclaim")


def case_mark_live_refresh():
    """mark_live on an existing slot updates tokens_so_far and bumps LRU."""
    s = LiveSlotStore(min_match_tokens=4)
    s.mark_live(0, [1, 2, 3, 4])
    s.mark_live(1, [5, 6, 7, 8])
    s.mark_live(2, [9, 10, 11, 12])
    # Re-mark slot 0 with longer tokens — should be MRU now
    s.mark_live(0, [1, 2, 3, 4, 5, 6])
    assert_eq(len(s), 3, "refresh keeps count")
    assert_eq(s.slot_ids(), [1, 2, 0], "slot 0 is now MRU")
    # New tokens should be matched
    slot, n = s.find_longest_match([1, 2, 3, 4, 5, 6, 7, 8])
    assert_eq(slot, 0, "refreshed tokens matched")
    assert_eq(n, 6, "refreshed n")
    print("  ✓ case_mark_live_refresh")


def case_ttl_expire():
    s = LiveSlotStore(min_match_tokens=4)
    s.mark_live(0, [1, 2, 3, 4])
    time.sleep(0.05)
    s.mark_live(1, [5, 6, 7, 8])
    freed = s.expire_stale(ttl_seconds=0.025)
    assert_eq(freed, [0], "only slot 0 stale")
    assert_eq(len(s), 1, "after TTL")
    # slot 1 should still be there
    assert 1 in s, "slot 1 retained"
    print("  ✓ case_ttl_expire")


def case_chat_turn2_simulation():
    """End-to-end chat-shaped scenario:
       turn 1: 25 prompt + 50 generated = 75 tokens_so_far
       turn 2: same 75 + 8 new user tokens = 83-token prompt
       Should find slot=0, n=75."""
    s = LiveSlotStore(min_match_tokens=4)
    turn1_tokens = list(range(100, 175))  # 75 tokens
    s.mark_live(0, turn1_tokens)
    turn2_prompt = turn1_tokens + [200, 201, 202, 203, 204, 205, 206, 207]
    slot, n = s.find_longest_match(turn2_prompt)
    assert_eq(slot, 0, "chat turn 2 finds slot 0")
    assert_eq(n, 75, "chat turn 2 matches full history")
    suffix_len = len(turn2_prompt) - n
    assert_eq(suffix_len, 8, "chat suffix = 8 new tokens")
    print("  ✓ case_chat_turn2_simulation")


def main():
    cases = [
        case_empty_store,
        case_mark_live_then_find,
        case_min_match_threshold,
        case_no_match,
        case_prompt_shorter_than_entry,
        case_mru_tie_break,
        case_longest_beats_mru,
        case_evict_lru,
        case_reclaim,
        case_mark_live_refresh,
        case_ttl_expire,
        case_chat_turn2_simulation,
    ]
    print(f"[prefix_cache_store] running {len(cases)} cases...")
    for c in cases:
        try:
            c()
        except AssertionError as e:
            print(f"  ✗ {c.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  ✗ {c.__name__}: unexpected {type(e).__name__}: {e}")
            return 1
    print(f"[prefix_cache_store] PASS — {len(cases)}/{len(cases)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
