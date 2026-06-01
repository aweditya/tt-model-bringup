"""LiveSlotStore — slot-level prefix cache for CB scheduler.

Holds completed slots indexed by hash(tokens_so_far) in LRU order. When a new
request's prompt has a cached prefix, the matching slot is reclaimed at
cur_pos = len(matched_prefix) — no re-prefill of the matched history.

Design rationale + threat model in research/27b_prefix_caching_plan.md.

The store does NOT touch any TTNN state. It is a pure-Python bookkeeping
structure; the scheduler calls into the model (cb_reset_slots / advance_decode)
based on the lookup result.

Operations are O(N_SLOTS) because N_SLOTS is small (<= 32 in production). Hash
chaining is forward-compatible for block-level eviction later.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class LiveEntry:
    slot_id: int
    tokens_so_far: tuple[int, ...]
    last_touch: float = field(default_factory=time.monotonic)


class LiveSlotStore:
    """Slot-level LRU cache keyed by full tokens_so_far.

    Two views over the same set of live entries:
      _by_slot: slot_id -> LiveEntry
      _lru: OrderedDict[slot_id, None] in least-recently-used-first order
    """

    def __init__(self, min_match_tokens: int = 16):
        self._by_slot: dict[int, LiveEntry] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self.min_match_tokens = min_match_tokens

    def __len__(self) -> int:
        return len(self._by_slot)

    def __contains__(self, slot_id: int) -> bool:
        return slot_id in self._by_slot

    def mark_live(self, slot_id: int, tokens_so_far: list[int] | tuple[int, ...]) -> None:
        """Insert (or refresh) a live entry for slot_id with the given prefix.
        Moves it to the most-recently-used position."""
        entry = LiveEntry(slot_id=slot_id, tokens_so_far=tuple(tokens_so_far))
        self._by_slot[slot_id] = entry
        self._lru.pop(slot_id, None)
        self._lru[slot_id] = None  # MRU = end of OrderedDict

    def find_longest_match(self,
                           prompt_tokens: list[int] | tuple[int, ...]
                           ) -> tuple[int | None, int]:
        """Return (slot_id, n_matched) for the live slot whose tokens_so_far is
        the longest prefix of prompt_tokens, or (None, 0) on miss.

        A "match" requires both:
          - tokens_so_far is a prefix of prompt_tokens (exact, all tokens equal)
          - len(tokens_so_far) >= min_match_tokens (below this, not worth it)

        Ties broken by most-recently-used (later in self._lru wins).
        """
        prompt = tuple(prompt_tokens)
        L = len(prompt)
        best_slot: int | None = None
        best_n = 0
        # Iterate in MRU-first order so ties prefer recent
        for slot_id in reversed(self._lru):
            entry = self._by_slot[slot_id]
            n = len(entry.tokens_so_far)
            if n < self.min_match_tokens:
                continue
            if n > L:
                continue
            if prompt[:n] == entry.tokens_so_far:
                if n > best_n:
                    best_slot = slot_id
                    best_n = n
        return best_slot, best_n

    def touch(self, slot_id: int) -> None:
        """Mark slot as recently used (move to MRU). Caller does this when
        reclaiming the slot after a cache hit, even though it's about to be
        removed — keeps LRU honest if the reclaim is aborted."""
        if slot_id not in self._lru:
            return
        self._lru.move_to_end(slot_id)
        self._by_slot[slot_id].last_touch = time.monotonic()

    def reclaim(self, slot_id: int) -> LiveEntry:
        """Remove slot from the live cache. Returns the entry. Caller now owns
        the slot for re-allocation. Raises KeyError if slot wasn't live."""
        entry = self._by_slot.pop(slot_id)
        self._lru.pop(slot_id, None)
        return entry

    def evict_lru(self) -> int:
        """Pop the least-recently-used slot. Returns slot_id; caller resets +
        reallocates. Raises LookupError if empty."""
        if not self._lru:
            raise LookupError("LiveSlotStore is empty; nothing to evict")
        slot_id, _ = next(iter(self._lru.items()))
        del self._by_slot[slot_id]
        del self._lru[slot_id]
        return slot_id

    def expire_stale(self, ttl_seconds: float) -> list[int]:
        """Remove any entry older than ttl_seconds. Returns the freed slot_ids."""
        now = time.monotonic()
        freed: list[int] = []
        for slot_id, entry in list(self._by_slot.items()):
            if now - entry.last_touch > ttl_seconds:
                del self._by_slot[slot_id]
                self._lru.pop(slot_id, None)
                freed.append(slot_id)
        return freed

    def slot_ids(self) -> list[int]:
        """Currently-live slot ids in LRU order (oldest first)."""
        return list(self._lru.keys())
