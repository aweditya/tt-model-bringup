#!/usr/bin/env python3
"""Pure-host unit tests for spec-dec scheduler logic.

No device, no ttnn. Validates the host-side accept walk semantics.
The off-by-one cache-advance bug (commit 5aa1550) lived alongside
correct accept-walk code — these tests pin down the accept-walk
contract so future refactors stay honest.

Adds regression gates for the four obvious accept-walk shapes plus
edge cases (K=1, all reject, all accept, mid-mismatch). New cases
should be added here whenever a real-world scheduler bug surfaces.

Run on qb1 or qb2 (no device, but keeps us remote-only-strict):
  ssh qb1 'cd ~/tt-xla && .venv/bin/python experiments/utils/spec_dec_unit_tests.py'
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

# Import the SpecDecScheduler class without triggering __init__ (which
# requires target/drafter states). We construct via __new__ + set K.
from spec_dec_scheduler import SpecDecScheduler  # noqa: E402


def _walk(K, draft, target_kp1):
    """Helper: construct minimal scheduler and run the accept walk."""
    sched = SpecDecScheduler.__new__(SpecDecScheduler)
    sched.K = K
    return sched._accept_walk(list(draft), list(target_kp1))


class TestAcceptWalk(unittest.TestCase):
    """Cases mirror HF candidate_generator accept walk semantics."""

    # ── K=3 cases ───────────────────────────────────────────────────

    def test_all_accepted_with_bonus(self):
        """All K drafts match target's predictions → emit K + 1 bonus."""
        emitted, n = _walk(K=3, draft=[10, 20, 30],
                            target_kp1=[10, 20, 30, 42])
        self.assertEqual(emitted, [10, 20, 30, 42])
        self.assertEqual(n, 3)

    def test_all_rejected(self):
        """First draft mismatches → emit target's correction only."""
        emitted, n = _walk(K=3, draft=[10, 20, 30],
                            target_kp1=[99, 88, 77, 66])
        self.assertEqual(emitted, [99])
        self.assertEqual(n, 0)

    def test_mid_mismatch_after_two_accepts(self):
        """drafts 0+1 match, draft 2 doesn't → emit 2 accepts + correction."""
        emitted, n = _walk(K=3, draft=[10, 20, 30],
                            target_kp1=[10, 20, 99, 88])
        self.assertEqual(emitted, [10, 20, 99])
        self.assertEqual(n, 2)

    def test_single_accept(self):
        """Only draft 0 matches."""
        emitted, n = _walk(K=3, draft=[10, 20, 30],
                            target_kp1=[10, 99, 88, 77])
        self.assertEqual(emitted, [10, 99])
        self.assertEqual(n, 1)

    # ── K=1 edge cases ─────────────────────────────────────────────

    def test_k1_accept(self):
        emitted, n = _walk(K=1, draft=[10], target_kp1=[10, 42])
        self.assertEqual(emitted, [10, 42])
        self.assertEqual(n, 1)

    def test_k1_reject(self):
        emitted, n = _walk(K=1, draft=[10], target_kp1=[99, 88])
        self.assertEqual(emitted, [99])
        self.assertEqual(n, 0)

    # ── K=5 (production default) ───────────────────────────────────

    def test_k5_all_accept(self):
        emitted, n = _walk(K=5,
                            draft=[1, 2, 3, 4, 5],
                            target_kp1=[1, 2, 3, 4, 5, 6])
        self.assertEqual(emitted, [1, 2, 3, 4, 5, 6])
        self.assertEqual(n, 5)

    def test_k5_partial_3(self):
        emitted, n = _walk(K=5,
                            draft=[1, 2, 3, 4, 5],
                            target_kp1=[1, 2, 3, 99, 88, 77])
        self.assertEqual(emitted, [1, 2, 3, 99])
        self.assertEqual(n, 3)

    # ── Contract sanity ───────────────────────────────────────────

    def test_emit_count_is_accept_plus_one(self):
        """len(emitted) MUST equal accept_count + 1 by construction.
        The scheduler relies on this invariant for cur_pos advance.
        """
        for K in (1, 2, 3, 5, 7):
            for accept in range(K + 1):
                draft = list(range(K))
                # Match first `accept` drafts, mismatch the next.
                tgt = list(draft[:accept]) + [99] * (K + 1 - accept)
                emitted, n = _walk(K=K, draft=draft, target_kp1=tgt)
                self.assertEqual(
                    n, accept,
                    f"K={K} accept={accept}: got n={n}")
                self.assertEqual(
                    len(emitted), accept + 1,
                    f"K={K} accept={accept}: len(emitted)={len(emitted)} "
                    f"expected {accept+1}")

    def test_returned_tokens_are_python_ints(self):
        """Cache-advance loop does `int(tok)` so this is mostly defensive,
        but the contract is host ints, not numpy scalars or torch tensors.
        """
        emitted, n = _walk(K=3, draft=[10, 20, 30],
                            target_kp1=[10, 99, 88, 77])
        for tok in emitted:
            self.assertIsInstance(
                tok, int,
                f"emitted tok {tok!r} should be Python int, got {type(tok)}")

    def test_target_bonus_used_only_on_full_accept(self):
        """`target_argmaxes_kp1[K]` is the K+1th row, used only when all
        K drafts accepted. On mid-rejection we use target[i] (the
        correction at the rejection point), not target[K]."""
        # Bonus would be 42; mid-reject should NOT include 42.
        emitted, _ = _walk(K=3, draft=[1, 2, 3],
                            target_kp1=[1, 99, 88, 42])
        self.assertNotIn(42, emitted)
        # Confirm 99 (the correction at row 1) IS used.
        self.assertEqual(emitted, [1, 99])


class TestAcceptWalkInputContract(unittest.TestCase):
    """target_argmaxes_kp1 has K+1 entries by contract (verify trace
    produces K+1 rows). draft_tokens has K entries. Off-by-one in the
    caller would surface here."""

    def test_target_kp1_must_have_k_plus_one_entries(self):
        """If the caller passes only K entries, the bonus access at
        target[K] would IndexError on the all-accept path. We don't
        guard against it (call sites are tightly coupled to verify
        trace's K+1 layout), so document this with a test."""
        with self.assertRaises(IndexError):
            _walk(K=3, draft=[1, 2, 3], target_kp1=[1, 2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
