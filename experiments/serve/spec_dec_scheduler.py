#!/usr/bin/env python3
"""Speculative-decoding scheduler — wraps target+drafter+verify with
accept-walk logic. Phase 3 v0.0 correctness-first implementation
(2026-06-08).

**v0.0 limitations** (documented):
1. Drafter runs AUTOREGRESSIVELY at L=1 × K calls per round (the
   "parallel K-position drafter" claim in feasibility doc would need
   re-bringup at L=K). Cost: K × 6.4 ms traced ≈ 32 ms instead of ~8 ms.
2. Verify is READ-ONLY (Phase 2.B.1 ship decision). Cache writes for
   accepted tokens via target B=1 × accepted_count per round. Cost:
   N × 47 ms ≈ 188 ms at α=0.7. Tok/s SLOWER than plain B=1 baseline.
3. v1.0 perf upgrade: write-during-verify (non-aliased page-table)
   + parallel L=K drafter. Projected 3× speedup.

Architecture (parallel-drafter pattern, NOT autoregressive Leviathan):

  Per step:
    1. target.step() — produces hidden_states + last-layer KV (one per
       layer_type) at the current position. (NEW: target must expose
       state.shared_kv_for_drafter — see Phase 2.A.)
    2. drafter.forward(target_hidden_states_2_last, target_kv) →
       K logits (K candidate next-tokens emerge from one drafter forward,
       since drafter is parallel — see
       research/gemma4_assistant_feasibility.md).
    3. host: argmax K drafter logits → K candidate tokens.
    4. target.verify(K+1) — single B=K+1 forward via the aliased
       page-table (forks DeepSeek-V3 _build_verify_alias_page_table_host
       at tt-metal/models/demos/deepseek_v3/tt/generator.py:43-101).
    5. host: accept-walk — for i=0..K-1, accept candidate[i] iff
       argmax(target_verify_logits[i]) == draft[i]. Emit longest
       accepted prefix + 1 correction at first mismatch (or K+1
       additional verify logits at last position if all accepted).
    6. Rewind KV cache to accepted-position+1 on target; drafter has
       no KV so nothing to rewind.

Forks `cb_scheduler.Scheduler` step interface but replaces the inner
forward with target+drafter+verify.

Status: SKELETON. Implementation lands at Phase 3.B + Phase 4.A.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch


def build_verify_alias_page_table_host(
    base_page_table: torch.Tensor,
    K: int,
    verify_offset: int = 1,
) -> torch.Tensor:
    """Build a host-side aliased page-table for B=K+1 spec-dec verify.

    Single-stream adaptation of DeepSeek-V3's
    `_build_verify_alias_page_table_host` (tt-metal/models/demos/deepseek_v3/
    tt/generator.py:58-99). For one active prompt, all K+1 verify rows
    are aliased to row 0's KV blocks so a single B=K+1 forward reads
    the SAME KV history K+1 times.

    Args:
      base_page_table: int32 tensor [num_rows, pages_per_seq]. Row 0 is
        the active prompt's slot; rows [1, num_rows) are spare.
      K: spec-dec lookahead depth. K+1 verify rows aliased.
      verify_offset: starting row for the aliased verify rows. Default 1
        (row 0 = active prompt, rows [1, 1+K+1) = aliased reads).

    Returns:
      int32 tensor [num_rows, pages_per_seq]. Rows
      [verify_offset, verify_offset+K+1) point at row 0's KV blocks.
      Other rows unchanged.

    Forked from `research/deepseek_v3_alias_page_table_reference.md` §
    "Our adaptation for single-stream B=K+1 spec-dec".
    """
    assert base_page_table.dim() == 2, \
        f"base_page_table must be rank-2; got {tuple(base_page_table.shape)}"
    num_rows = int(base_page_table.shape[0])
    assert num_rows >= verify_offset + K + 1, (
        f"page_table has {num_rows} rows; need at least "
        f"{verify_offset + K + 1} for K={K} + verify_offset={verify_offset}"
    )
    alias = base_page_table.clone().to(torch.int32)
    for i in range(K + 1):
        alias[verify_offset + i] = alias[0]
    return alias


@dataclass
class SpecDecConfig:
    K: int = 5                  # lookahead depth (candidates per round)
    max_new: int = 256
    eos_ids: frozenset = frozenset()
    target_temperature: float = 0.0  # greedy by Leviathan correctness contract
    log_accept_rate: bool = True


@dataclass
class StepResult:
    """One spec-dec round's output."""
    accepted_tokens: list[int]     # length 1..K (always >= 1; K-th comes from target verify)
    accept_count: int              # how many drafter candidates were accepted (0..K)
    target_step_ms: float          # wall time for target forward + KV expose
    drafter_step_ms: float         # wall time for drafter parallel forward
    verify_step_ms: float          # wall time for target B=K+1 verify
    host_walk_ms: float            # accept walk + sample on host

    @property
    def n_emitted(self) -> int:
        return len(self.accepted_tokens)

    @property
    def alpha(self) -> float:
        """Per-round acceptance rate. K accepts → 1.0; 0 accepts → 0.0."""
        return self.accept_count / max(1, self._K)

    _K: int = 5  # set by SpecDecScheduler.step()


class SpecDecScheduler:
    """Single-stream spec-dec wrapper. v0 = B=1 (single client).
    Multi-slot CB integration is Phase 4 follow-up.

    Constructor:
        target_state: bootstrapped Gemma 4 12B State (must support
                      attn_decode_step_tt + shared_kv_for_drafter export)
        drafter_state: bootstrapped Gemma4UnifiedAssistant State (forks
                       server_gemma4_unified_ttnn pattern; Phase 1 deliverable)
        config:       SpecDecConfig (K, max_new, eos)
    """

    def __init__(self, target_state, drafter_state, config: SpecDecConfig):
        self.target_state = target_state
        self.drafter_state = drafter_state
        self.config = config
        self.K = config.K
        # Stats
        self.total_rounds = 0
        self.total_accepts = 0
        self.total_emitted = 0
        # Rolling per-step times for diagnostics
        self._times = []

    # ─── Phase 2/3 device APIs (to be filled in) ─────────────────────────

    def _target_step(self, token_id: int, cur_pos: int) -> int:
        """Run target B=1 decode forward at cur_pos via existing eager path.
        Returns argmax_int. Cache advances by 1.

        This is the cache-write path. For v0.0 read-only verify, called N
        times per round to advance cache by N=accepted_count positions.
        """
        import server_gemma4_unified_ttnn as srv
        return srv.step_forward_v031(
            self.target_state, tok_id=token_id, pos=cur_pos)

    def _read_target_shared_kv(self, L_kv: int):
        """Read target's last-sliding + last-full KV history of length L_kv
        from the target's paged cache. Output matches HF oracle shape:
            sliding: (1, NKV=8, L_kv, 256)
            full:    (1, NKV=1, L_kv, 512)
        """
        import server_gemma4_unified_ttnn as srv
        return srv.read_shared_kv_for_drafter(self.target_state, L_kv=L_kv)

    def _drafter_autoregressive_K(
        self,
        target_h_prev_np,  # numpy [1, 1, BACKBONE_HIDDEN=3840]
        target_h_last_np,  # numpy [1, 1, BACKBONE_HIDDEN=3840]
        shared_kv_np,       # dict: "sliding_attention"/"full_attention" → (K, V)
    ) -> list:
        """Autoregressive drafter ×K — each call generates 1 candidate +
        its hidden output, which feeds the next call.

        Drafter is locked at L=1 (our v0.0 bringup); to get K candidates
        we chain K calls. Drafter's `hidden` output is the post-projection
        target-hidden-equivalent shape [1, 1, 3840], so we use it as the
        "current target hidden" approximation for the next drafter call.

        Returns: K candidate ints.
        """
        import server_gemma4_12b_assistant_ttnn as drf
        import numpy as np
        candidates = []
        # Round 0 of chain: use target's actual (prev, last) hidden.
        prev_h = target_h_prev_np
        cur_h = target_h_last_np
        for k in range(self.K):
            inputs_embeds = np.concatenate([prev_h, cur_h], axis=-1).astype(np.float32)
            # inputs_embeds shape [1, 1, 2*BACKBONE_HIDDEN]
            out = drf.drafter_forward(self.drafter_state, inputs_embeds,
                                       shared_kv_np)
            tok = int(out["argmax"].flatten()[0])
            candidates.append(tok)
            # Chain: next call's "prev" = current "cur"; next "cur" = drafter's hidden
            prev_h = cur_h
            cur_h = out["hidden"]  # [1, 1, 3840]
        return candidates

    def _target_verify_kp1(self, base_token: int, draft_tokens: list,
                            cur_pos: int) -> list:
        """Target B=K+1 verify trace: K+1 Q rows with tokens
        [base_token, draft_tokens[0], ..., draft_tokens[K-1]] all reading
        the same cache history through `cur_pos` via the alias page-table.

        Returns K+1 argmaxes (host ints) — one per row.
        """
        import server_gemma4_unified_ttnn as srv
        assert len(draft_tokens) == self.K
        candidate_token_ids = [base_token] + list(draft_tokens)
        # setup_verify_kp1_state is idempotent at same K.
        srv.setup_verify_kp1_state(self.target_state, K=self.K)
        srv.update_verify_inputs(self.target_state,
                                  current_pos=cur_pos,
                                  candidate_token_ids=candidate_token_ids)
        # Trace must be captured once before first call (caller invokes
        # ensure_verify_trace_kp1 ahead of the first step).
        argmaxes = srv.verify_step_traced(self.target_state)
        # verify_step_traced returns numpy [Bv] = K+1 argmaxes.
        return [int(x) for x in argmaxes]

    # ─── Host-side accept walk (final Phase 3.B) ─────────────────────────

    def _accept_walk(self, draft_tokens: list, target_argmaxes_kp1: list) -> tuple:
        """Greedy accept walk over K+1 argmaxes vs K draft candidates.

        For i in 0..K-1: target_argmaxes_kp1[i] is target's prediction
        for position (cur_pos+i+1) given context [base, draft_0..draft_{i-1}].
        Accept draft[i] iff target_argmaxes_kp1[i] == draft[i].

        At first mismatch: emit accepted prefix + target's correction at
        that row. If all K accepted: emit accepted + target_argmaxes_kp1[K]
        as the K+1-th bonus token.

        Returns (emitted_tokens_list, accept_count).
        """
        emitted = []
        for i in range(self.K):
            if target_argmaxes_kp1[i] == draft_tokens[i]:
                emitted.append(draft_tokens[i])
            else:
                # First reject. Emit target's correction; stop.
                emitted.append(int(target_argmaxes_kp1[i]))
                return emitted, i  # i = number of draft accepts
        # All K drafts accepted. Emit row K's argmax as bonus.
        emitted.append(int(target_argmaxes_kp1[self.K]))
        return emitted, self.K

    # ─── Public step API ─────────────────────────────────────────────────

    def step(self, base_token: int, target_h_prev_np, target_h_last_np,
             cur_pos: int) -> StepResult:
        """One spec-dec round at cur_pos.

        Args:
          base_token: the last-accepted token, used as Q row 0 of verify
            (predicts the next position the round starts at).
          target_h_prev_np, target_h_last_np: target's last 2 hidden
            states (post-final-norm) at positions cur_pos-1 and cur_pos.
            Used to bootstrap the drafter's autoregressive chain.
          cur_pos: target's cur_pos_buf value (cache valid through here).

        Returns StepResult. Cache advances by accept_count + 1 positions
        via the target B=1 calls at the end of this method.
        """
        t0 = time.time()
        # Read target's shared KV history (used by drafter cross-attention).
        shared_kv = self._read_target_shared_kv(L_kv=cur_pos + 1)
        t_kv = time.time()

        # Drafter ×K chain → K candidate tokens
        draft_tokens = self._drafter_autoregressive_K(
            target_h_prev_np, target_h_last_np, shared_kv)
        t_draft = time.time()

        # Target B=K+1 verify (read-only)
        target_argmaxes_kp1 = self._target_verify_kp1(
            base_token, draft_tokens, cur_pos)
        t_verify = time.time()

        # Host accept walk
        emitted, accept_count = self._accept_walk(draft_tokens, target_argmaxes_kp1)
        t_walk = time.time()

        # ── Cache advance ──
        # Read-only verify (Phase 2.B.1 ship constraint): must run target B=1
        # for each emitted token to write K/V to cache at positions
        # cur_pos+1, cur_pos+2, ..., cur_pos+len(emitted).
        # cur_pos at scheduler entry points at last-written cache slot;
        # emitted tokens go into slots cur_pos+1..cur_pos+len(emitted).
        # The token to FEED to target B=1 at slot cur_pos+i is emitted[i-1]
        # (the previous emitted token; first one is the base_token).
        # We discard the argmax outputs — they would be the NEXT prediction
        # which we've already verified.
        feed_tokens = [base_token] + emitted  # length len(emitted)+1
        for i in range(len(emitted)):
            slot_pos = cur_pos + 1 + i
            self._target_step(token_id=feed_tokens[i], cur_pos=slot_pos)
        t_advance = time.time()

        # Stats
        self.total_rounds += 1
        self.total_accepts += accept_count
        self.total_emitted += len(emitted)

        return StepResult(
            accepted_tokens=emitted,
            accept_count=accept_count,
            target_step_ms=(t_advance - t_walk) * 1e3,
            drafter_step_ms=(t_draft - t_kv) * 1e3,
            verify_step_ms=(t_verify - t_draft) * 1e3,
            host_walk_ms=(t_walk - t_verify) * 1e3,
            _K=self.K,
        )

    # ─── Diagnostics ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return cumulative acceptance rate + per-component timings."""
        alpha = self.total_accepts / max(1, self.total_rounds * self.K)
        emitted_per_round = self.total_emitted / max(1, self.total_rounds)
        return {
            "K": self.K,
            "rounds": self.total_rounds,
            "alpha": alpha,
            "tokens_per_round": emitted_per_round,
            "speedup_vs_K1_baseline": emitted_per_round,
            # Note: above is rough — true speedup needs wall-time comparison
            # vs plain B=1 decode. Phase 3.D bench delivers that number.
        }


# Module-level sanity: skeleton imports clean. No ttnn dependency until
# Phase 1+ device-side implementation lands.
if __name__ == "__main__":
    print(f"spec_dec_scheduler skeleton OK. Status: Phase 0 deliverable.")
    print(f"Implementation tasks: {SpecDecScheduler._target_step.__doc__.split(chr(10))[-1]}")
