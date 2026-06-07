#!/usr/bin/env python3
"""Speculative-decoding scheduler — wraps cb_scheduler with parallel-drafter
verify/accept logic.

This is the SKELETON committed during Phase 0 of the Gemma 4 spec-dec
build (greenlit 2026-06-07). Methods marked NotImplementedError still
need device-side work (see Phase 2 + 3 in
`research/gemma4_mtp_plan_of_action.md`).

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

    def _target_step(self, token_id: int) -> tuple:
        """Run target's decode forward at the new token, return
        (target_hidden_state, shared_kv_dict).

        NEW vs current `server_gemma4_unified_ttnn.attn_decode_step_tt`:
        target must populate `state.shared_kv_for_drafter = {
            "sliding_attention": (K_last, V_last),
            "full_attention":    (K_last, V_last),
        }` — the K/V tensors from the LAST sliding attn layer and LAST
        full attn layer. Phase 2.A deliverable.
        """
        raise NotImplementedError("Phase 2.A: target.expose_shared_kv hook")

    def _drafter_parallel_forward(
        self,
        target_hidden_2_last,  # ttnn.Tensor [B, 2, 3840] — concat of last 2 target hidden
        shared_kv,             # dict per layer_type → (K_tensor, V_tensor)
    ):
        """One drafter forward → K candidate logits + drafter's projected
        last_hidden_state (for next round's t-1 slot).

        Forks Phase 1's `server_gemma4_unified_assistant_ttnn.py` pattern
        (4 Gemma 4 layers + pre/post Linear projection + lm_head). At
        v0 the drafter is eager B=1; trace integration is Phase 3.C.
        """
        raise NotImplementedError("Phase 1: drafter bringup")

    def _target_verify_kp1(self, draft_tokens: list[int]) -> list:
        """Single target forward at B=K+1 verifying the K draft tokens
        in parallel.

        Uses aliased page-table (forks DeepSeek-V3
        `_build_verify_alias_page_table_host` at
        `tt-metal/models/demos/deepseek_v3/tt/generator.py:43-101`)
        so all K+1 logical batch rows read from the same physical KV
        slot.

        Returns K+1 logits arrays (numpy) — one per logical row.
        """
        raise NotImplementedError("Phase 2.B: aliased page-table + B=K+1 trace")

    # ─── Host-side accept walk (final Phase 3.B) ─────────────────────────

    @staticmethod
    def _argmax_with_tiebreak(logits) -> int:
        """numpy argmax breaks ties by lowest index — exactly the
        deterministic tie-break we want for Leviathan correctness
        (matches HF do_sample=False)."""
        return int(logits.argmax(axis=-1))

    def _accept_walk(self, draft_tokens: list[int], target_logits_kp1) -> tuple:
        """Greedy accept walk. For i=0..K-1, accept draft[i] iff
        argmax(target_verify_logits[i]) == draft[i]. At first mismatch,
        emit (accepted prefix) + (target's argmax at the mismatch row).
        If all K accepted, also emit target_verify_logits[K]'s argmax
        (the bonus K+1-th token).
        """
        accepted = []
        for i in range(self.K):
            target_tok = self._argmax_with_tiebreak(target_logits_kp1[i])
            if target_tok == draft_tokens[i]:
                accepted.append(draft_tokens[i])
            else:
                # First reject. Emit target's correction.
                accepted.append(target_tok)
                return accepted, i  # i = number of draft accepts
        # All K drafts accepted. Emit bonus token from row K (K+1-th logit).
        bonus = self._argmax_with_tiebreak(target_logits_kp1[self.K])
        accepted.append(bonus)
        return accepted, self.K

    # ─── Public step API ─────────────────────────────────────────────────

    def step(self, prev_token: int) -> StepResult:
        """One spec-dec round. Returns 1..K+1 tokens.

        Call repeatedly with `prev_token = result.accepted_tokens[-1]`
        until EOS or max_new reached.
        """
        t0 = time.time()
        target_hidden_2_last, shared_kv = self._target_step(prev_token)
        t_target = time.time()

        # Drafter parallel forward → K candidate logits
        draft_logits = self._drafter_parallel_forward(
            target_hidden_2_last, shared_kv,
        )
        draft_tokens = [
            self._argmax_with_tiebreak(draft_logits[i])
            for i in range(self.K)
        ]
        t_draft = time.time()

        # Target B=K+1 verify
        target_logits_kp1 = self._target_verify_kp1(draft_tokens)
        t_verify = time.time()

        # Host accept walk
        accepted, accept_count = self._accept_walk(draft_tokens, target_logits_kp1)
        t_walk = time.time()

        # Stats
        self.total_rounds += 1
        self.total_accepts += accept_count
        self.total_emitted += len(accepted)

        return StepResult(
            accepted_tokens=accepted,
            accept_count=accept_count,
            target_step_ms=(t_target - t0) * 1e3,
            drafter_step_ms=(t_draft - t_target) * 1e3,
            verify_step_ms=(t_verify - t_draft) * 1e3,
            host_walk_ms=(t_walk - t_verify) * 1e3,
            _K=self.K,
        )

    def generate(self, prompt_ids, log=None):
        """Drive `step` until max_new or EOS. Returns the generated
        token list (not including prompt).
        """
        if log is None:
            def log(_msg): pass

        # NOTE: prompt prefill happens OUTSIDE this scheduler — caller
        # runs the existing target prefill via cb_scheduler's prefill
        # path, gets the first sampled token, then drives spec-dec from
        # there.
        raise NotImplementedError("Phase 4: integrate with cb_api prefill")

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
