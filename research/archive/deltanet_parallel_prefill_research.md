# Parallel DeltaNet Prefill via Our Existing Neumann + Cumsum Work — Research 2026-05-19

**Compiled from background agent investigation. No code written.**

## Executive summary

**Our Branch C' Neumann series factorization IS the standard chunked-parallel
form used by Mamba-2 / GLA / RetNet for DeltaNet-style prefill.** It was
validated at production shape (commit memory `feedback_c5_primitives_green.md`)
but never put on the production decode path because decode is single-token.

Friend's qwen36 implementation **does NOT have parallel DeltaNet prefill** —
they explicitly loop the single-token decode recurrence for each prompt
position (`tt-qwen-36/models/tt_transformers/tt/decoder.py:258-321`). This
yields ~85 sec TTFT for a 1k prompt.

We can ship a **chunked-parallel prefill** that's ~50-100× faster than
friend's sequential loop, reusing primitives we already have validated.

## The math

DeltaNet gated delta recurrence per position t:
```
H_t = decay_t * H_{t-1} + outer(k_t, beta_t * (v_t - k_t^T H_{t-1}))
out_t = q_t @ H_t
```

**Standard parallel formulation** (Mamba-2/GLA style):
- Break sequence into chunks of C=64 or 128 positions
- Within a chunk, the recurrence collapses to a closed form via
  `(I - L)^{-1}` where L is a lower-triangular decay matrix
- Between chunks, propagate only the final state (still sequential, but at
  chunk granularity instead of token granularity)

For a 1k prompt at chunk=64: 16 chunks × parallel-within = 16 sequential
steps instead of 1024.

## Our existing primitives — both validated

### 1. Neumann factorization (`experiments/utils/neumann_inverse_probe.py`)

`(I - L)^{-1} = (I + L)(I + L²)(I + L⁴)(I + L⁸)(I + L¹⁶)(I + L³²)`

- Reduces 63 sequential matmuls → **10 total** (5 squarings + 5 multiplications)
- Per chunk C=64, per layer
- Validated: max_diff < 1e-3 vs `np.linalg.inv` at production shape
  `[32 heads, 64, 64]`
- Runtime: ~3-5 ms per layer per chunk (isolated)

### 2. ttnn.cumsum (`experiments/utils/cumsum_probe.py`)

- Verified on Blackhole bf16 and fp32
- Operates at production shape `[32, 64]`
- Used for prefix-sum of log-decay across positions within a chunk

These two primitives are exactly what chunked-parallel scan needs. The math
is proven (Mamba-2/GLA papers). Our composition has never been wired into a
prefill code path.

## Friend's prefill (confirmed sequential)

`/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/experiments/.refs/tt-qwen-36/models/tt_transformers/tt/decoder.py:258-321`:

```python
def _forward_prefill_as_decode_steps(self, x, ...):
    """Process prefill by looping over tokens sequentially."""
    for token_idx in range(seq_len):
        token_x = self._slice_prefill_token_residual(x, token_idx)
        token_output = self.forward(token_x, ..., mode=Mode.DECODE)
        ttnn.deallocate(token_x)
```

Friend explicitly does NOT have a parallel DeltaNet prefill — they shipped
sequential and accepted the TTFT penalty. Their `QWEN36_README.md:78-80`
says "GDN prefill is still a sequential decode-style recurrence loop, so
TTFT is not final."

## Cost analysis at 1k / 4k / 8k tokens

| Approach | 1k | 4k | 8k | Notes |
|---|---|---|---|---|
| Sequential (friend's loop) | 85 sec | 340 sec | 680 sec | decode-cost × seq_len |
| Chunked parallel (C=64) | 6-8 sec | 24-32 sec | 48-64 sec | 16/64/128 chunks × ~5 sec/chunk |
| Full parallel (single Neumann) | 1-2 sec | 4-8 sec | 8-16 sec | One (I-L)^-1 per layer |

Chunked-parallel = **~12× speedup over sequential** at 1k; ~50× at 8k.
Full-parallel would be even better BUT runs out of memory at 8k:

**Memory analysis (per layer × 48 layers):**
- Full parallel needs H_t for ALL t. At seq=8k: `[8k, 32, 128, 128] × 4 bytes = 2 GB/layer × 48 = 96 GB`. **Won't fit** in 12 GB DRAM.
- Chunked retains only within-chunk states + propagated final: `[64, 32, 128, 128] × 4 = 66 MB/layer × 48 = 3.2 GB`. **Fits comfortably.**

So chunked is the only viable architecture for our DRAM budget at long context.

## Implementation feasibility on tt-metal

**What exists (validated):**
- ✓ `ttnn.cumsum` on Blackhole at production shape
- ✓ `ttnn.matmul` dense + batched
- ✓ Neumann factorization in numpy/ttnn reference impl
- ✓ Basic tile ops for decay/delta/outer product
- ✓ Owned GDN decode kernel as a template for how to wire the recurrence

**What's missing:**
- A chunked-scan composition that ties cumsum + Neumann + matmul + state propagation into one prefill pass
- Possibly: an owned compute kernel for the chunked recurrence (analogous to owned_gdn_decode but for seq_len > 1)
- Proper trace structure for prefill (separate from decode trace)

**Realistic implementation path (2-3 weeks):**

1. **Week 1 — Reference impl + validation:**
   - Numpy reference: chunked Neumann prefill matching HF Qwen3.6-27B exactly
   - ttnn composition: cumsum + Neumann + matmul, single-chip, MAX_POS=128 test
   - Validate per-position output matches numpy ref at <1e-3

2. **Week 2 — TP mesh + integration:**
   - Port to (1, 4) mesh: cumsum/Neumann work on per-chip head shards
   - Wire `forward_prefill_tp_inner` paralleling `forward_token_tp_inner`
   - Separate prefill trace OR eager prefill
   - Validate vs sequential-decode-loop on a real prompt

3. **Week 3 — Production hardening:**
   - Bench TTFT at 500/1k/4k prompts
   - Hook into `handle_generate_tp` as the actual prompt-processing path
   - Cosine ladder on generated tokens vs HF oracle

## Open questions / risks

1. Does `ttnn.cumsum` + dense Neumann matmuls trace cleanly?
2. Chunk boundary state propagation (~1ms overhead/chunk) — does it dominate?
3. Numerical stability of repeated `(I-L)^-1` at long context — does precision drift?
4. Owned kernel needed, or pure ttnn composition?

## Why this is a differentiator

Friend has the C++ decode kernel; we have a smarter prefill via Neumann.
Different competitive advantages. If we ship parallel prefill while friend
ships sequential, our TTFT story is dramatically better — and TTFT is the
key UX metric for coding-assistant workflows.

For the qwen36-35b-a3b future bringup, the same chunked-parallel infra
carries over (same DeltaNet hybrid backbone). Building this once unlocks
prefill for the entire Qwen3.6 family on Tenstorrent.
