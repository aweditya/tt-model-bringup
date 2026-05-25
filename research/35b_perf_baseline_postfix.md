# 35B-A3B perf baseline — post q/k_norm fix (2026-05-24)

Correctness floor: cleaned-up server (commit `fd4367f`), all sub-ops bit-clean
end-to-end, needle haystack L=100 retrieves verbatim.

## End-to-end (profile_35b_ttnn.py)

Prompt: `"The capital of France is"`. Warmup 3 forwards, then 5-token prefill + 24-token decode, all sync-bounded.

| | median ms/tok | tok/s |
|---|---|---|
| Prefill | 483.4 | 2.07 |
| Decode  | 480.2 | 2.08 |

Bootstrap (cold start): 112.8 s (40 layer weights). Generated text matches HF baseline.

## Per-block (profile_blocks_35b_ttnn.py — sync-bounded final decode step)

| Block | Total / tok | Per-layer | Layers | Share |
|---|---|---|---|---|
| **MoE** | 339.1 ms | 8.48 ms | 40 | **68.3%** |
| DN | 132.2 ms | 4.41 ms | 30 | 26.7% |
| ATTN | 24.8 ms | 2.48 ms | 10 | 5.0% |
| sum | 496.1 ms |  |  |  |
| other (embed, lm_head, RoPE, etc) | ~23 ms |  |  | ~5% |

Sum-of-blocks slightly exceeds end-to-end (496 vs 480) due to extra sync points
in the per-block timer; the share percentages are the authoritative number.

## Roofline check (P150 4-chip mesh, 404 GB/s/chip DRAM, [[reference-p150-roofline-priority]])

### MoE per layer per chip
- 8 experts × (gate_up + down) matmuls
- gate_up: [1, 2048] × [2048, 704_per_chip] (bf8 weight) → 1.44 MB
- down:    [1, 1408] × [1408, 512_per_chip] (bf8 weight) → 0.72 MB
- per expert: ~2.16 MB; × 8 experts = 17.3 MB/layer/chip
- BW floor: 17.3 MB / 404 GB/s = **0.043 ms/layer/chip**

Observed: **8.48 ms/layer**. Ratio observed/floor ≈ **200×**. MoE is dispatch-bound, not BW-bound.

### DN per layer
B17 demo (commit `71df77b`) showed eager → traced 4.14 ms → 0.72 ms (5.72×) for a
DN block in isolation. Current measured 4.41 ms/layer aligns. ⇒ DN is also
dispatch-bound; trace recovers most of it.

## Targets (rough projections — not measurements)

| Scenario | Saved | New total | tok/s |
|---|---|---|---|
| DN traced only (5.72×) | 109 ms | 371 ms | 2.69 |
| MoE traced (assume 8× like DN demo) | 297 ms | 183 ms | 5.46 |
| Both traced | 406 ms | 74 ms | **13.5** |
| 27B production reference | — | 77 ms | 13.0 |

If the MoE 200× dispatch overhead is real and tractable, 35B-A3B sits in the
same tok/s range as 27B once dispatch is eliminated. This is a projection. Will
be revised against real numbers after P1 (DN trace landed) and P3 (MoE trace).

## Priority shift

Pre-baseline plan had P1 (DN trace) ahead of P3 (MoE trace), reasoning that DN
trace was the easy win and we'd "gate P3 on P0+P1 measurements". The baseline
data inverts this: MoE dominates 68% of wall time; DN is 27%; ATTN is 5%. The
biggest leverage is MoE trace, not DN.

Updated order:
1. **P1 — DN trace in server** (still first because the pattern lifts directly from B17 demo and unblocks confidence in the trace path). Expected ~22% wall savings.
2. **P3 — MoE trace** (now the headline). Need to solve the data-dependent top-k dispatch. Options A/B/C in the task description; pick after a tt-metal docs read on indirect indexing / parameterized traces.
3. **P2 — full-step trace** (folds DN + attn + MoE + glue into a single capture, eliminates remaining inter-block dispatch).
4. **P4 — profile-driven optimizations** after trace is in.
