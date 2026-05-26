# Phase A6 — Parallel Scan Kernel for Gated DeltaNet Prefill

## Why this matters

DeltaNet decode is `seq_len = 1` per step — trivially parallel across the (already-empty) seq dim, no scan needed. **Prefill is the problem.** A chat interface where the user pastes 4K tokens of code expects prefill to take <5 s; naive serial-over-time would take many seconds *just for the recurrence*. Full Blelloch / Heinsen is the right investment for future-proofing.

## The recurrence we need to parallelize

From `qwen36_modeling_excerpts.md`, the per-token update is:

```
H_t = H_{t-1} · exp(g_t)   +   outer(K_t, beta_t · (V_t - sum(H_{t-1}·exp(g_t) · K_t, dim=-2)))
out_t = sum(H_t · Q_t, dim=-2)
```

This looks gnarly, but for the scan we can flatten it. The state H lives in `[B, n_v_heads, d_k, d_v]`. Treat each "head" independently — they don't talk. Per head, we have a linear recurrence on a `d_k × d_v` matrix:

```
H_t = α_t · H_{t-1} + Δ_t
```

where `α_t = exp(g_t)` (per-head scalar, same across all elements of H) and `Δ_t = outer(K_t, β_t · (V_t - K_t^T · α_t · H_{t-1}))`.

**The wrinkle:** Δ_t depends on H_{t-1} through `sum(H_{t-1} · K_t, dim=-2)`. So it's not a "pure" scan with state-independent inputs. This is the **selective scan** problem that Mamba solved.

## Two paths

### Path A — Chunked-serial (v1, bootstrap)

Pretend the recurrence IS pure. Compute `Δ_t` first by *approximating* `H_{t-1}` with the chunk's previous state (committed-but-stale H). Run the recurrence chunk-by-chunk:

```
chunk_size = 64
for c in 0..N/chunk_size:
    H_chunk_start = H_{c·64 - 1}  (committed from previous chunk)
    for i in 0..chunk_size:        # serial WITHIN chunk
        # use H_chunk_start as the "anchor" — correct because we're inside the chunk
        compute Δ_i exactly w.r.t. H_{c·64 + i - 1}
    commit H_{(c+1)·64 - 1}
```

This is **NOT a real parallel scan** — it's just chunked-serial-along-time. Each chunk runs ttnn ops in parallel across heads / batch but not within the chunk's time dim. For 1024 tokens / 64-chunk size = 16 chunks, each serial.

Why this is OK as v1: per-chunk we still get parallelism across `[B, n_v_heads]` so the work isn't wasted. Each chunk is one ttnn dispatch, so we pay 16 dispatch costs (~1.5 ms total) plus 16 × per-chunk-compute. For 1024 tokens / 4K-token prefill = 64 chunks, so prefill cost is dominated by per-chunk compute, not the scan.

**Limitation:** no asymptotic improvement vs serial — just constant-factor better via chunking. For 4K-token prefill, this is ~1-2 s. Tolerable, not great.

### Path B — Full Blelloch tree scan (v2, ship)

The selective scan IS expressible as an associative scan on **transition pairs**:

```
T_t = (α_t, Δ_t)
T_a ⊕ T_b = (α_b · α_a, α_b · Δ_a + Δ_b)
T_1 ⊕ T_2 ⊕ ... ⊕ T_n  computed in log(N) depth
final H_t = (T_1 ⊕ ... ⊕ T_t).Δ
```

The catch with selective scan: Δ_t depends on H_{t-1}, which is the partial-scan-up-to-t-1. Looks circular — but Mamba's trick is to **rewrite the recurrence** so Δ_t depends only on inputs (Q, K, V, g, β at time t), not on the running state. The Schlag et al. delta-rule formulation lets us do this:

```
Δ_t = outer(K_t, β_t · V_t) - β_t · K_t · (K_t^T) · accumulated_H
```

If we define `M_t = (α_t, K_t, V_t, β_t)` and a clever composition operator that handles the "subtract β·K·K^T from the running state" cleanly, the operator IS associative. This is the **Gated DeltaNet associative scan** specifically (Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", 2024).

The composition is:
```
(α_a, KKβ_a, KVβ_a) ⊕ (α_b, KKβ_b, KVβ_b) =
    (α_b · α_a, KKβ_b + KKβ_a · (1 - α_b · KKβ_b), KVβ_b + KVβ_a · (1 - α_b · KKβ_b))
```

(rough form — need to double-check against the paper)

**Implementation:** Blelloch upsweep/downsweep on the time dim, with each "scalar" being the composition tuple. Work O(N), depth O(log N). For N=4096 and 110 tensix cores, log_2(4096)=12 depth steps × constant per step.

### Decision: do BOTH, ship v1 quickly, follow up with v2

| | Path A (chunked serial) | Path B (full Blelloch) |
|---|---|---|
| Impl effort | ~3-4 hrs | ~10-15 hrs |
| Asymptotic cost (4K prefill) | O(N) serial, ~1-2 s | O(N log N) parallel, ~200-500 ms |
| Correctness risk | Low (it's still the exact recurrence per chunk) | Medium (operator algebra is fiddly) |
| When | First, to unblock A3 isolated test | Second, before shipping chat interface |

---

## Per-chip utilization target (the user's invariant)

We will measure scan performance with TWO metrics, in this order:

1. **Single-chip util %:** what fraction of one P150's memory bandwidth + tensix compute is the scan kernel actually using?
2. **Multi-chip scaling:** how does 2-chip TP compare to 1-chip, normalized for compute?

Memory bandwidth ceiling for the scan: per step we read α_t (scalar), K_t (per-head 128-dim), V_t (per-head 128-dim) and write H_t (per-head 128×128). For 32 v-heads:

- Per token: 32 × (1 + 128 + 128 + 128×128) bytes (bf16 except H in fp32 = 4× bigger)
- ≈ 32 × (16384·4 fp32 + 256·2 bf16) = 2.1 MB
- For 4096 tokens: 4096 × 2.1 MB = **8.6 GB total state I/O**
- DRAM bandwidth 450 GB/s → **memory-bound floor: 8.6 / 450 = 19 ms** for the scan over 4K tokens on one chip

So v2 Blelloch's target is **~20 ms scan time for 4K prefill** — beat that and we're hitting the bandwidth ceiling. v1 chunked-serial will likely land at 200-500 ms (dispatch-dominated). Both should fit "future-proof" with v2.

---

## Test plan

Both v1 and v2 land as `experiments/86_gated_deltanet_scan.py`:

```
1. Generate random {Q, K, V, g, β} for sequences of length N ∈ {64, 256, 1024, 4096}
2. Compute numpy fp32 reference H_t and out_t serially (gold)
3. Run our scan implementation on Blackhole
4. Cosine ≥ 0.99 vs numpy reference (each timestep)
5. Time the scan kernel only (separate from projections)
6. Report: ms/token amortized, % of one-chip memory ceiling
```

Then a 2-chip variant for v2 only, with the head-dim split across chips. Reports same metrics plus speedup vs 1-chip.

---

## When does this happen

Phase A6 lands AFTER A3 (Gated DeltaNet decode-path isolated) so we have the surrounding kernel infrastructure (projections, conv1d, L2 norm, etc.) already validated. The scan is just one piece of the DeltaNet path.

Order:
- A3: DeltaNet **decode** path only (1-step recurrence, no scan needed). Cosine + perf. ← gives us projections + conv + state update mechanics
- A6.v1: chunked-serial scan, validates prefill correctness
- A6.v2: Blelloch tree scan, beats v1 by ~10× for longer prefill
- Phase B integration uses A6.v2

---

## Open questions to resolve before coding

1. **Exact gated-delta-rule operator.** The composition algebra I sketched above is approximate. Need to either read Yang et al. directly or pull the parallel scan implementation from the `mamba-ssm` repo and translate.
2. **fp32 in ttnn.** Path B's intermediate state must be fp32. ttnn supports it but not all ops are fp32-fast on Blackhole. May force us to do part of the scan in bf16 with fp32 accumulator only at the very end.
3. **conv1d kernel=4 prefill.** Not strictly part of the scan but adjacent — it's applied BEFORE the scan, along seq dim. ttnn has `ttnn.conv1d`, need to check that depthwise-along-channel works with kernel=4 at our shapes.

I'll resolve these alongside the A3 implementation when the device comes back.
