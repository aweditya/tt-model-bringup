# C'5 — Chunked Prefill for DeltaNet (Qwen3.6-27B)

**Date**: 2026-05-12
**Goal**: replace per-token sequential prefill through `deltanet_step_ondevice` with a
chunked-prefill kernel matching HF's `torch_chunk_gated_delta_rule`. Unblocks long
contexts (32k+) which are required for daily-driver use. At 32k context, sequential
prefill costs ~80 min/prompt; chunked target ~80 sec.

**Correctness gate**: chunked output cosine ≥ 0.9997 vs sequential output on
matched prompts (8, 64, 256, 1024 token cases).

**Reference**: HF `transformers/src/transformers/models/qwen3_next/modeling_qwen3_next.py`,
function `torch_chunk_gated_delta_rule` starting line 797 (main branch, 2026-05).
Our sequential impl: `experiments/91f_qwen36_27b_full_ondevice.py:122` (`deltanet_step_ondevice`).

---

## 1. The algorithm — math statement

DeltaNet recurrence (per V-head, fp32). Let `S_t ∈ R^{K×V}` be the recurrent SSM state,
`q_t, k_t ∈ R^K`, `v_t ∈ R^V`, decay scalar `g_t ∈ R`, beta `β_t ∈ R`. After Q-scaling
and L2-norm on q,k:

```
S_t = exp(g_t) · S_{t-1} + β_t · k_t · (v_t - exp(g_t) · k_t^T S_{t-1})^T
o_t = q_t^T S_t            (output before per-head RMSNormGated + silu(z) gate)
```

`torch_chunk_gated_delta_rule` splits the `N`-token sequence into `M = N/C` chunks of
size `C` (default `C=64`). Let upper-case denote chunk-level tensors. For chunk `i`:
`Q_i, K_i ∈ R^{C×K}`, `V_i ∈ R^{C×V}`, `g_i ∈ R^C`, `β_i ∈ R^C`. Define cumulative
chunk decay `G_i[c] = Σ_{s≤c} g_i[s]` (HF line: `g = g.cumsum(dim=-1)` after reshape).
Chunk-internal pairwise decay:

```
D_i[a,b] = exp(G_i[a] - G_i[b])    for a ≥ b (lower-triangle, else 0)
```

The clever trick: in the original sequential update, `S_t` depends on `S_{t-1}` via a
linear-in-`S` term plus a rank-1 update from `(v_t - exp(g_t)·k_t^T S_{t-1})`. Inside
a chunk we can isolate the part that depends only on `S_{i-1}` (the entering state)
and the part that mixes within the chunk. The chunk-internal mixing factorises as:

```
attn := -(K_β · K^T) ⊙ D            (lower-tri off-diagonal only, line: masked_fill diag-or-above)
T    := (I - attn)^{-1}             (effective resolvent of the within-chunk recurrence)
V'   := T · V_β                     (chunk-rotated values)
K'   := T · (K_β ⊙ exp(G)[:,None])  (chunk-rotated decayed keys for state update)
```

where `K_β = β · K`, `V_β = β · V` (HF lines `k_beta`, `v_beta`).

The HF implementation computes `T = (I - attn)^{-1}` by an explicit forward-substitution
loop (line: `for i in range(1, chunk_size): attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)`),
exploiting the fact that `attn` is strict-lower-triangular so `I - attn` is unit-lower-triangular
and its inverse can be built row-by-row in `O(C^2)` per row. Final `T = attn + I` (line:
`attn = attn + torch.eye(...)`).

**Per-chunk forward** (HF inner loop):

```
A_i      = (Q_i · K_i^T) ⊙ D_i              (within-chunk attention, lower-tri)
v_prime  = K'_i · S_{i-1}                    (subtract leakage from entering state)
v_new    = V'_i - v_prime
attn_int = (Q_i ⊙ exp(G_i)[:,None]) · S_{i-1}  (cross-chunk contribution)
O_i      = attn_int + A_i · v_new            (chunk output)

S_i = exp(G_i[-1]) · S_{i-1} + (K_i ⊙ exp(G_i[-1] - G_i)[:,None])^T · v_new
```

Final `O = [O_0; O_1; …; O_{M-1}]` of shape `(N, V)`; trailing `pad_size` rows dropped.

**Why this is equivalent to sequential**: the delta rule is a *rank-1 gated linear update*.
Linear updates have closed-form parallel-scan structure when expressed in terms of the
resolvent of the within-chunk decay matrix. The `(I - attn)^{-1}` matrix is exactly the
matrix that, applied to the chunk's `(V_β, K_β·exp(G))`, telescopes all sequential
delta-rule mixing within the chunk into a single matmul. The state transition between
chunks is the standard linear-recurrence `S_i = a_i · S_{i-1} + b_i` form where `a_i =
exp(G_i[-1])` (scalar per-head) and `b_i = K_i^⊤_decayed · v_new` (rank-`C`). Same
inputs → same outputs up to fp arithmetic ordering.

---

## 2. Numerical correctness

Chunked == sequential in exact arithmetic. fp drift sources, ranked:

1. **`(I - attn)^{-1}` forward-substitution** accumulates `C×C` updates in the chunk.
   In bf16, the unit-triangular inverse rows grow as products of `|attn| < 1` terms but
   can still cancel poorly. **Must run this in fp32.**
2. **`G = cumsum(g)`** at `C=64` accumulates 64 fp adds. Stable, but bf16 would lose
   precision near token 64 of a chunk. **Run cumsum in fp32.**
3. **`D = exp(G_a - G_b)`** lower-tri ranges across `[0, max_drift]`. If `g` is bounded
   by `softplus(a) · -exp(A_log)` (≈ negative reals), `D` entries are in `(0, 1]` —
   well-conditioned. No special treatment needed.
4. **Final matmuls** `A·v_new`, `K^T·v_new` follow the same HiFi4 + fp32 DEST policy
   as elsewhere — no change.

**Concrete dtype plan**: q/k/v/g/β enter as bf16 (our residual dtype); we cast to fp32
for cumsum / `(I-attn)^{-1}` / inter-chunk recurrence; chunk-output `O_i` cast back to
bf16 before per-head RMSNormGated. Net precision ≥ sequential (which already runs at
this mixed dtype).

---

## 3. Memory cost

Qwen3.6-27B DeltaNet dims (`research/qwen36_arch_notes.md`):
`N_V_HEADS=32, N_K_HEADS=16, K_DIM=128, V_DIM=128, N_REP=2`.

Per V-head, per chunk, intermediate fp32 tensors of size `C×C` (attn, D, T) and `C×K`,
`C×V` (Q,K,V,Kβ,Vβ,K',V'). At `C=64`:
- `attn/D/T`: 3 × 64×64 × 4 B = 49 kB / head → 1.57 MB / layer (32 heads)
- `Q,K,K_β,K'`: 4 × 64×128 × 4 B = 131 kB / head → 4.19 MB / layer
- `V,V_β,V'`: 3 × 64×128 × 4 B = 98 kB / head → 3.14 MB / layer
- **Per-layer working set: ~9 MB (fp32) or ~4.5 MB (bf16)**. Fits L1 easily.

State `S_i` (carried across chunks) is `K×V = 128×128 × 4 B = 64 kB / head = 2 MB / layer`
(fp32) — same shape as current `ssm_state`, no growth.

For N=32k tokens, M=500 chunks: streaming compute means **only one chunk's working set
resident at a time per layer**. Total active DRAM cost is dominated by the chunk-output
buffer `O ∈ R^{N×V_HEADS×V_DIM}` we must accumulate for the post-DeltaNet path.
At N=32k, bf16: `32k × 32 × 128 × 2 = 256 MB / layer`. We have 48 DeltaNet layers
(48 of 64; `i%4 != 3`). **Holding all 48 layer outputs simultaneously is 12 GB.** Cannot.

**Resolution — fused prefill loop**: prefill processes layer-by-layer (current pattern,
`91l:220`). For each chunk, we feed it through ALL layers in sequence, persisting only
the SSM state + KV cache slots. The 256 MB `O` buffer is per-layer transient, freed when
that layer's MLP+next-layer output is produced. **Peak working set ≈ a few hundred MB on
top of the 27 GB weights + KV cache** — fits a single P150 (32 GB DRAM).

---

## 4. Mapping to ttnn ops

| Math step | HF line | ttnn op | Notes |
|---|---|---|---|
| `q,k = l2norm`, `q *= 1/√K` | early | `ttnn.rsqrt(sum(x*x))`, `ttnn.mul` | already in `deltanet_step` (line 164-176) |
| Cast to fp32 | `.to(torch.float32)` | `ttnn.typecast` | used in `gated_attn_step:279` |
| Pad to chunk | `F.pad` | host-side pre-tokenize, or `ttnn.pad` | prefer host: prompt-length known at entry |
| Reshape `(N,*) → (M,C,*)` | `x.reshape(...)` | `ttnn.reshape` | TILE-aligned at `C=64` (tile=32, multiple) |
| `g.cumsum(dim=-1)` | line in HF | **`ttnn.cumsum`** | **UNVERIFIED on Blackhole.** Probe needed. Fallback: explicit prefix-sum tree using `ttnn.slice + ttnn.add` (log₂(C)=6 levels). |
| `tril` triangular mask | `torch.tril(ones)` | upload pre-baked bf16 mask tensor | one-time, host-side |
| `D = (G.unsqueeze - G.unsqueeze).tril().exp()` | line | `ttnn.sub` broadcast + `ttnn.exp` + masked-mul | `ttnn.sub` of `(C,1)` − `(1,C)` then mul by tril mask |
| `attn = -(K_β @ K^T) ⊙ D ⊙ strict-lower-mask` | line | `ttnn.matmul`, `ttnn.mul`, `ttnn.neg` | matmul shape `(C,K) × (K,C) → (C,C)` per V-head |
| `(I - attn)^{-1}` forward-sub | for-loop | **no native triangular-solve.** Manual loop. | See below |
| `T @ V_β`, `T @ (K_β ⊙ exp(G))` | matmuls | `ttnn.matmul` | `(C,C) × (C,V)` |
| Per-chunk `(Q K^T) ⊙ D` | line | `ttnn.matmul` + `ttnn.mul` | |
| `K' @ S_{i-1}`, `Q' @ S_{i-1}` | line | `ttnn.matmul` | `(C,K) × (K,V) → (C,V)` |
| `S_i` update | line | `ttnn.mul` (decay) + `ttnn.matmul` (rank-C update) | |

**`(I - attn)^{-1}` — no ttnn op exists.** Two options:
- **Option A (port HF literally)**: 63-iteration Python for-loop with `ttnn.slice` /
  `ttnn.mul` / `ttnn.add`. Stays small per op but **adds 63 dispatch barriers per
  chunk per layer per V-head**. With 48 layers × 32 heads × 500 chunks × 63 iters =
  48M dispatches per 32k prefill. Catastrophic.
- **Option B (host-precompute T₀, parametrise by `attn`)**: the inverse can be
  written as `T = I + attn + attn² + … + attn^{C-1}`. For strict-lower-triangular
  `attn ∈ R^{C×C}`, only `C-1` powers are nonzero. Six `ttnn.matmul` calls then
  square-and-multiply gets `attn^k` for all needed `k`, summed. **6 matmuls per
  chunk per head + 5 adds**, parallel across all heads. ~10× fewer dispatches than
  Option A.
- **Option C (preferred)**: batch all V-heads into a single `(N_V, C, C)` tensor and
  do **5 batched matmuls** for `attn² ... attn⁶`. Then assemble `T = Σ attn^k`. This
  is mathematically `(I - attn)^{-1}` ONLY if `attn^C = 0`, which holds for strict
  lower-tri at exactly `k = C` levels. We need all `C-1=63` powers though — that's
  6 squarings (log₂(64)) plus 6 sums. Total **~12 batched matmuls per chunk per
  layer**. This is the only realistic path.

**The `(I-L)^{-1}` solve is the algorithmic risk.** A working prototype must
benchmark Option C end-to-end before committing.

---

## 5. Implementation plan

### Function signature

```python
def deltanet_chunk_ondevice(x_chunk_tt, w_tt, ssm_state_tt, conv_state_tt, cfg, device):
    """
    x_chunk_tt: [C, HIDDEN] bf16 — C tokens of residual stream
    Returns:
      x_out_tt:        [C, HIDDEN]      bf16 (residual-added)
      ssm_state_new:   [N_V, K, V]      fp32
      conv_state_new:  [CONV_DIM, K-1]  bf16 (last C tokens' conv tail)
    """
```

C is chosen at call site (probe `C ∈ {32, 64, 128}`; HF default 64).

### Where it slots in

- Add new function next to `deltanet_step_ondevice` in `91f_qwen36_27b_full_ondevice.py`.
- In `91l_fp32_residual_generate.py:248-253` (prefill block): replace the
  per-token loop with a chunked loop. Skeleton:
  ```python
  CHUNK = 64
  for start in range(0, len(prompt_ids), CHUNK):
      chunk_ids = prompt_ids[start:start+CHUNK]
      x_chunk = embed_np[chunk_ids]               # (c, HIDDEN)
      x_tt = upload(x_chunk, device, bf16)
      for i in range(NUM_LAYERS):
          if layer_type == 'linear_attention':
              x_tt, H_new, c_new = deltanet_chunk_ondevice(x_tt, w_tt, ssm_states[dn], conv_states[dn], cfg, device)
          else:
              # gated_attn_chunk_ondevice — out of scope for C'5; for now keep
              # gated_attn as per-token sequential within the chunk loop
              for tok in range(len(chunk_ids)):
                  x_one = ttnn.slice(x_tt, [tok, 0], [tok+1, HIDDEN])
                  x_one, kv_k, kv_v = gated_attn_step_ondevice(x_one, ...)
                  # scatter back; or write a chunked gated_attn (future C'5b)
          x_tt = mlp_chunk_ondevice(x_tt, w_tt)   # MLP is trivially batched
  ```
- **Decode path unchanged**: continue using `deltanet_step_ondevice` for the
  generation loop. This preserves the C'4 trace-capture optimization.

### Conv-state handling

The 1D causal conv has kernel `K_conv=4`. Within a chunk it can be expressed as a
batched 1D conv over `C+K_conv-1` inputs (padding from `conv_state` on the left).
New `conv_state_new` = last `K_conv-1` rows of the chunk's mixed_qkv. Use
`ttnn.conv1d` if available, else materialize the `K_conv` shifts and sum (matches
the per-token formulation, just batched).

### Test plan

1. **Unit cosine test** (`experiments/93_chunk_prefill_unit.py`): random `(N=128,
   HIDDEN)` input, run sequential `deltanet_step_ondevice` for 128 steps; run
   `deltanet_chunk_ondevice` with `C=64`. Per-position cosine ≥ 0.9999.
2. **Sweep**: `N ∈ {8, 32, 64, 65, 127, 128, 256, 1024}`. Catches padding bugs
   (non-multiple-of-C cases).
3. **Layer-0 sanity**: prefill the actual prompt through real layer-0 weights;
   compare against `91l` sequential output on the same prompt. ≥ 0.9997.
4. **Full-model gate**: `91r_per_layer_diff.py` extended with `--prefill-mode chunked`
   flag. Pass when DeltaNet ≥ 0.9997 every layer.
5. **Paris demo** still produces "Paris" after chunked prefill.

---

## 6. Expected perf

**Dispatch count.** Sequential prefill at N=32k: 32k tokens × 48 DeltaNet layers ×
~30 ttnn ops/layer ≈ 46M dispatches. At ~30 µs/dispatch ≈ 1380 sec just in
dispatch. Chunked at C=64: 500 chunks × 48 layers × ~50 ops/chunk = 1.2M dispatches
≈ 36 sec. **40× dispatch reduction.**

**Weight bandwidth.** Each chunk reads the layer's projection weights once and
applies them to a `(C, HIDDEN)` GEMM batched. Sequential reads weights `C` times per
chunk-worth of tokens. **64× weight-read reduction** = 64× less DRAM bandwidth for
proj weights. DeltaNet projections are most of its DRAM footprint.

**Matmul efficiency.** `(64, HIDDEN) × (HIDDEN, K_DIM)` is a much better-shaped matmul
for the Tenstorrent matrix engine than `(1, HIDDEN) × …` — single-row matmul is
notoriously bandwidth-bound (no reuse). Chunked should hit the compute floor instead.

**Realistic speedup estimate.** Per-token chunked-prefill cost is bounded below by
the weight-bandwidth floor for one chunk's reads, amortized over C tokens. Today's
562 ms/tok sequential prefill → estimate **~80-150 ms/tok chunked at C=64** at 32k
context. 4-7× speedup, in line with the roadmap's 4-8× target.

**Caveats**:
- `(I - attn)^{-1}` solve adds per-chunk overhead. At C=64, it's ~12 batched matmuls
  on `(N_V=32, 64, 64)` tensors — small absolute cost, but it's serial relative to
  the per-chunk main compute.
- gated_attn layers (16 of 64) are NOT chunked in C'5 — they stay per-token within
  the chunk. A future "C'5b" extends to chunked gated_attn (full-context SDPA), but
  is independent of DeltaNet chunking.
- At small N (e.g., 5-token Paris prompt) chunked has overhead and may be a wash.
  Decision rule: dispatch chunked only when `N ≥ 2·C`.

---

## 7. Risks / unknowns

1. **`ttnn.cumsum` may not exist or may be broken on Blackhole.** No prior usage in
   our codebase. Probe via `experiments/93a_ttnn_cumsum_probe.py`. Fallback: manual
   log₂(C)=6-step prefix sum tree via `slice + add`.
2. **`(I - attn)^{-1}` cost.** Option C requires ~12 batched matmuls; if Blackhole's
   batched-matmul-with-small-K performance is poor for `(32, 64, 64) × (32, 64, 64)`,
   the entire chunked approach can lose to sequential. **Must benchmark first.**
3. **Precision of `(I - attn)^{-1}` in bf16/fp32.** The matrix-power expansion has
   nontrivial fp behaviour when entries approach 1. Validate on real Qwen weights
   (not random) before declaring correctness.
4. **`ttnn.matmul` shape constraints.** Our matmuls so far are `(1, HIDDEN) × (HIDDEN, X)`.
   Batched `(N_V, C, C) × (N_V, C, V)` shapes may hit untested TILE_LAYOUT paths.
   Risk of crashes or silent wrong-shape outputs.
5. **Conv-state at chunk boundaries.** Bug-prone: must pre-pad with the previous
   chunk's conv tail, post-extract the new tail. Off-by-one will silently corrupt
   subsequent chunks' early tokens (cosine drops on tokens 0-3 of every chunk).
6. **MLP/gated_attn not chunked.** If they remain per-token within the chunk loop,
   they dominate post-DeltaNet-speedup. Net prefill speedup is bounded by
   1/(fraction_of_time_in_gated_attn_and_mlp). Need to also chunk MLP (trivial) and
   eventually gated_attn (harder — full-context SDPA prefill).

**Top-3 to mitigate first**: risks 1, 2, 3 — all are gated by a single 1-day probe
(`93a_ttnn_cumsum_probe.py` + `93b_inverse_lower_tri_probe.py`). Run those before
writing `deltanet_chunk_ondevice`.

---

## Sequencing

1. **Day 1** — probes: `ttnn.cumsum` availability, `(I-L)^{-1}` Option C performance
   and precision on Blackhole.
2. **Day 2** — `deltanet_chunk_ondevice` skeleton: cast/cumsum/decay-mask, batched
   K_β/V_β/K'/V'.
3. **Day 3** — inter-chunk recurrence + conv-state plumbing. First end-to-end
   chunk-call test on `N=64`.
4. **Day 4** — unit sweep (N=8..1024), debug padding/off-by-one bugs.
5. **Day 5** — wire into `91l` prefill; full-model `91r` gate; Paris demo.
6. **Day 6** — measure 32k prompt prefill; commit.

If `(I-L)^{-1}` Option C fails on Day 1, pivot to Option B (matrix-powers via
explicit `attn^k` chain) and re-estimate. If both fail, the whole chunked-prefill
approach for *DeltaNet* is blocked and we revisit either (a) writing a custom
ttnn kernel for triangular solve, or (b) restricting context to 4k-8k where
sequential prefill is tolerable (~5-10 min) and shipping the daily-driver use case
without chunked prefill for v1.
