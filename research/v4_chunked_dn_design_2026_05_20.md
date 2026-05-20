# v4 prefill — chunked DeltaNet integration (design doc)

**Status**: design only. v3 ships with per-position DN; v4 replaces the
DN per-position loop with chunked-parallel via the C'5 algorithm.

**Goal**: close the per-position DN bottleneck in prefill. v3 already
batches attention + MLP. The remaining serial cost is DN's 5×-per-layer
per-position loop. Chunked DN runs all positions through DN in 1 call
per layer.

**Prereq satisfied**: C'5 math primitives validated
(`feedback_c5_primitives_green`):
- `ttnn.cumsum` works at production shape
- `(I-attn)^{-1}` Neumann factorization is numerically stable in bf16/fp32 mix
- `ttnn.slice` on row-aligned TILE works bit-exact
- Algorithm + memory budget already specced in
  `research/c5_chunked_prefill_plan.md`

## Integration plan

### 1. New function `forward_prefill_tp_inner_v4_chunked_dn`

Mirror v3's structure, replacing only the per-position DN body:

```python
def forward_prefill_tp_inner_v4_chunked_dn(state, prompt_ids, capture_logits=False):
    # Step 1-2: batched embed + RoPE table lookup (SAME AS v3)
    x_seq = ...  # [seq_len, HIDDEN], clone-safe per v3 fix

    # Step 3: layer loop
    for layer_idx, layer in enumerate(state.layers):
        if layer['type'] == 'linear_attention':
            # ← NEW: chunked-parallel DN instead of per-position loop
            x_seq = deltanet_step_tp_chunked(
                state, x_seq, layer['dn'], cfg, seq_len)
        else:
            x_seq = gated_attn_step_prefill_tp(...)  # SAME AS v3
        x_seq = mlp_step_tp(state, x_seq, layer['mlp'])  # SAME AS v3 (batched)

    # Step 4: final norm + LM head (SAME AS v3)
```

### 2. New op `deltanet_step_tp_chunked`

Takes `[seq_len, HIDDEN]` input, returns `[seq_len, HIDDEN]` residual-
added output. Internally processes ONE chunk-sized batch at a time
(default C=64); for seq_len ≤ C, single-chunk path; for longer, chunk loop.

Per-layer state update happens at chunk boundary — final SSM state at
chunk_M-1 becomes initial state for chunk_M, naturally matching
sequential semantics.

Internal math (per C'5 plan §1):
```python
# Per V-head, processing one chunk of size C ≤ seq_len:
G = ttnn.cumsum(g, dim=-1)                              # fp32 cumsum of decays
D = ttnn.exp(G[:,None] - G[None,:]) * lower_tri_mask    # [C, C] decay matrix
K_beta = beta[:,None] * k                               # [C, K]
V_beta = beta[:,None] * v                               # [C, V]
attn = -(K_beta @ k.T) * D                              # [C, C] within-chunk attn
T = inverse_unit_lower_triangular(I + attn)             # Neumann (I-attn)^-1
V_prime = T @ V_beta
K_prime = T @ (K_beta * exp(G)[:,None])
A = (q @ k.T) * D                                       # [C, C]
v_prime = K_prime @ S_prev                              # [C, V]
v_new = V_prime - v_prime
attn_int = (q * exp(G)[:,None]) @ S_prev                # [C, V]
O = attn_int + A @ v_new                                # chunk output [C, V]
S_new = exp(G[-1]) * S_prev + (k * exp(G[-1] - G)[:,None]).T @ v_new  # next state
return O, S_new
```

Cast policy (per C'5 plan §2):
- `cumsum`, `(I-attn)^{-1}`, inter-chunk recurrence: fp32
- All other ops: bf16
- DEST register: fp32 accumulation, bf16 output

### 3. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| `ttnn.cumsum` perf on real shapes | Slow Neumann setup | Probe at C=64, [N_V_HEADS, C] — should be cheap |
| `(I-attn)^{-1}` via Neumann series convergence in bf16 | Numerical instability | Run inverse in fp32, cast back |
| TILE_LAYOUT padding interactions on C × C matrices | Subtle math errors | Use ROW_MAJOR for triangular ops, TILE for matmuls |
| reshape-view bugs (the B.2.2 trap) | Silent corruption | Apply `ttnn.clone` proactively per audit |
| Cross-chunk state threading via `ttnn.copy` | State leak | Use the explicit-copy pattern from v3 DN |
| Per-layer chunk-output memory budget | OOM at 32k context | C'5 plan §3: single-chunk-resident working set, ~few hundred MB |

### 4. Validation

Same probe harness: `probe_prefill_vs_decode_loop_tp --mode chunked_dn`.
Gate: per-position cos ≥ 0.9997 vs decode-loop reference.
At seq_len ≤ C, math is single-chunk (closest to sequential). Validate at:
- seq_len=5 (sub-chunk, edge case)
- seq_len=32 (single chunk, exactly half C)
- seq_len=64 (single chunk, full C)
- seq_len=128 (2 chunks, tests cross-chunk transition)
- seq_len=500 (long context regime)

### 5. Effort estimate

~3-5 day-arcs of focused work:
1. Day 1: build `deltanet_step_tp_chunked` for SINGLE chunk (no chunking loop yet)
   validate against per-position DN at seq_len=8
2. Day 2: add chunk loop + state threading; validate at seq_len=64, 128
3. Day 3: integrate into `forward_prefill_tp_inner_v4_chunked_dn`; validate end-to-end
4. Day 4-5: debug numerical edge cases (last chunk padding, cross-chunk drift),
   long-context validation (seq=500, 1024)

### 6. Expected payoff

At seq_len=N, v3 prefill currently spends ~80% of time in per-position DN
inner loops (estimate from v3 wall time analysis). Chunked DN reduces that
to ~1 call per layer. Expected speedup at seq_len=N:
- N=32: ~3× faster prefill (DN was 25 ops/layer × 32 layers, now 1 × 32)
- N=500: ~10× faster prefill (DN was bottleneck)
- N=32k (32 chunks): ~20× faster prefill (closes the gap to ttnn dispatch limit)

Together with v3's batched attn + MLP, v4 would deliver true batched prefill
on tt-metal — the daily-driver long-context goal.

### 7. NOT in scope

- Tracing v4 (separate effort; trace + chunked DN is the ultimate ship)
- chunked attention (current paged SDPA already chunks internally via SDPAProgramConfig)
- chunked MLP (already batched in v3; no further gain)

## Progress

### v4 Step 1 — scaffold (commit `432e606`, 2026-05-20) ✅
- Added `deltanet_chunked_neumann_tp(state, x_seq, dn, cfg, seq_len)` as a
  STUB that internally loops per-position calling `deltanet_step_tp`.
- Added `state.use_chunked_dn` flag (default False → no behavior change).
- v3 prefill's DN block routes through the stub when flag is set.
- Probe gains `--use-chunked-dn` for A/B testing.
- Validated: stub gives cos identical to v3 (trivially — same math).

### v4 Stage 1 — batched pre-norm + in_proj (commits `8c9488c` + `cea8641`, 2026-05-20) ✅
- Refactor: extracted `_deltanet_step_tp_from_inproj` helper (stages 3-11 of DN).
- Chunked stub computes batched `h_seq = rms_norm(x_seq)` + `all_seq = linear(h_seq, w_in)` ONCE, then per-position loop slices `all_seq[pos]` and calls helper.
- Cos validated **bit-identical** to v3.
- Perf trade-off: slower at short seq (slice overhead), faster at long seq.

### v4 Stage 3 — batched decay/gate (commit `f4dbe06`, 2026-05-20) ✅
- Helper extended with `precomputed_decay` + `precomputed_beta` params.
- Chunked stub batches the a/b slice + decay/gate computation (manual softplus path; owned_decay_gate kernel is single-pos only).
- Cos bit-identical-within-bf16-noise to v3.

### v4 cumulative perf table

```
                          seq=5            seq=32           seq~87
v3 baseline (no stages)   1264 ms          6089 ms          28230 ms (regresses vs decode-loop @ 22308!)
v3 + Stage 1              2427 ms (-92%)   5841 ms (+4%)    -
v3 + Stage 1+3            2808 ms (-122%)  5097 ms (+16%)   15162 ms (+46% vs v3, +36% vs decode-loop)
```

**Top1 agreement (vs decode-loop):** 5/5 (seq=5), 27/32 (seq=32), 69/87 vs v3's 72/87 (seq=87 — ~4% noise from fp accumulation order, neither is "more correct" without HF gold).

**Key finding at seq=87:** v3 alone is SLOWER than decode-loop reference (28s vs 22s). v3+Stage 1+3 is FASTER than decode-loop by 36%. This is the first config that delivers real prefill speedup at long context — the daily-driver use case.

**Stage 2 (conv1d batching)** deferred to a later session due to state-coupling complexity. Stage 4 (Neumann recurrence) is next; that's where DN per-position loop disappears entirely.

### Stage stages summary (updated)

| Stage | Status | Cumulative perf at seq=87 |
|---|---|---|
| 1 (pre-norm + in_proj batched) | ✅ done | +6% vs v3 alone (extrapolated) |
| 2 (conv1d batched) | ⏸️ deferred | TBD |
| 3 (decay/gate batched) | ✅ done | +46% vs v3 cumulative |
| 4 (Neumann (I-attn)⁻¹ chunked recurrence) | next | huge — eliminates per-pos loop entirely |
| 5 (multi-chunk loop + state thread) | future | enables 32k+ context |
| 6 (batched output gate) | future | small cleanup |

API surface and harness locked in. Future stages just replace one piece at a time.

## Next steps when picking this up

Replace the STUB body in `deltanet_chunked_neumann_tp` (in `server_tp.py`,
right after `deltanet_step_tp`) stage by stage. Each stage = one commit +
validate via `--use-chunked-dn` flag on the probe.

### Stage 1 — batched pre-processing (no Neumann yet)
Trivial batched ops. No state interaction. Should give cos = same as v3
because the math is identical, just batched.

In the chunked function, replace per-position loop with:
```python
# Batched pre-norm + in_proj
h_seq = _rms_norm_manual(x_seq, dn['input_norm'], EPS, HIDDEN)  # [C, HIDDEN]
all_seq = ttnn.linear(h_seq, dn['w_in'])                         # [C, IN_PROJ_OUT_CHIP]
# Batched slice — per-position is already a column slice
mixed_qkv_seq = ttnn.slice(all_seq, [0, 0], [C, CONV_DIM_CHIP])
z_seq = ttnn.slice(all_seq, [0, CONV_DIM_CHIP], [C, CONV_DIM_CHIP + VAL_DIM_CHIP])
a_seq = ttnn.slice(all_seq, [0, CONV_DIM_CHIP + VAL_DIM_CHIP], [C, ...])
b_seq = ttnn.slice(all_seq, [0, ...], [C, ...])
# Then per-position loop for the rest (conv1d, decay, recurrence, output gate)
```

Validation: cos ≥ 0.9997 vs full per-position. Should be trivial.

### Stage 2 — batched conv1d
Replace per-position conv1d call with batched 1D conv over `[C + KERNEL-1, CONV_DIM_CHIP]`
input. Need to prepend conv state taps (`conv_st`) as left padding.

Risk: `ttnn.conv1d` may not exist or may have shape constraints. Fallback:
materialize the kernel shifts via `ttnn.slice` + `ttnn.mul` + `ttnn.sum`,
batched across positions.

### Stage 3 — batched decay/gate computation
For each position, compute `g_t = -exp(A_log) · softplus(A_t + dt_bias)` and
`β_t = sigmoid(B_t)`. Batched form: input shape `[C, NV_PER_CHIP]`, output
shape `[C, NV_PER_CHIP]`.

Per C'5 plan: cumsum the `g` to get `G = cumsum(g)` per chunk. Use
`ttnn.cumsum` (validated by `feedback_c5_primitives_green`).

### Stage 4 — the Neumann (I-attn)^{-1} chunked recurrence

This is THE algorithmic stage. Per C'5 plan §1:
```
K_β = β · K                                # [C, K_DIM]
V_β = β · V                                # [C, V_DIM]
G = cumsum(g)                              # [C]
D = exp(G[:,None] - G[None,:]) ⊙ lower_tri # [C, C]
attn = -(K_β @ K.T) ⊙ D                    # [C, C]
T = (I - attn)^{-1}  ← Neumann factorization, batched per V-head
V_prime = T @ V_β
K_prime = T @ (K_β * exp(G)[:,None])
A = (Q @ K.T) ⊙ D
v_prime = K_prime @ S_prev                 # [C, V] from entering SSM state
v_new = V_prime - v_prime
attn_int = (Q * exp(G)[:,None]) @ S_prev   # [C, V]
O = attn_int + A @ v_new                   # [C, V] chunk output
S_new = exp(G[-1]) * S_prev + (K * exp(G[-1] - G)[:,None]).T @ v_new
```

Implement in fp32 per `feedback_c5_primitives_green`. Use
`experiments/utils/neumann_inverse_probe.py:neumann_inverse_ttnn` as the
reference impl for the inverse.

### Stage 5 — multi-chunk loop + state threading

Once single-chunk works at C=8, add chunk loop:
- Process `seq_len` in chunks of `C` (default 8 → 32 → 64).
- Carry `S` (SSM state) across chunks.
- Last chunk may be partial (pad to C, mask out junk rows in output).

### Stage 6 — output gate + residual

Final stage: per-head `rms_norm` + `silu(z)` gate + `w_out` projection.
These are already in the existing per-position fallback; batched form is
straightforward.

### Stages summary

| Stage | What | Risk | Gate |
|---|---|---|---|
| 1 | Batched pre-norm + in_proj + slice | low | cos ≥ 0.9997 vs per-pos |
| 2 | Batched conv1d | medium (op semantics) | cos ≥ 0.9997 |
| 3 | Batched decay/gate + cumsum | low (primitives validated) | cos ≥ 0.9997 |
| 4 | Neumann (I-attn)^{-1} recurrence | HIGH (algorithmic) | cos ≥ 0.9997 at C=8 |
| 5 | Multi-chunk loop + state thread | medium | cos ≥ 0.9997 at C=64, seq=128 |
| 6 | Batched output gate | low | end-to-end cos ≥ 0.9997 |

Each stage = ~1 day-arc + 1-2 bootstraps. Total: 3-5 day-arcs to ship v4.
