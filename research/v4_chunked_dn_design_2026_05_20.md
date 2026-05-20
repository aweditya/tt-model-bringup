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

## Next steps when picking this up

1. Re-read `research/c5_chunked_prefill_plan.md` end-to-end
2. Find `feedback_c5_primitives_green` memory note for the validated primitives
3. Scaffold `deltanet_step_tp_chunked(state, x_seq, dn, cfg, seq_len)` with C=8 single-chunk first
4. Probe vs `deltanet_step_tp` per-position output, gate cos ≥ 0.9997
5. Iterate
