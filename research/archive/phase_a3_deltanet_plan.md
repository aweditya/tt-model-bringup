# Phase A3 — Gated DeltaNet (Isolated Kernel)

Non-negotiables for this phase, in order:
1. **Plan first** — this file.
2. **Numpy fp32 reference BEFORE ttnn.** Per `feedback_numpy_reference.md`: HF AutoModel crashes on remote; we always own the reference.
3. **Cosine ≥ 0.99 BEFORE perf work.** Per `feedback_correctness_first.md`.
4. **Single device, device 0.**
5. **Permanent script, run on qb1 via SSH, no /tmp, no inline.**
6. **Single-chip utilization metric.** Report µs/call AND % of memory ceiling.

## What we're isolating

The CORE delta-rule recurrence at decode shape (T=1), per the HF
`Qwen3_5MoeGatedDeltaNet` source (see `research/qwen36_modeling_excerpts.md`):

```
# Inputs (all bf16 on device, except state in fp32):
#   q, k:  [B, n_v_heads=32, d_k=128]
#   v:     [B, n_v_heads=32, d_v=128]
#   g:     [B, n_v_heads=32]   (decay scalar per head)
#   beta:  [B, n_v_heads=32]   (delta-rate scalar per head)
#   H_prev:[B, n_v_heads=32, d_k=128, d_v=128]  (fp32 recurrent state)

# 1) L2 normalize Q and K (use_qk_l2norm_in_kernel=True in HF)
q = q / (||q|| + eps);   k = k / (||k|| + eps)
# 2) Decay state
H_decayed = H_prev * exp(g)               # broadcast g over (d_k, d_v)
# 3) Read current V via K
kv_mem = (H_decayed * k[..., :, None]).sum(axis=-2)         # [B, H, d_v]
# 4) Delta correction
delta = (v - kv_mem) * beta[..., None]                       # [B, H, d_v]
# 5) Update state: H += outer(K, delta)
H_new = H_decayed + k[..., :, None] * delta[..., None, :]    # [B, H, d_k, d_v]
# 6) Read output via Q
out = (H_new * q[..., :, None]).sum(axis=-2)                 # [B, H, d_v]

# Returns: out, H_new
```

**This is THE recurrence.** We are NOT testing in-projections, conv1d, or output-projection in this experiment — those are bog-standard ops we already use elsewhere and we'll wire them up in Phase B. A3 is purely about the unique-to-DeltaNet recurrence.

## What we deliberately exclude from A3

| | Excluded because |
|---|---|
| `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` | Standard `ttnn.linear`. No new physics. |
| `conv1d` kernel=4 | Only used in prefill. Phase A6 covers it. For decode T=1 it's just a state-update on the conv state which is trivial. |
| `silu(z)` output gate, `out_proj(4096→2048)` | Standard ops. |
| GQA replication of K-heads from 16 to 32 | Just a `repeat` — no new physics. Test assumes inputs are already replicated. |
| Multi-step (prefill) recurrence | Phase A6. |

This keeps A3 focused. If A3 passes and Phase B integrates fine, we know the recurrence is correct in isolation.

## Test plan

### Step 1 — Numpy reference (`deltanet_step_numpy`)

Reference implementation in plain numpy fp32. Single time step. ~30 lines.
**Self-test:** verify the H state is exactly recoverable from inputs (i.e., the recurrence is deterministic). Property-test with random inputs and `H_prev=0`.

### Step 2 — ttnn implementation (`deltanet_step_ttnn`)

Same shape conventions. Watch:
- State H must be fp32 (per `mamba_ssm_dtype: float32`). Use `dtype=ttnn.float32` on H tensor.
- Q, K, V are bf16. Multiplication of bf16 × fp32 → need to verify ttnn handles this or up-cast.
- Broadcast patterns:
  - `H * k[..., None]` where H is [..., d_k, d_v] and k is [..., d_k]: broadcast last dim
  - `k[..., None] * delta[..., None, :]`: outer-product via two 4-D tensors
- ttnn's L2-norm — check if `ttnn.normalize` exists or compose from `rsqrt(sum(x*x))`.

### Step 3 — Cosine check

Random inputs (seed-fixed), random H_prev, run both. Compare:
- Cosine(out_np, out_tt) ≥ **0.99**
- Cosine(H_np, H_tt) ≥ **0.99**

If cosine < 0.99: ablate. Test pieces in order — L2 norm only, then add decay, then add delta, etc.

### Step 4 — Perf measurement

`bench(label, fn)` 200 iters, median + p90.
Report:
- **Time per call** (µs)
- **% of memory-bandwidth ceiling on one chip:**
  - Read H_prev: 32 × 128 × 128 × 4 bytes (fp32) = 2 MB
  - Write H_new: 2 MB
  - Tiny Q/K/V/g/β: negligible
  - Total ~4 MB / 450 GB/s = **9 µs memory floor**
  - Target: ≤ 50 µs eager, ≤ 20 µs traced

If we're stuck > 200 µs eager, fusion / sharding may be needed.

## Open questions to resolve while implementing

1. **fp32 in ttnn on Blackhole** — do `ttnn.float32` tensors work for `.exp` / `.sum` / element-wise mul? My memory says they should but it's worth a quick smoke test before writing the full kernel.
2. **L2 norm in ttnn** — `ttnn.rms_norm` is close but not the same (RMS uses mean(x²), not sum(x²) with sqrt). Need either `ttnn.l2_normalize` or compose manually.
3. **Outer product via broadcast** — does `ttnn.mul([..., d_k, 1], [..., 1, d_v])` broadcast correctly to `[..., d_k, d_v]`? If not, need an explicit `ttnn.outer` or reshape+matmul.

## Stopping rules

- Cosine ≥ 0.99 + perf measured: ship A3, move to A4.
- Cosine < 0.99 after 2 hours of ablation: stop, document the specific failure (e.g. "fp32 sum in ttnn drifts to bf16 internally"), revise A3 plan.
- µs/call > 1ms eager: stop, identify the slow op, decide whether to fix it now or move on.

## Output

- `experiments/82_gated_deltanet.py` — single permanent file with numpy ref + ttnn impl + tests + bench
- `research/phase_a3_deltanet_results.md` — final numbers, what we learned
- Commit at: (1) numpy ref + self-test, (2) ttnn passes cosine, (3) perf measured
