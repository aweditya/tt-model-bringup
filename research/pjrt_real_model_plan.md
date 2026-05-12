# PJRT Real-Model Test Plan — Qwen2.5-0.5B end-to-end

Date: 2026-05-11
Author: PJRT track (Opus, autonomous)

## The question

"The best test for the PJRT plugin is if we're able to run an actual
model." — does our PJRT plugin compile and execute a real transformer
that produces coherent text, and how fast vs the native ttnn port we
have at 142 tok/s for Qwen2.5-0.5B?

## Target

Qwen2.5-0.5B (validated as canonical model on qb1; 24 layers, GQA
14q/2kv heads, head_dim=64, hidden=896, vocab=151936). Reference port:
`experiments/60_native_rope_decode.py` hits 142 tok/s with native
ttnn + paged KV cache + native RoPE.

## Design decisions

### 1. Decode-only, prefill on CPU/numpy

Prefill needs variable-length sequences; our PJRT plugin's trace cache
keys on bytecode hash and requires fixed shapes per trace. Prefilling
the prompt ("The capital of France is" = 5 tokens) through the PJRT
trace is one StableHLO program **per prompt length** — wasteful for a
single test. Strategy:

- **Prefill**: pure numpy reference (built from
  `experiments/76_8b_numpy_reference.py` patterns). Populates the
  initial KV cache state.
- **Decode**: ONE `jax.jit`'d function `decode_step(token_id, pos,
  k_cache, v_cache, ...weights...) -> (logits, k_cache, v_cache)`,
  fixed shape `[1, 1, hidden]` for hidden state. Run through PJRT plugin.

This is the same split the native ttnn port uses (prefill via numpy
RoPE + ttnn matmul, decode via a pure-device trace).

### 2. KV cache: per-call carry, not in-place scatter

The PJRT plugin's engine has `scatter` support (via numpy roundtrip
through `_execute_scatter_device`). But for our test:

- KV cache shape: `[1, n_kv_heads=2, MAX_SEQ=64, head_dim=64]` per
  layer × 24 layers. We'll start with MAX_SEQ=64 (prompt 5 + 100
  decode tokens ≤ 64? actually 105 — bump to 128).
- Each decode step: `k_cache = k_cache.at[:, :, pos:pos+1, :].set(new_k)`
  → JAX lowers to `stablehlo.scatter`. Same shape for every step (pos
  bound at trace time as an input). The engine's `scatter` goes
  through a numpy roundtrip → host-transfer op → **not trace-eligible**
  per current `_HOST_TRANSFER_DEVICE_OPS`. So the program falls back to
  parse-cached eager — slow but correct.

**Alternative**: pass `pos` as a scalar **input** (not baked into the
trace), do `dynamic_update_slice` — JAX lowers this to scatter anyway.

**Decision**: don't try to make the KV update trace-safe in v0. Accept
parse-cached eager (~2-3 ms per call from `e2e: softmax` floor scaled
by op count). The test still validates correctness and gives a real
end-to-end number.

If too slow, fall back: each step returns `(logits, new_k_per_layer,
new_v_per_layer)` and the host (Python) does the scatter as numpy on
the returned `k_cache`. The scatter then happens **outside** the JAX
program and is free.

**Final decision**: do option B (host-side scatter). Decode step is:
- inputs: `token_id [1]`, `pos_scalar [1]`, `k_caches_concat`,
  `v_caches_concat`, weights (all device-pinned).
- outputs: `logits [vocab]`, `new_k [24, 2, 1, 64]`, `new_v [24, 2, 1, 64]`.

Host writes new_k/new_v into the cache at `pos` (as numpy) and feeds
back next call. This keeps the JAX program shape-stable AND trace-eligible.

### 3. RoPE

JAX-side, pure jnp:

```python
# pos is a scalar int (or [1] tensor)
angles = pos * freqs  # [head_dim/2]
cos = jnp.cos(angles)
sin = jnp.sin(angles)
# half-format rotate
def rope(x):  # x: [n_heads, head_dim]
    x1, x2 = jnp.split(x, 2, axis=-1)
    rot = jnp.concatenate([-x2, x1], axis=-1)
    cos_b = jnp.concatenate([cos, cos], axis=-1)
    sin_b = jnp.concatenate([sin, sin], axis=-1)
    return x * cos_b + rot * sin_b
```

JAX lowers this to `multiply`, `add`, `negate`, `concatenate`,
`slice` — all of which our engine handles (with `concatenate` and
`slice` being host-transfer in trace, but we don't trace them anyway
because of the scatter issue).

### 4. Attention

```python
scores = q @ k.transpose(0, 1, 3, 2) / sqrt(head_dim)  # [B, H, 1, T]
# causal mask: pos is the current token's position; mask out future
mask_idx = jnp.arange(MAX_SEQ) > pos  # [T] -> True for future
scores = jnp.where(mask_idx, -1e9, scores)
probs = jax.nn.softmax(scores, axis=-1)  # JAX lowers to max/sub/exp/sum/div
attn = probs @ v
```

For GQA (14 q-heads vs 2 kv-heads), reshape q to `[B, n_kv_groups=2,
n_q_per_group=7, T_q=1, head_dim=64]` and broadcast k/v.

### 5. The big shape that matters

Decode step processes ONE token. Hidden=896, n_q=14, n_kv=2,
head_dim=64. Cache is MAX_SEQ=128. The biggest single matmul is the
LM head at the end: `[1, 896] @ [896, 151936]` = `[1, 151936]` — fine.

## Op coverage check

From `pjrt_plugin/tests/inspect_transformer_decode.py` we already know
the decode-step lowering uses: add, multiply, divide, sqrt, rsqrt,
exp, reduce(add+max), broadcast_in_dim, reshape, transpose,
dot_general, slice, compare, select, iota, concatenate, scatter, gather,
constant. **All supported in our engine.**

## Risks & known issues

1. **Scatter not trace-eligible** — host roundtrip per call. Mitigation:
   external scatter (option B above).
2. **Softmax not fused** — 2.08× vs vanilla per the bench. Mitigation:
   accept for v0; potentially do the fusion later if model is too slow.
3. **151936 vocab is huge** — `argmax` on `[1, 151936]` goes through a
   numpy roundtrip (`_execute_reduce_argmax_device`). One transfer per
   step, ~1ms. Acceptable.
4. **24 layers × ~10 ops/layer = ~240+ ops per decode**. Trace replay
   floor is ~150us per op-block — best case ~30-40 ms/step ≈ 30 tok/s.
   Worst case (no trace, parse-cached eager): 24 × softmax × 3ms
   ≈ 70 ms/step ≈ 14 tok/s.

## Plan of execution

1. Write `experiments/jax_qwen05b_pjrt.py`:
   - Load weights (HF safetensors)
   - Numpy prefill that populates k_cache, v_cache numpy arrays
     (shape `[n_layers, 1, n_kv_heads, MAX_SEQ, head_dim]`)
   - `jax.jit` decode step
   - Greedy decode loop in Python, host-side cache update
   - Compare text to native ttnn (run `experiments/60` reference text)
   - Measure tok/s
2. Run on qb1
3. Iterate if broken; otherwise report

## Success criteria

- Generates coherent text (≥10 tokens match native ttnn or are
  meaningfully coherent English continuation)
- Reports tok/s end-to-end (full decode loop time)
- Reports breakdown of where time goes (per-step parse vs replay vs
  argmax)

## Optional follow-up

Softmax pattern-match fusion in the engine. Defer until base flow works.
