# Post owned_gdn Profile + Next Fusion Target Menu — 2026-05-18 evening

First op-attribution profile of the qb2 decode body with the custom GDN kernel
defaulted (`server_tp.py:115` → `"owned_gdn"`, commit `26cad39`). Eager-proxy
profile via `profile_decode_tp_ops --timed --deltanet-recurrence-mode owned_gdn`
on the canonical "The capital of France is" prompt.

Artifact: `.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_default_20260518_1645.json`

## Headline

The owned_gdn kernel collapsed DeltaNet recurrence from the #1 cluster
(128 ms pre-fusion per the 2026-05-15 profile) to #12 (28.8 ms). The
fusion worked exactly as designed. With recurrence dethroned, the new
top clusters are:

| Category | Count | ms | % | Comment |
|---|---|---|---|---|
| DeltaNet_decay_gate | 480 | 99.32 | 13.52% | softplus + neg-exp chain (10 ops/layer × 48 layers) |
| matmul | 321 | 90.57 | 12.33% | unavoidable for in/out projections |
| DeltaNet_other | 384 | 86.26 | 11.74% | conv-state mgmt + slice/reshape + qkv prep |
| DeltaNet_conv | 288 | 83.00 | 11.30% | 4-tap depthwise conv + state shift |
| DeltaNet_qkv_repeat | 336 | 60.38 | 8.22% | GQA repeat-interleave |
| RoPE | 320 | 54.68 | 7.44% | manual rotate-only path |
| RMSNorm | 305 | 45.05 | 6.13% | pre + post + l2 + final norms |
| collectives | 129 | 43.52 | 5.92% | all_reduce after MLP/attn/output |
| attention_other | 320 | 43.39 | 5.91% | gated_attn plumbing |
| DeltaNet_output_gate | 240 | 43.21 | 5.88% | linear_attn norm + silu(z) gate |

Top single ops (cross-category):

| Op | Count | ms | Comment |
|---|---|---|---|
| ttnn.slice | 593 | 119.18 | layout shuffling, often consecutive with reshape |
| ttnn.reshape | 659 | 89.68 | pure layout, no compute |
| ttnn.linear | 321 | 90.57 | real matmul work |
| ttnn.add | 304 | 76.92 | concentrated in decay_gate, output_gate |
| ttnn.mul | 288 | 70.91 | same |
| ttnn.all_reduce | 128 | 42.82 | 2 per layer × 64 layers |
| ttnn.concat | 112 | 31.56 | conv-state shift, k/v concat |
| ttnn.exp | 144 | 29.09 | decay_gate softplus + exp(A_log) |
| ttnn.repeat | 96 | 26.23 | GQA repeat |

> Caveat: eager-proxy ≠ trace time. The 734 ms eager-proxy total compresses
> to ~80 ms in the actual production trace (~9× compression). Categories
> compress at different rates — small-op clusters compress more than
> matmul. Use this table to rank candidates, not to project savings.

## Ranked next-fusion candidates

### Candidate 1 — **DeltaNet decay/gate fusion** (biggest cluster)

The code in `server_tp.py:681-690`:
```python
a_biased  = ttnn.add(a_tt, dn['dt_bias'])
softplus_a = ttnn.log(ttnn.add(ttnn.exp(a_biased), 1.0))   # OR ttnn.softplus(a_biased)
g          = ttnn.mul(ttnn.neg(ttnn.exp(dn['A_log'])), softplus_a)
beta       = ttnn.sigmoid(b_tt)
decay      = ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1])
```

10 ops × 48 layers = 480 calls. A single custom op
`qwen36_decay_gate(a, b, dt_bias, A_log) → (decay, beta)` would collapse
this to 48 calls.

Pros:
- Largest single category (13.52% of eager time)
- Math is straightforward elementwise + softplus + exp
- Same scaffolding pattern as `qwen36_gdn_decode_owned`

Cons:
- Native softplus probe broke 20-token identity earlier — need careful
  token-level validation
- Math involves transcendentals (exp/log) which can shift bf16-ULP-scale

### Candidate 2 — **DeltaNet conv1d fused kernel** (best-studied)

`server_tp.py:565-573`. 4-tap depthwise conv + state-shift. Already
deeply analyzed in `feedback_conv1d_diagnosis.md`: 65% sum-reduce, 21%
state-mgmt (concat+slice), 13% mul. Estimated 30+ ms/tok savings with a
custom kernel.

Pros:
- Best-studied target; prior analysis identified the bottleneck precisely
- Bigger projected savings (30+ ms eager-proxy = ~3 ms trace estimate)
- Clean fusion candidate (depthwise conv + ring shift)

Cons:
- More complex than decay_gate (needs ring buffer or sliding-window kernel)
- Prior `feedback_conv1d_circular_buffer.md` rejected two approaches
- Estimated 2-3 weeks

### Candidate 3 — **DeltaNet QKV repeat / prepare fused op** (existing reference)

`server_tp.py:589-599` does GQA repeat-interleave via reshape + repeat +
reshape. 336 calls / 60 ms.

Pros:
- Friend's `qwen36_gdn_prepare_decode` op already exists at
  `experiments/.refs/tt-qwen-36/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_prepare_decode/`
- Reference implementation reduces this dramatically (per earlier survey)
- Could be ported faster than a from-scratch op

Cons:
- Medium savings (8.22%)
- Friend repo has known errors elsewhere — port cautiously

### Candidate 4 — **Layout-shuffle audit** (no custom kernel, refactor)

Slice + reshape together are 209 ms (28% of eager-proxy time) and do no
real compute. A careful audit of `deltanet_step_tp` could eliminate
redundant slices via better up-front tensor partitioning.

Pros:
- No new custom op; just Python refactor
- Potentially fast win (days, not weeks)
- Touches all categories (not just DeltaNet)

Cons:
- Hard to project savings without doing it
- Some slices are structurally required (vocab sharding, head splitting)
- Easy to break things if shape contracts change

## Recommendation

Either **Candidate 2** (conv1d — bigger projected savings, prior analysis
done) or **Candidate 1** (decay/gate — biggest current cluster, simpler
scaffold). Choose based on appetite for kernel-engineering complexity vs
biggest single-cluster cleanup.

Defer **Candidate 4** (layout audit) until after one more custom-op
fusion lands — the slice/reshape count will likely drop naturally as
more compute paths get fused into single kernels.
