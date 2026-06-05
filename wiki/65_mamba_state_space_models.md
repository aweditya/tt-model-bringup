# 65. Mamba and state-space models — what the math is and why we need a kernel

**Audience**: you know transformer attention end-to-end (we shipped 27B,
35B, Gemma 4); you've watched a custom kernel get built (35B
`qwen36_gdn_decode_owned`). You're about to bring up Nemotron-3 Nano,
which has 23 / 52 layers in **Mamba2 SSD** form, and `tt-metal` does
not ship an SSD primitive — we will author one.

This page is a precise pedagogical primer. By the end you should:
1. Know what an SSM is in one paragraph;
2. See the math derivation from a linear ODE to Mamba2's per-token decode step;
3. Understand the "duality" that lets training use matmuls while inference uses a recurrence;
4. Be able to read `modeling_nemotron_h.py` and recognise every line.
5. Know exactly what the kernel we will write must compute.

> **Resources cited**: this primer aggregates Tri Dao's
> [Mamba2 blog series](https://tridao.me/blog/2024/mamba2-part1-model/),
> the Nemotron-H paper ([arXiv 2504.03624](https://arxiv.org/abs/2504.03624)),
> our own
> [`research/nemotron3_nano_architecture_brief.md`](../research/nemotron3_nano_architecture_brief.md),
> and the `state-spaces/mamba` reference implementation. Read those for
> depth; this primer is the bridge.

------------------------------------------------------------------------

## 1. The one-paragraph intuition

A transformer treats a sequence as an **explicit graph of pairwise
interactions** — every new token re-attends to every prior token,
held verbatim in a KV cache of size O(T). A state-space model
**summarises the past into a fixed-size hidden state** — every new
token reads a single state `h ∈ R^N`, updates it, writes it back. The
state is the model's memory; nothing else from prior tokens is kept.
You trade attention's ability to address any past token directly for
the SSM's promise of *O(1) memory per token, regardless of T*. The
trick is choosing the state-update math so that `h` keeps "the right
information" for downstream prediction.

That choice — the state-update math — is what S4 → S5 → Mamba (S6)
→ Mamba2 (SSD) iterates on.

------------------------------------------------------------------------

## 2. Where SSMs come from (10-minute lineage)

### 2.1 The classical continuous SSM

A linear time-invariant continuous-time system is the ODE
```
dh(t)/dt = A · h(t) + B · u(t)
   y(t)  = C · h(t) + D · u(t)
```
where `h ∈ R^N` is the latent state, `u ∈ R` is the input signal,
`y ∈ R` is the output, and `A ∈ R^{N×N}`, `B ∈ R^{N×1}`,
`C ∈ R^{1×N}`, `D ∈ R` are learned. Read it as: *the state changes
linearly based on its current value and the current input; the output
is a linear projection of the state plus a direct feedthrough.* This
is the standard control-theory model of an LTI system — the same math
your EE friends use for filters.

### 2.2 Discretization

You can't feed a transformer-style discrete token sequence into a
continuous ODE. So we **discretize**: pick a step size Δ and
approximate
```
h_k = Ā · h_{k-1} + B̄ · u_k
y_k = C  · h_k    + D · u_k
```
where `Ā = exp(Δ·A)` (matrix exponential) and `B̄ = (Δ·A)^{-1}·(Ā − I)·Δ·B`
(zero-order-hold formula; you can also use bilinear / Euler). The
zero-order-hold ZOH approximation is just "assume `u` is constant
within each Δ, integrate exactly".

Note: Δ is a per-token learned parameter in Mamba — that's the
"selective" part below.

### 2.3 S4 (2021): structure for speed

The discretized recurrence is exact but **slow to train** at sequence
length T because you have a sequential chain of `h_k = Ā · h_{k-1} +
B̄ · u_k`. Gu et al.'s **S4** insight: if `A` has a special structure
(HiPPO-LegS diagonal-plus-low-rank), you can rewrite the recurrence as a
**convolution** `y = K̄ * u` where `K̄` is a closed-form long kernel.
Convolutions parallelise on GPUs. So S4 trains in O(T log T) via FFT
convolution while still summarising long sequences.

S4's three big knobs (each gets a name in later models):
- **A** is parameterised as a diagonal `A_log = log(−A_diag)` (so the
  effective `A = −exp(A_log)` stays negative-definite → stable
  decay; we'll see this exact pattern in Mamba2).
- **Initial values** of A from HiPPO theory (chosen so the SSM
  approximates a polynomial decomposition of the input over a long
  window).
- **Diagonal A** restricts expressivity but makes the per-channel
  state independent.

### 2.4 Mamba (S6, 2023): selectivity

S4's `A`, `B`, `C` are *input-independent* — same coefficients for
every token. That's a problem: the model can't choose what to
remember. Mamba (= S4 + Selection) makes `B`, `C`, and Δ **functions
of the input**:
```
B_k = LinearB(u_k)        # was constant
C_k = LinearC(u_k)        # was constant
Δ_k = softplus(LinearΔ(u_k) + bias)   # was constant
```
A stays input-independent and (per channel) diagonal-with-scalar.
Now the recurrence becomes
```
h_k = Ā_k · h_{k-1} + B̄_k · u_k                            (*)
y_k = C_k · h_k
```
with `Ā_k = exp(Δ_k · A)`, `B̄_k = Δ_k · B_k · u_k` (ZOH simplified
when A is scalar). This is selective: at each step the model decides
how much to retain (Δ small → Ā ≈ I → state persists) vs reset (Δ
large → Ā near zero → state forgotten).

**But** input-dependent `B`, `C` break the convolution trick — we
can no longer write `y = K̄ * u` because each token's kernel is
different. Mamba's contribution was a hardware-aware
**parallel-scan kernel** for the recurrence (*) that's fast enough
to train on GPUs without convolution.

### 2.5 Mamba2 (SSD, 2024): scalar A and the matrix dual

Mamba2 takes one more structural cut: `A` is restricted further to a
single learned **scalar per head** (multiplied by the identity).
That's weaker than S4/Mamba's diagonal, but it has a magic
consequence: the per-token recurrence (*) can be rewritten as a
**structured matrix multiplication** between input and output. This
is the **state-space duality** (SSD), the headline result of the
Mamba2 paper. Two formulations of the same function:

- **Recurrence form** (inference-friendly, O(1) per token):
  ```
  h_t = a_t · h_{t-1} + B_t · u_t        h ∈ R^N, scalar a_t
  y_t = C_t · h_t
  ```
- **Matrix form** (training-friendly, all matmuls):
  ```
  M_{ij} = (∏_{k=j+1}^{i} a_k) · ⟨C_i, B_j⟩   for j ≤ i, else 0
  y     = M · u                              [T × T] · [T × 1]
  ```

Where the `∏ a_k` factor is a **cumulative product of scalars**
along the lower triangle. With all `a_k = 1`, M collapses to a
plain `C · B^T` causal mask — i.e. **linear attention without
softmax**. Mamba2 is exactly that, with input-dependent decay
multipliers along the diagonal.

The "duality" is the magic: **same model, two computation paths**.
You train with the matrix form (matmul-friendly on GPUs), you decode
with the recurrence form (O(1) per token). This is why SSD is fast on
both ends; S4/Mamba had to choose.

------------------------------------------------------------------------

## 3. Mamba2 SSD math — the decode step, precisely

What we will implement on Blackhole is the decode (single-token) path
of equation (*) above, broadcast over heads/groups. Pseudocode for
ONE decode token, per head h ∈ [0, num_heads), inputs all bf16
except `ssm_state` in fp32:

```python
# State (fp32 on device, persisted across tokens):
ssm_state[h, d, s]            ∈ R^{head_dim, ssm_state}   # h=head, d=channel, s=state

# Per-step inputs (bf16, from this token's projections):
x[h, d]                       ∈ R^{head_dim}             # input projected to head channels
z[h, d]                       ∈ R^{head_dim}             # gate (multiplied at output)
dt[h]                         ∈ R                          # raw dt scalar per head
B[g, s]                       ∈ R^{ssm_state}             # group-broadcast B  (g = h // heads_per_group)
C[g, s]                       ∈ R^{ssm_state}             # group-broadcast C

# Learned per-head parameters (bf16, fixed across tokens):
A_log[h]                      ∈ R          # so that A = -exp(A_log[h])  (stable: A < 0)
dt_bias[h]                    ∈ R
D[h]                          ∈ R          # direct skip connection

# ────── one decode step ──────
# 1. Time-step discretization (fp32 throughout)
dt_eff = softplus(dt[h] + dt_bias[h])
dt_eff = clamp(dt_eff, time_step_floor, time_step_max)
A      = -exp(A_log[h])                    # scalar; A < 0 so |exp(dt*A)| < 1 → stable

# 2. State update (fp32 accumulator; broadcast B over channels)
decay = exp(dt_eff * A)                    # scalar
input_gain = dt_eff * B[g, :]              # shape [ssm_state]
for d in range(head_dim):                  # channel loop
    ssm_state[h, d, :] = decay * ssm_state[h, d, :] + input_gain * x[h, d]

# 3. Output projection (fp32 reduce, bf16 result)
for d in range(head_dim):
    y[h, d] = sum_s(C[g, s] * ssm_state[h, d, s]) + D[h] * x[h, d]

# 4. (Outside the kernel) MambaRMSNormGated: norm y in groups of head_dim, then gate by z, then out_proj
```

The key shapes for Nemotron-3 Nano (from the architecture brief):
```
num_heads = 64
head_dim  = 64
ssm_state = 128
n_groups  = 8   ⇒ heads_per_group = 64 / 8 = 8
```

So per head the kernel does `head_dim × ssm_state = 64 × 128 = 8192`
fused multiply-adds for the state update, plus another `head_dim ×
ssm_state = 8192` for the output reduce. Per token, per layer, 64
heads × 16384 = ~1M FMAs. Across 23 Mamba layers: ~24M FMAs per token
just from the SSM recurrences.

For comparison, a single decode step of a 27B-style attention layer
with KV cache reads `2 × NKV × head_dim × cur_pos × 2 (K and V)` bf16
operands; at cur_pos=8K, NKV=4, head_dim=128 that's ~16 MB of memory
traffic per layer per token. Mamba2's per-step cost is **constant in
context length** — that's the entire point.

------------------------------------------------------------------------

## 4. The selectivity intuition — what does `dt` actually do?

This is the part that took me longest to "get" when reading the
Mamba2 paper, so I'll spell it out.

`dt` (also written Δ) is the **discretization step size** of the
underlying continuous-time ODE. Recall:
- `Ā = exp(dt · A)` and `A < 0` → `exp(dt · A) ∈ (0, 1)`.
- `Ā` close to 1 ⇒ state mostly **persists** to next step (slow decay).
- `Ā` close to 0 ⇒ state mostly **resets** (fast decay).

Mamba makes `dt` a function of the input token. So:
- Token says something **important to remember** → linear-Δ projection
  outputs a small value → `Ā ≈ 1` → state persists → memory of this
  context section preserved.
- Token says something **boring / a separator** → linear-Δ projection
  outputs a large value → `Ā ≈ 0` → state largely cleared → ready
  for a new context.

That's why it's called *selective*: the model **selects what to
remember and what to forget**, per token, based on what the current
token is. This is the soft equivalent of attention's "head 7 looks
mostly at the previous token; head 12 looks across the whole
context".

Without selectivity (S4), the same decay applies regardless of input
— the model can summarise *long-range trends* well (HiPPO-style
polynomial basis) but can't *selectively remember* a specific entity
mentioned 1000 tokens ago.

------------------------------------------------------------------------

## 5. How Nemotron-3 wires Mamba2 into the model

Per the [architecture brief](../research/nemotron3_nano_architecture_brief.md)
§4.3, one Mamba2 layer of Nemotron-3 does this end-to-end:

```python
# h: [B, 1, 2688] (hidden state at one token)
zxbcdt = in_proj(h)                              # [B, 1, ~10304]
z, xbc, dt = split(zxbcdt, [4096, 4096+2·1024, 64])
                                                  # z: [B,1,4096]  gate
                                                  # xbc: [B,1,4096+2·1024]  state input + B + C concatenated
                                                  # dt: [B,1,64]    one per head
xbc = conv1d_step(xbc, conv_state)                # causal Conv1d with kernel=4
                                                  # conv_state ∈ R^{4096+2·1024, 4}, rolling buffer
xbc = silu(xbc)
x, B, C = split(xbc, [4096, 1024, 1024])           # x: [B,1,4096]  fed to SSM
                                                  # B,C: [B,1,1024]  ssm_state coords, group-broadcast
# Reshape into [B, num_heads=64, head_dim=64]:
x = x.reshape(B, 1, 64, 64)
B = B.reshape(B, 1, 8, 128)                        # 8 groups, ssm_state=128
C = C.reshape(B, 1, 8, 128)

# ── the SSD recursion (THE KERNEL) ────────────────────────────────────
# Per-step state update + output reduce; mutates ssm_state in place.
y = mamba2_decode_owned(x, z, dt, B, C, ssm_state, A_log, dt_bias, D)
# y: [B, 1, 64, 64]
# ──────────────────────────────────────────────────────────────────────

y = mamba_rms_norm_gated(y, z)                    # fused RMSNorm + gate (group_size=head_dim=64)
out = out_proj(y.reshape(B, 1, 4096))             # [B, 1, 2688]
```

The pieces that already exist as ttnn ops:
- `in_proj`, `out_proj`: plain matmuls.
- `silu`: `ttnn.silu`.
- `split`, `reshape`: ttnn views.

The pieces that **don't** exist on Blackhole and we will build:
- `conv1d_step` (causal Conv1d with rolling state cache) — `ttnn.conv1d`
  exists but only for the full-sequence form, not the per-step caching
  form. This is **G0a/G1 territory** but smaller than the SSD kernel.
- `mamba2_decode_owned` — **the kernel**. The 23 Mamba layers × 64
  heads × the per-step math above. This is the bulk of Phase 0 work
  (G0..G4).
- `mamba_rms_norm_gated` — fused RMSNorm-then-gate; can be implemented
  as `ttnn.rms_norm(y) · z` (two ops) for v0, fused at perf pass.

------------------------------------------------------------------------

## 6. Mamba2 vs GatedDeltaNet (our 35B mixer)

Both are recurrent linear-attention-style mixers that we lump under
"non-quadratic alternatives to softmax attention". But the math is
genuinely different.

### 6.1 GatedDeltaNet (35B)

DeltaNet is a **delta-rule online linear-attention** mixer. Its state is
a learned **matrix** `S ∈ R^{K × V}` (K-V outer product); each token
updates `S` via the delta rule:
```
S_t = (I - β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T   # rank-1 update
y_t = q_t · S_t                                       # query the matrix
```
Roughly: the state is a running **soft key→value lookup table**.
Each token edits it by subtracting the projection along its key
direction (`(I - β k k^T) S` "forgets" along k) and adding `β k v^T`
("remembers" the new key→value association). Read it as
"key-conditioned write to associative memory".

35B has `K_DIM=128` and `V_DIM=128`, so S has 128×128 = 16,384 entries
per V-head per layer. 16 K-heads, 32 V-heads, 40 layers → quite a lot
of state.

### 6.2 Mamba2 SSD (Nemotron)

Mamba2's state is a **rank-2 tensor** `h ∈ R^{head_dim × ssm_state}`
per head (64×128 = 8192 entries per head for Nemotron-3). The update
math is **scalar-decay-plus-rank-1-add**:
```
h_t[d, s] = a_t · h_{t-1}[d, s] + (Δ_t · B_t[s]) · x_t[d]
```
There's no key-conditioned forgetting; the decay is uniform across
all `(d, s)` entries (scalar `a_t`). The "selectivity" lives entirely
in *how much to decay* (Δ controls `a_t`) and *what to add* (B, x are
input-dependent).

### 6.3 Side-by-side

| Aspect | GatedDeltaNet (35B) | Mamba2 SSD (Nemotron) |
|---|---|---|
| State shape per head | matrix `S ∈ R^{K × V}` (~16K entries) | matrix `h ∈ R^{head_dim × ssm_state}` (~8K entries) |
| Update math | `(I − β k k^T) S + β k v^T` (delta rule) | `a · h + Δ B x^T` (scalar decay + rank-1 add) |
| Forgetting | **key-conditional** (project out along k direction) | **uniform** (scalar `a` decays all entries equally) |
| Selectivity | β controls update strength; k/v are input-dep | Δ controls decay; B, C, x all input-dep |
| State dtype on chip | fp32 (proven on 35B) | fp32 (config-specified) |
| Owned-kernel name | `qwen36_gdn_decode_owned` (35B) | `nemotron3_mamba2_decode_owned` (to build) |
| Per-token FMAs | ~16K × num_heads | ~8K × 2 × num_heads = ~16K × num_heads |

**The plumbing reuses from 35B's owned-GDN kernel**:
- Recurrent state in/out plumbing pattern (input + state → output, mutates state)
- fp32 accumulator inside the kernel for numerical stability
- Per-head sharding strategy (each Tensix core handles N heads)
- Two-phase warmup + trace capture discipline

**The math does NOT port** — different state update. The Mamba2
kernel author will read the GDN kernel for the architectural pattern,
then write the new tile math.

> Memory cross-reference: `[[35b-dn-h-state-drift-lever]]` showed
> fp32 H state inside a TT trace caused a 30+ minute trace capture
> hang on Blackhole. Nemotron's fp32 ssm_state hits the same risk
> surface — see §7.3 of the architecture brief for the validate-eager-
> before-trace mitigation plan.

------------------------------------------------------------------------

## 7. Why hybrid? (Why Nemotron-3 uses both Mamba2 AND attention)

A pure-Mamba model can't do everything attention can — specifically,
it struggles with **exact long-range retrieval** ("find the token
that mentioned 'Aditya' 1000 tokens ago and recall the next sentence
verbatim"). The selective state remembers *gist* well but loses
*specific token identity* over long context. Attention is great at
exact retrieval because every past token is addressable verbatim
via softmax-weighted average over KV.

Hybrids (Jamba, Nemotron-H, Zamba-2) interleave the two:
- Mamba layers do the **constant-memory long-range summarisation**.
- A few sprinkled attention layers do the **exact retrieval** when
  needed.

Nemotron-3 Nano's pattern `MEMEM*EMEMEM*…` puts 6 attention layers
in 52 (~12%) at indices `[5, 12, 19, 26, 33, 42]`. The Nemotron-H
paper says this ratio is close to optimal for retrieval-heavy tasks:
enough attention to handle needles, not so much that you lose
Mamba's memory benefit.

Three concrete consequences for our bringup:

1. **The 6 attention layers have NO RoPE.** Positional information
   already lives in the Mamba state (the per-token decay sequence
   `∏ a_k` encodes "how long ago" implicitly). Adding RoPE on top
   would be redundant and the modeling source explicitly avoids it.
   We must triple-check we don't accidentally apply RoPE during
   bringup — silent quality degradation, otherwise.

2. **Long-context behaviour is asymmetric.** Mamba layers naturally
   scale O(1) per token in memory. Attention layers scale O(T). With
   only 6 attention layers and small NKV (2 heads), the long-context
   KV budget is tiny: at the model card's 256K context with
   `head_dim=128`, NKV=2: `262144 × 128 × 2 × 2 (K+V) × 2 (bf16) × 6 =
   1.5 GB` total KV across all attention layers. Fits comfortably on
   a 4-chip mesh. Mamba state is a constant 23 layers × 64 heads ×
   64 × 128 × 4 (fp32) ≈ 1.5 MB per slot, also fine.

3. **The CB scheduler's state model gets a new component.** We
   already have per-slot KV cache (27B/35B/Gemma 4) and per-slot DN
   state (35B). For Nemotron-3 we additionally need **per-slot
   Mamba2 state** (conv state + ssm state per Mamba layer). Plumbing
   pattern reuses from 35B's per-slot DN state.

------------------------------------------------------------------------

## 8. What we will implement on Blackhole (the practical bottom line)

Phase 0 of the
[Nemotron bringup plan](../research/nemotron3_nano_30b_a3b_bringup_plan.md)
calls for an owned Mamba2 SSD decode kernel. After reading this primer
you should be able to read the kernel-stage gates and understand
what's being measured:

| Stage | What it does | Maps to math above |
|---|---|---|
| **G0** | Numpy oracle: pure-numpy single-token SSD step. | §3 pseudocode, implemented in numpy. |
| **G0a** | Isolation harness: random inputs → run oracle, return expected outputs. | Same math; just packages it as a test harness for G1+. |
| **G1** | Single-core kernel (B=1, single head). | §3 pseudocode, written in TT-LLK / circular buffers for one Tensix core. |
| **G2** | Multi-core sharding (64 heads across cores). | Same math per head; just distribute the head loop across cores. |
| **G3** | Batched (B=1..32 leading dim). | Same math; outer batch loop. |
| **G4** | Python wrapper at `ttnn.experimental.nemotron3_mamba2_decode_owned`. | API surface for the server. |

The hard part is G1 — getting the per-head SSD math right in tile
arithmetic with the right fp32 accumulator placement. Once G1 passes
its cosine gate vs the numpy oracle, G2..G4 follow the same pattern
35B's owned-GDN kernel used (`feedback_owned_decay_gate_shipped`).

------------------------------------------------------------------------

## 9. Reading list (in order, optional)

If you want to go deeper than this primer:

1. **Tri Dao's Mamba2 blog series** —
   [Part 1: the model](https://tridao.me/blog/2024/mamba2-part1-model/),
   Part 2: theory, Part 3: algorithm, Part 4: systems. The model
   author's own pedagogy. Part 1 is enough for our purposes.
2. **Mamba2 paper** — [Dao & Gu 2024, "Transformers are SSMs"](https://arxiv.org/abs/2405.21060).
   The duality result is theorems 4.1-4.5; the SSD algorithm is
   section 6.
3. **Nemotron-H paper** — [Bercovich et al. 2024, arXiv 2504.03624](https://arxiv.org/abs/2504.03624).
   The hybrid pattern + ablations on attention-vs-mamba ratio.
4. **state-spaces/mamba** — [reference implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py).
   The actual code that ships in `mamba-ssm`. Read
   `Mamba2Mixer.forward` and `mamba_chunk_scan_combined`.
5. **Mamba (S6) paper** — [Gu & Dao 2023, arXiv 2312.00752](https://arxiv.org/abs/2312.00752).
   The selective-state-space paper; reads more like "how we fixed
   what S4 couldn't do" and is good background.
6. **S4 paper** — [Gu et al. 2021, arXiv 2111.00396](https://arxiv.org/abs/2111.00396).
   Where the SSM lineage actually starts as a usable deep-learning
   building block; the HiPPO connection is in this paper. Worth
   skimming if you want the full ancestral chain.

------------------------------------------------------------------------

## 10. The "Q&A" — common confusions

These are questions I had during research; including them so you
don't have to re-derive.

**Q: Is the SSM state like a KV cache?**
Sort of. Both persist information across tokens. But: KV cache stores
*verbatim* per-token K/V vectors (O(T) memory growth, exact recall);
SSM state stores a *summary* of the past in a fixed `[head_dim, ssm_state]`
matrix per head (O(1) memory, soft recall). They are complementary.

**Q: Why does the kernel mutate `ssm_state` in-place?**
Two reasons. (a) The state is **persisted across decode steps** —
next token's update reads the current state, so it must be written
back. (b) fp32 in-place mutation avoids allocating a new fp32 tensor
per step (would dominate memory traffic).

**Q: Why fp32 for `ssm_state` specifically?**
The recurrence `h_t = a · h_{t-1} + …` accumulates over potentially
thousands of tokens. bf16's 8-bit mantissa loses precision in the
multiply step; over 1000 multiplies the drift compounds enough that
output quality degrades visibly. Verified on 35B's DN H state (also
fp32 by analogous argument); same risk here. The Nemotron-3 config
explicitly states `mamba_ssm_cache_dtype="float32"`.

**Q: What's the connection to linear attention?**
SSD's matrix form (§2.5) is a **lower-triangular causal matrix**
that, with all `a_k = 1`, reduces to `C · B^T` — that's literally
linear attention (kernel-form softmax-free attention). Mamba2 is
linear attention *with input-dependent decay multipliers* along the
diagonal. The decay multipliers are the selectivity mechanism.

**Q: Why doesn't the Mamba layer need positional encoding?**
The cumulative product `∏_{k=j+1}^{i} a_k` in the matrix form (§2.5)
is itself a positional signal — it encodes "how many tokens between
j and i" weighted by the model's learned per-token decay. So
"position" is implicit in the decay pattern. No explicit RoPE needed.
The 6 attention layers piggyback on this — they see queries/keys
that already contain positional information indirectly through the
Mamba states preceding them.

**Q: How do you parallelise the recurrence at training time?**
For training/prefill where the full sequence is known, use the
**SSD chunked scan**: split the sequence into chunks of size ~64-256.
Within each chunk, use the matrix form (matmul-friendly). Between
chunks, thread the SSM state forward (a tiny recurrence over chunks
instead of tokens). Result: O(T·N²) FLOPs but expressed as matmuls.
For decode (one token at a time, which is what our kernel handles),
just use the plain recurrence form.

------------------------------------------------------------------------

## What to read next

- The
  [Nemotron-3 architecture brief](../research/nemotron3_nano_architecture_brief.md)
  — §4.3 has the exact per-step decode pseudocode mapped to
  Nemotron-3's specific shapes.
- The
  [Nemotron bringup plan](../research/nemotron3_nano_30b_a3b_bringup_plan.md)
  §3a — Phase 0 G0..G4 stage gates for the owned kernel.
- The
  [35B GDN kernel build](../experiments/cb/isolate/owned_gdn.py) +
  memory entry `feedback_owned_decay_gate_shipped` — for the
  architectural pattern we'll reuse (not the math).
