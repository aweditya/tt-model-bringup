# Owned DeltaNet decay/gate kernel — Bring-Up Plan (2026-05-18 evening)

Designs the next custom TT-Metal op after the owned conv1d kernel landed
its G0 scaffold today (commit `a6ebdc4`). Target: collapse the 10-op
`add → softplus → exp → neg → mul → exp → sigmoid → reshape × 2` chain
in `server_tp.py:681-690` into a single owned op
`qwen36_decay_gate_decode_owned`.

This is the **biggest remaining cluster** in the post-owned-gdn profile
(`research/post_owned_gdn_profile_2026_05_18.md`): DeltaNet_decay_gate at
**480 ops / 99.32 ms / 13.52%** of eager-proxy time. Same staged
validation gates as the GDN/conv1d bring-ups.

## Why decay_gate (post-owned-gdn profile)

Per `research/post_owned_gdn_profile_2026_05_18.md`, decay_gate is the #1
category by eager-proxy time. The cost is **dispatch-dominated**, not
arithmetic: 10 ttnn ops × 48 DeltaNet layers = 480 dispatches per token,
each operating on a tiny [1, NV_PER_CHIP=12]-shape tensor (one tile). A
single fused op collapses 480 dispatches → 48, with the actual eltwise
math contributing < 1% of the eager-proxy cost.

Projected trace-mode savings: applying the same ~50:1 eager→trace
compression observed for owned_gdn (100 ms eager → 2 ms trace) suggests
~74 ms eager savings → ~1.5 ms trace savings. Smaller than the GDN
ship's 2.6 ms delta, but real and additive.

## The math contract — what the kernel must implement

Current eager body (`server_tp.py:681-690`):

```python
a_biased  = ttnn.add(a_tt, dn['dt_bias'])
softplus_a = (ttnn.softplus(a_biased)
              if state.deltanet_decay_mode == "native_softplus"
              else ttnn.log(ttnn.add(ttnn.exp(a_biased), 1.0)))
g         = ttnn.mul(ttnn.neg(ttnn.exp(dn['A_log'])), softplus_a)
beta      = ttnn.sigmoid(b_tt)
decay     = ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1])
```

Where (per chip on qb2 TP4):
- `a_tt`, `b_tt`: `[1, NV_PER_CHIP=12]` bf16 TILE_LAYOUT (1 input tile each,
  real data in row 0 cols 0..11)
- `dn['dt_bias']`, `dn['A_log']`: `[NV_PER_CHIP=12]` bf16 (per-layer
  weights, persistent)
- `decay`: `[1, NV_PER_CHIP, 1, 1]` bf16 (one scalar per slot, broadcast
  shape for downstream recurrence)
- `beta`: `[1, NV_PER_CHIP]` bf16 (sigmoid output)

The kernel
`qwen36_decay_gate_decode_owned(a, b, dt_bias, A_log) → (decay, beta)`
must:

1. Compute `softplus(a + dt_bias)` per element. **Decision (see prior-art
   audit): use native `softplus_tile` SFPU.** The manual
   `log(exp(x) + 1)` path is what the production code uses by default
   because the earlier `state.deltanet_decay_mode = "native_softplus"`
   probe broke 20-token identity — but inside our kernel we control the
   precise sequence, and the SFPU `softplus_tile` is the same primitive
   the SFPU's `log(exp(x)+1)` chain ultimately calls. Validate against
   the manual chain at G0.
2. Compute `-exp(A_log) * softplus`.
3. Compute `decay = exp(g)` and emit in `[1, NV, 1, 1]` shape (1 tile
   logical per slot, each 32×32 padded with the scalar in [0,0]).
4. Compute `beta = sigmoid(b)` and emit in `[1, NV]` shape (1 tile
   logical, padded `[32, 32]`).

## Prior-art audit — what NOT to re-attempt

| approach | why rejected | citation |
|---|---|---|
| `ttnn.softplus(a_biased)` alone (no other fusion) | passed isolated correctness gate, broke 20-tok token identity in combined-path benchmark | ACTIVE_CONTEXT.md Section "Native DeltaNet softplus is also not promotable" + `feedback_softplus_decay_*.md` |
| Toggle `state.deltanet_decay_mode = "native_softplus"` per call | drift on multi-token generation despite first-token argmax match | same |

**Our approach side-steps both:** by fusing the entire chain
(softplus + neg-exp + mul + exp for decay, sigmoid for beta) in one
kernel, we control the rounding/materialization schedule end-to-end.
The earlier native-softplus failure was likely a mid-chain
bf16-pack-roundtrip difference (manual path: `add → exp → add(+1) → log →
mul`, each packing to bf16 between; native softplus: one SFPU call with
fp32 dst preserved). Inside our owned kernel we keep all intermediate
math in **fp32 dst** across the entire chain (matches the GDN-kernel
correctness pattern shipped today), so the kernel is *strictly more
accurate* than either the manual or native-softplus production paths.

## Hardware mapping

| quantity | value |
|---|---|
| `NV_PER_CHIP` (decay/gate axis on qb2 TP4) | 12 (fits in 1 tile of 32 elements) |
| input shape per tensor | `[1, NV]` logical, `[1, 32]` padded — **1 tile each** |
| output decay shape | `[1, NV, 1, 1]` logical, padded `[1, NV, 32, 32]` — **12 tiles** (one per slot, scalar at [0,0]) |
| output beta shape | `[1, NV]` logical, `[1, 32]` padded — **1 tile** |
| work blocks per call | 1 (the entire computation fits in one tile of math) |
| compute kernel config | `MathFidelity::HiFi4`, `fp32_dest_acc_en = true` (matches owned_gdn/owned_conv1d) |

**Work decomposition.** Trivial — one work block on one core, since the
compute is 1 tile total. The output expansion (1 tile of decay scalars →
12 tiles each with one scalar at [0,0]) happens in the writer kernel.

## Compute kernel structure (per work block, 1 tile)

```cpp
// Stage 1: softplus(a + dt_bias) -> cb_softplus (1 tile)
add_tiles(cb_a, cb_dt_bias, dst=0)            // a + dt_bias
softplus_tile(0)                                // SFPU softplus
pack_tile(0, cb_softplus)

// Stage 2: g = -exp(A_log) * softplus -> cb_g
exp_tile_init(); exp_tile(cb_A_log, ...)        // exp(A_log)
negative_tile_init(); negative_tile(...)        // -exp(A_log)
pack_tile to cb_neg_exp_A_log
mul_tiles(cb_neg_exp_A_log, cb_softplus, dst=0)
pack_tile(0, cb_g)

// Stage 3: decay_compact = exp(g) -> cb_decay_compact (1 tile)
exp_tile_init(); exp_tile(cb_g, ...)
pack_tile to cb_decay_compact

// Stage 4: beta = sigmoid(b) -> cb_beta_out (1 tile)
sigmoid_tile_init(); sigmoid_tile(cb_b, ...)
pack_tile(0, cb_beta_out)
```

The writer kernel then writes:
- `cb_beta_out` → `beta_out_addr` at tile_id 0 (1 tile)
- `cb_decay_compact` → expanded into 12 tile writes to `decay_out_addr`,
  one tile per slot. Each expanded tile is a fresh 32×32 buffer with
  only [0,0] set (from the compact tile's `[0, slot_index]` position).

The expansion is the only non-trivial bit. Two designs:

### Writer expansion design

**Option A** — writer computes 12 tiles, reads the compact tile element by
element and writes 12 separate scalar-padded tiles to DRAM. Per-element
NoC writes; might be slow but simple.

**Option B** — compute kernel produces 12 separate tiles (one per slot)
using SFPU element-pluck + zero-fill. More compute work but the writer
just does straight tile writes.

**Pick: Option B.** The compute kernel work for 12 tiles is still tiny
(~12 SFPU instructions per output tile), and the writer becomes
identical to the conv1d/GDN writers (just `noc_async_write` per output
tile, no element-level math).

Hmm — actually plucking element `i` from a [1, 32] tile and zeroing the
other 31 columns requires SFPU mask ops. A simpler design is to compute
the [1, NV] result, then have the **reader kernel of the next op
(owned_gdn)** consume the compact tile directly. But that requires
changing the owned_gdn op signature.

For G0, do **Option A** (writer-side per-element expansion) as the
minimum-viable starting point. If the writer is too slow, profile and
move to Option B.

### Even simpler approach

For the very first scaffold, emit `decay` in the same `[1, NV]` compact
shape as `beta`, and keep the Python-side `ttnn.reshape([1, NV, 1, 1])`
that the production code already does. The reshape is metadata-only
(no data movement) and cheap. Lose almost nothing.

**Final design**: output both `decay` and `beta` as `[1, NV]` (1 tile
each). Python wrapper reshapes decay → `[1, NV, 1, 1]` before passing to
owned_gdn. Two-output op like owned_gdn.

## Staged validation gates (same shape as GDN/conv1d)

### G0 — Standalone single-device synthetic correctness
- Op tree at `experiments/owned_ops/qwen36_decay_gate_decode_owned/`.
- BF16-native ladder on synthetic random `[1, NV=12]` tensors.
  Reference: numpy oracle computing the exact same chain (using fp32
  `log(exp(x)+1)` for softplus — matches the **production-default**
  manual path, not the native_softplus variant).
- Gate: PCC ≥ 0.99999, max_abs_diff ≤ 0.0005 in BF16-native mode.
  Plus a stricter gate comparing against `ttnn.softplus`-based reference
  to characterize the kernel's accuracy vs both reference paths.
- Cost: ~2 days (kernel is much simpler than GDN/conv1d).

### G1 — Real-tensor probe via resident server endpoint
- `handle_probe_deltanet_owned_decay_gate_real_tensors_tp` that pulls
  live `a_tt`, `b_tt`, `dn['dt_bias']`, `dn['A_log']` from layer 0 and
  validates against the manual chain.
- Gate: output PCC ≥ 0.999999, layer-sweep across all 48 DeltaNet
  layers.

### G2 — Guarded trace probe (1 prompt, 20 tokens)
- `state.deltanet_decay_gate_mode` flag in `MeshServerState`. Default
  `"manual"`. Setting `"owned_decay_gate"` routes through the new op.
- Endpoint `handle_probe_deltanet_owned_decay_gate_trace_tp`.
- Gate: 20/20 token identity vs manual.

### G3 — `cosine_ladder_tp` at 500 positions (the qb1 long-context bar)
- Re-use the existing endpoint with the new `--deltanet-decay-gate-mode`
  arg, run base owned_gdn / decay_gate manual vs owned_gdn /
  decay_gate owned, MAX_POS=512, max_tokens=500, JSON parser prompt.
- Gate: 10/500 disagreement rate or better (matches conv1d/GDN ships),
  median cosine ≥ 0.999, NO cliff.

### G4 — Promotion (default flip)
- Edit `MeshServerState.__init__` to set
  `self.deltanet_decay_gate_mode = "owned_decay_gate"`.
- Cold-bootstrap qb2; verify-after-flip on canonical prompts.
- Expected per-tok delta: ~1.5 ms (per the eager→trace compression
  hypothesis above).
- Commit + update HANDOFF + ACTIVE_CONTEXT.

## Open design questions to resolve during G0

1. **`softplus_tile` SFPU correctness vs manual `log(exp(x)+1)`.** The
   prior production probe found token-identity drift on multi-token
   decode. Inside our owned kernel, both paths can be implemented and
   gated by a compile-time arg; G0 should compare them side-by-side
   against the numpy oracle.

2. **Weight pre-format.** `dn['dt_bias']` and `dn['A_log']` are currently
   `[NV_PER_CHIP]` (rank-1). They need to be `[1, NV]` (rank-2) tile-padded
   as kernel inputs. Pre-format at upload time.

3. **Output expansion timing.** Whether to emit `[1, NV]` (compact) or
   `[1, NV, 1, 1]` (expanded) from the kernel. Start with compact + Python
   reshape; revisit only if the reshape shows up in a fresh profile.

## What we will NOT do in this bring-up

- Will not try to also fuse with the recurrence (decay output feeds
  `H_decayed = H * decay` inside owned_gdn — that's a different kernel).
- Will not attempt to combine decay_gate with QKV repeat into a giant
  "prepare decode" op (the friend's `qwen36_gdn_prepare_decode` reference
  could inform this, but it's a separate fusion target — see C3 in the
  post-owned-gdn profile menu).
- Will not optimize beyond the friend-kernel-equivalent design until G3
  passes.

## Estimated effort

| phase | wall time |
|---|---|
| G0 (build + standalone) | 2 days (kernel is much simpler than GDN/conv1d) |
| G1 (resident-server real-tensor) | half day |
| G2 (guarded trace) | half day |
| G3 (cosine_ladder_tp 500 positions) | half day on qb2 |
| G4 (default flip + verify + docs) | half day |

Total: ~4 days of focused work, vs ~1 week for GDN/conv1d. The simpler
kernel pays off in faster bring-up.

## Rollback path

`state.deltanet_decay_gate_mode = "manual"` reverts to the production
default. Same pattern as owned_gdn (commit `26cad39`) and the planned
owned_conv1d default flip (per `research/conv1d_custom_kernel_plan_2026_05_18.md`).
