# Owned Conv1d Decode Kernel — Bring-Up Plan (2026-05-18 evening)

Designs the next custom TT-Metal op after the owned GDN kernel shipped today
(commit `26cad39`). Target: collapse the 4-tap depthwise conv + state shift
in `server_tp.py:570-580` into a single owned op
`qwen36_conv1d_decode_owned`.

This plan must be read end-to-end before any code lands. Same workflow
shape as the GDN bring-up (`research/owned_gdn_diagnosis_2026_05_18.md`),
which is the precedent for "what worked": staged correctness gates,
ULP-aware acceptance, no default-flip without long-context evidence.

## Why conv1d (post-owned-gdn profile)

Per `research/post_owned_gdn_profile_2026_05_18.md` (commit `1bcbcbb`),
DeltaNet_conv is the #4 category at **83 ms eager-proxy / 11.30%** with
288 ops/token (6 ops × 48 DeltaNet layers). Prior diagnosis
(`feedback_conv1d_diagnosis.md`) attributed:

| sub-op | mean ms | % of layer |
|---|---:|---:|
| `ttnn.sum` (small-K reduce) | 0.674 | 65.5% |
| concat + slice (state mgmt) | 0.218 | 21.1% |
| `ttnn.mul` (elementwise) | 0.133 | 13.0% |
| `ttnn.silu` | 0.005 | 0.5% |

Projected savings (per `feedback_conv1d_diagnosis.md`): a fused
mul-sum-silu + state-shift custom kernel reaches the bandwidth floor of
~6.5 ms/tok across 48 layers, saving **~30 ms/tok eager-proxy**.

## The math contract — what the kernel must implement

Current eager body (`server_tp.py:570-580`):

```python
mixed_col       = ttnn.reshape(mixed_qkv, [CONV_DIM_CHIP, 1])    # [D, 1]
conv_input      = ttnn.concat([dn['conv_st'], mixed_col], dim=-1) # [D, K=4]
conv_prod       = ttnn.mul(conv_input, dn['w_conv'])              # [D, 4]
conv_out        = ttnn.silu(ttnn.sum(conv_prod, dim=-1))          # [D]
conv_state_new  = ttnn.slice(conv_input, [0, 1], [D, K])          # [D, K-1=3]
```

Where:
- `D = CONV_DIM_CHIP = 2 * KEY_DIM_CHIP + VAL_DIM_CHIP` (10240 / 4 = **2560** per chip on qb2 TP4)
- `K = cfg['conv_kernel'] = 4`
- `mixed_qkv` is `[D]` bf16 (current step's conv input)
- `dn['conv_st']` is `[D, K-1=3]` bf16 (last 3 timesteps' values)
- `dn['w_conv']` is `[D, K=4]` bf16 (4-tap depthwise weights, persistent)
- `conv_out` is `[D]` bf16 (silu of the 4-tap dot product)
- `conv_state_new` is `[D, K-1=3]` bf16 (replaces `conv_st` for next step)

The kernel `qwen36_conv1d_decode_owned(mixed, conv_st, w_conv) →
(conv_out, conv_st_next)` must:

1. For each element of `D`, compute `out[d] = silu(conv_st[d,0]*w[d,0] +
   conv_st[d,1]*w[d,1] + conv_st[d,2]*w[d,2] + mixed[d]*w[d,3])`.
2. Compute next state as a one-position shift:
   `conv_st_next[d,0] = conv_st[d,1]`,
   `conv_st_next[d,1] = conv_st[d,2]`,
   `conv_st_next[d,2] = mixed[d]`.

It must mutate `conv_st` in place (analog to the owned GDN op that mutates
`state` via the writer kernel), so the resident DeltaNet `dn['conv_st']`
buffer is updated for the next forward without a separate `ttnn.copy`.

## Prior-art audit — what NOT to re-attempt

Three approaches the project has already evaluated and rejected:

| approach | why rejected | citation |
|---|---|---|
| `ttnn.conv1d` for 3-tap decode | designed for length>>3, setup overhead exceeds savings | `feedback_conv1d_depthwise_deferred.md` |
| `update_cache_for_token_` ring buffer for state | L1 overflow at `D=10240, K=4` (26 MB > 1.5 MB max) | `feedback_conv1d_circular_buffer.md` |
| `ttnn.copy(src, ttnn.slice(buf,...))` slice-as-destination shift | slice materializes a fresh tensor; write goes nowhere | `feedback_conv1d_circular_buffer.md` |
| `mul_reduce_scalar` C++ primitive via Python | C++-only, no Python binding; building a custom op anyway is the cheaper path | `feedback_conv1d_depthwise_deferred.md` |

**Our approach side-steps all four:** the kernel takes the 4 tap states as
**separate compute-side circular-buffer inputs** (reader kernel feeds them
in the right order from the persistent DRAM buffer + the per-step mixed
input). The state shift happens in the **writer kernel**, not as a Python
ttnn op. This is the same trick the friend's vendored kernel uses (see
below).

## Reference implementation — friend's `qwen36_causal_conv_decode`

Friend's vendored op at
`experiments/.refs/tt-qwen-36/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_causal_conv_decode/`
is the direct template. Compute kernel
(`device/kernels/compute/qwen36_causal_conv_decode.cpp`):

```cpp
// Per tile, explicit 4-tap unroll (no ttnn.sum / no small-K reduction)
binary_op_init_common(cb_state1, cb_weight0, cb_acc0);
for (uint32_t tile = 0; tile < tile_count; ++tile) {
    mul_front(cb_state1, cb_weight0, cb_acc0);       // acc0 = state1 * w0
    mul_front(cb_state2, cb_weight1, cb_product);
    add_front(cb_acc0,   cb_product,  cb_acc1);      // acc1 = acc0 + state2*w1
    mul_front(cb_state3, cb_weight2, cb_product);
    add_front(cb_acc1,   cb_product,  cb_acc0);      // acc0 = acc1 + state3*w2
    mul_front(cb_mixed,  cb_weight3, cb_product);
    add_front(cb_acc0,   cb_product,  cb_acc1);      // acc1 = acc0 + mixed*w3
    silu_front(cb_acc1,  cb_conv_out);                // conv_out = silu(acc1)
}
```

Key design choices we will mirror:
1. **4 separate state CBs** — `state1, state2, state3` are read in the
   tap order the reader kernel feeds them; the kernel doesn't care which
   memory slot they live in.
2. **Explicit unroll** — no `ttnn.sum`. Four `mul` + three `add` + one
   `silu`. Eliminates the 65% sum-reduce cost the diagnosis flagged.
3. **State shift in the writer kernel, not the compute kernel** —
   sidesteps the in-place L1 update problem entirely. The writer reads
   `state2/state3/mixed` from input CBs and writes them to slots 0/1/2
   of the persistent state buffer.
4. **No CB type beyond bf16** — same as owned GDN; intermediate accumulation
   stays in fp32 dst register inside each `mul_tiles`/`add_tiles` call.

Note: the friend repo is known to have GDN-recurrence errors. The
conv1d kernel is structurally simpler (no contraction, no state-update
math beyond shift), so the risk of porting math errors is low — but we
will still validate against an independent oracle.

## Hardware mapping

| quantity | value |
|---|---|
| `D` (CONV_DIM_CHIP at qb2 TP4) | 2560 |
| `K` (conv kernel size) | 4 |
| tile rows along D | 2560 / 32 = **80 tiles** |
| tile cols (state width) | 1 tile per row (state lives in `[D]` columns) |
| inputs needed in compute kernel | mixed (1), state1/2/3 (3), weights 0..3 (4), product (1), acc0/acc1 (2), conv_out (1) = **12 CBs** |
| L1 footprint per work block | 12 tiles × 2 KB (bf16 tile) = **24 KB** — fits comfortably in 1.5 MB/core L1 |
| compute kernel config | `MathFidelity::HiFi4`, `fp32_dest_acc_en = true`, all bf16 CBs (same as owned GDN) |

**Work decomposition.** 80 tile-rows across the 8×8 Tensix grid: 80 / 64 cores
= ~1.25 tiles/core, so use a 10×8 = 80-core slice or similar. (Final layout
chosen during program-factory implementation; not gate-relevant.)

## Staged validation gates (Tier-G structure, analogous to GDN bring-up)

No phase advances without passing the prior phase's gate. Each artifact
goes into `.cache/qb2_tp_deltanet/` (or a sibling) and a result note goes
into `research/`.

### G0 — Standalone single-device synthetic correctness
- New op tree at `experiments/owned_ops/qwen36_conv1d_decode_owned/`
  (mirrors `qwen36_gdn_decode_owned/`: device/{kernels/{compute,dataflow},
  device_operation,program_factory}, nanobind, hpp/cpp, integrate_into_
  ttmetal.py, README, INTEGRATION.md, test_qwen36_conv1d_decode_owned.py).
- BF16-native ladder on synthetic random tensors: shape sweep
  `D∈{32,128,2560} × K=4`. Compare against CPU numpy oracle
  (`mixed @ w + state.flip(-1) @ w[:,:3]` equivalent).
- Gate: PCC ≥ 0.99999 at every shape, max_abs_diff ≤ 0.0005 in BF16-native
  mode. Plus state-shift exactness check (`state_next[d,0]` equals
  input `state[d,1]` etc.) at byte level.
- Cost: ~3-4 days (kernel + program factory + nanobind + tt-metal rebuild).

### G1 — Real-tensor probe via resident server endpoint
- New endpoint `handle_probe_deltanet_owned_conv1d_real_tensors_tp` that
  pulls live `mixed_qkv` / `conv_st` / `w_conv` from the production
  forward at layer 0 and runs both the manual eager body and the owned
  op; compares.
- Gate: output PCC ≥ 0.999999, conv_st_next match the manual shift
  exactly (it's a bf16 copy, should be byte-exact). Acceptable: zero
  argmax-impacting elements above 1e-3.
- Run across all 48 DeltaNet layers (single forward sweep), confirm
  every layer accepts.

### G2 — Guarded trace probe (1 prompt, 20 tokens)
- New `state.deltanet_conv1d_mode` flag in `MeshServerState`. Default
  `"manual"`. Setting to `"owned_conv1d"` makes `deltanet_step_tp` route
  through the new op.
- Endpoint `handle_probe_deltanet_owned_conv1d_trace_tp` captures a guard
  trace with the new mode active, runs 20-token decode, compares
  generated IDs to manual.
- Gate: 20/20 token identity; same-session full decode latency at least
  comparable (no regression > 1 ms/tok).

### G3 — cosine_ladder_tp at 500 positions (the qb1 long-context bar)
- Re-use the existing `cosine_ladder_tp` endpoint (commit `088c33b`),
  add an extra arg `--deltanet-conv1d-mode` to toggle the new path.
- Run base mode `owned_gdn` / conv1d `manual` vs base mode `owned_gdn` /
  conv1d `owned_conv1d` at MAX_POS=512, max_tokens=500, JSON parser prompt.
  (owned_gdn stays on for both — we're testing the conv1d change in
  isolation against the current production decode path.)
- Gate: 10/500 disagreement rate or better (matches Tier 3 GDN
  result), median cosine ≥ 0.999, NO cliff in rolling 50-step buckets.

### G4 — Promotion (default flip)
- Edit `MeshServerState.__init__` to set
  `self.deltanet_conv1d_mode = "owned_conv1d"`.
- Re-bootstrap qb2 (cold) so the trace re-captures with owned_conv1d.
- Verify-after-flip: `generate_tp` on canonical prompts. Expected per-tok
  decode time delta: somewhere between the diagnosis projection
  (~30 ms eager savings → maybe ~3-4 ms trace savings, comparable to or
  bigger than the owned_gdn 2.6 ms delta) and "no measurable change"
  (eager projections compress at unpredictable rates in trace).
- Commit + update HANDOFF + ACTIVE_CONTEXT exactly as we did for owned_gdn.

## Open design questions to resolve during G0

1. **In-place state mutation vs returned state.** The owned GDN op returns
   `(state_next, out)` and has both `"owned_gdn"` (copy-back to `dn['ssm']`)
   and `"owned_gdn_inplace"` (state input is aliased) modes. Conv1d should
   follow the same dual-mode pattern; default to the safer copy-back
   mode first.

2. **Weight pre-split.** `dn['w_conv']` is currently `[D, 4]` and would
   be sliced into 4 weight CBs inside the reader kernel. Alternative:
   pre-split at upload time into 4 separate `[D]` weight tensors per
   layer to avoid the runtime slice. The friend kernel takes 4 separate
   weight CBs already (`cb_weight0..3`).

3. **Whether the state shift happens in writer or compute.** Friend uses
   the writer. We'll do the same; the writer reads from `cb_state2`,
   `cb_state3`, `cb_mixed` (the same CBs the compute kernel reads) and
   writes them to slots 0/1/2 of the persistent state buffer. No
   additional reads required.

4. **Whether to fuse upstream the `ttnn.reshape(mixed_qkv, [D, 1])`** into
   the kernel. The reshape is logical; the kernel can take `[D]` directly.
   Likely a perf no-op but reduces an op count.

## What we will NOT do in this bring-up

- Will not optimize beyond the friend-kernel-equivalent design until G3
  passes. No CB-count micro-tuning, no pretransposed weights, no
  packer-acc tricks until correctness is locked.
- Will not refactor `dn['w_conv']` storage on disk; pre-split happens in
  the upload path at bootstrap, not in the safetensor format.
- Will not introduce a separate eager-only conv1d_mode toggle until G2;
  one mode flag + default suffices.

## Estimated effort

| phase | wall time |
|---|---|
| G0 (build + standalone) | 3-4 days (mostly kernel + tt-metal rebuild iteration) |
| G1 (resident-server real-tensor) | 1 day (probe endpoint + sweep) |
| G2 (guarded trace) | half day (one new endpoint, leverages existing trace infra) |
| G3 (cosine_ladder_tp 500 positions) | half day on qb2 (server restarts dominate) |
| G4 (default flip + verify + docs) | half day |

Total: ~1 week of focused work if no surprises. Same shape as the GDN
bring-up which landed in 4 days of session work.

## Rollback path

`state.deltanet_conv1d_mode = "manual"` reverts to the current production
eager body. The custom kernel never replaces the manual code path; it's
gated by the mode flag, exactly like owned_gdn today.
