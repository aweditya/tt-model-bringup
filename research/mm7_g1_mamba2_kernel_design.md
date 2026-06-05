# MM7 G1 — Mamba2 SSD single-core kernel design

**Owner**: task #186 (in-progress 2026-06-04).
**Fork base**: `experiments/owned_ops/qwen36_gdn_decode_owned/`
(installed at `tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/`).
**Math reference**: `wiki/65_mamba_state_space_models.md` §3 +
`research/nemotron3_nano_architecture_brief.md` §4.3.
**Numpy oracle (already shipped, G0)**: `experiments/utils/mamba2_numpy_oracle.py`.
**Validation gate (already shipped, G0a)**: `experiments/utils/test_mamba2_decode_isolated.py`
with `--kernel-callable nemotron3_mamba2_decode_owned:fn`.

------------------------------------------------------------------------

## 0. Survey result — what already exists on Blackhole

The 35B bringup left us a *deep* shelf of owned ops at
`experiments/owned_ops/`:

| Op | Math | Reuse for Mamba2 |
|---|---|---|
| `qwen36_gdn_decode_owned` | delta-rule recurrence + Q@state output | **fork base** for the SSD kernel (math differs, scaffolding ports) |
| `qwen36_conv1d_decode_owned` | 4-tap depthwise causal Conv1d + SiLU + rolling state shift | **DIRECT REUSE** for Mamba2's `conv1d_step` |
| `qwen36_decay_gate_decode_owned` | fused decay-then-gate | LOW reuse (Mamba2 has its own scalar-decay structure) |
| `qwen36_gdn_prediction` | `pred = k @ state_scaled` sub-op | LOW reuse (Mamba2 doesn't have a separate prediction stage) |
| `qwen36_gdn_output`, `qwen36_gdn_delta`, `qwen36_gdn_outer_update`, `qwen36_gdn_decay_state`, `qwen36_gdn_prepare_decode` | other GDN sub-ops | not applicable to Mamba2 |
| `qwen36_moe_ffn_decode_owned` | fused MoE expert FFN | applies to the 23 MoE layers (Phase 1, not G1) |

**Critical reuse win**: `qwen36_conv1d_decode_owned` already implements
the per-step causal Conv1d + state-shift + SiLU that Mamba2 needs for
`conv1d_step`. Same `kernel_size=4` depthwise structure; just
parametrise `D = mamba_x_dim + 2 * (n_groups * ssm_state) = 4096 + 2*1024 = 6144`.
**Zero new code needed for Mamba2 conv1d_step.** Verify at v0.1.1.

------------------------------------------------------------------------

## 1. File-by-file map: qwen36_gdn_decode_owned → nemotron3_mamba2_decode_owned

Fork the entire directory; rename + rewrite the math. Approximate
LOC counts measured from the GDN kernel:

```
experiments/owned_ops/nemotron3_mamba2_decode_owned/
├── README.md                                              [new]   ~150 LOC
├── INTEGRATION.md                                         [fork]  ~80
├── nemotron3_mamba2_decode_owned.{hpp,cpp}                [fork]  ~50
├── nemotron3_mamba2_decode_owned_nanobind.{hpp,cpp}       [fork]  ~80
├── sources.cmake                                          [fork]  ~10
├── integrate_into_ttmetal.py                              [fork]  ~50
├── test_nemotron3_mamba2_decode_owned.py                  [new]   ~150
├── benchmark_nemotron3_mamba2_decode_owned.py             [fork]  ~80
└── device/
    ├── nemotron3_mamba2_decode_owned_device_operation.{hpp,cpp}   [fork]  ~250
    ├── nemotron3_mamba2_decode_owned_device_operation_types.hpp   [fork]  ~80
    ├── nemotron3_mamba2_decode_owned_program_factory.{hpp,cpp}    [fork]  ~260
    └── kernels/
        ├── compute/nemotron3_mamba2_decode_owned.cpp              [rewrite] ~500
        └── dataflow/
            ├── reader_nemotron3_mamba2_decode_owned.cpp           [rewrite] ~120
            └── writer_nemotron3_mamba2_decode_owned.cpp           [fork]    ~70
```

Total: ~1.9k LOC, of which ~1.0k is straight fork + rename, ~500 is
new compute math, ~270 is new reader (different input tensor list).

------------------------------------------------------------------------

## 2. Op contract (the Python signature G4 must expose)

```python
ttnn.experimental.nemotron3_mamba2_decode_owned(
    x:          Tensor,   # [B, num_heads=64, head_dim=64]              bf16, tile layout
    z:          Tensor,   # [B, num_heads=64, head_dim=64]              bf16, passed through to caller
    dt:         Tensor,   # [B, num_heads=64]                            bf16, padded to tile cols
    B_in:       Tensor,   # [B, n_groups=8, ssm_state=128]               bf16
    C_in:       Tensor,   # [B, n_groups=8, ssm_state=128]               bf16
    ssm_state:  Tensor,   # [B, num_heads=64, head_dim=64, ssm_state=128] fp32  — mutated in place
    A_log:      Tensor,   # [num_heads=64]                                bf16, replicated weight
    dt_bias:    Tensor,   # [num_heads=64]                                bf16
    D:          Tensor,   # [num_heads=64]                                bf16
    *,
    debug_fill: bool = False,
    time_step_floor: float = 1e-4,
    time_step_max: float = 0.1,
    output_memory_config: Optional[MemoryConfig] = None,
    output_tensor: Optional[Tensor] = None,
) -> Tensor:                # y: [B, num_heads=64, head_dim=64]          bf16
```

Returns `y`. **Mutates `ssm_state` in place** (same pattern as
`qwen36_gdn_decode_owned`'s `state` argument). `z` is taken in but
**not consumed inside the kernel** — kept in the signature so the
caller doesn't have to thread it separately to the downstream
`MambaRMSNormGated(y, z)`.

`debug_fill=True` writes a known constant to `y` and skips compute —
first-build sanity for scaffolding before any math lands. Same trick
that worked for `qwen36_conv1d_decode_owned`.

------------------------------------------------------------------------

## 3. SPMD work unit + sharding

GDN sharded across `block = (slot, value_tile)`. Mamba2's natural
sharding is across `block = (batch, head)` because:
- The per-head SSD recursion is **fully independent across heads**
  (the only cross-head coupling is the per-group B/C broadcast,
  which is a read-only fan-out).
- Per (batch, head), the inner loop processes `head_dim × ssm_state =
  64 × 128 = 8192` element ops.
- With 64 heads and Blackhole's ~110 Tensix cores, a single-batch
  decode fits 64 cores with 46 idle. Two batches use 128 cores (over-
  subscribed); we'll size cores in G2.

G1 ships **single-core, B=1, single head** to validate the math.
G2 then expands to all 64 heads / multi-core. G3 expands to B>1.

------------------------------------------------------------------------

## 4. Circular buffers (compute kernel internals)

Forking GDN's 18-CB layout. Many will be reused as-is; some renamed;
some retired (GDN-specific intermediates that Mamba2 doesn't need).
Mamba2-specific additions:

```cpp
// Inputs (reader fills, compute reads)
constexpr uint32_t CB_X      = tt::CBIndex::c_0;   // [head_dim] vector tile
constexpr uint32_t CB_Z      = tt::CBIndex::c_1;   // [head_dim] vector tile (pass-through, may not enter compute)
constexpr uint32_t CB_DT     = tt::CBIndex::c_2;   // scalar tile
constexpr uint32_t CB_B      = tt::CBIndex::c_3;   // [ssm_state] vector tile
constexpr uint32_t CB_C      = tt::CBIndex::c_4;   // [ssm_state] vector tile
constexpr uint32_t CB_A_LOG  = tt::CBIndex::c_5;   // scalar tile
constexpr uint32_t CB_DT_BIAS = tt::CBIndex::c_6;  // scalar tile
constexpr uint32_t CB_D      = tt::CBIndex::c_7;   // scalar tile
constexpr uint32_t CB_STATE_IN  = tt::CBIndex::c_8; // [head_dim, ssm_state] fp32 — many tiles

// Intermediates (compute-only)
constexpr uint32_t CB_DT_EFF = tt::CBIndex::c_9;   // softplus(dt+dt_bias).clamp()
constexpr uint32_t CB_A      = tt::CBIndex::c_10;  // -exp(A_log)
constexpr uint32_t CB_DECAY  = tt::CBIndex::c_11;  // exp(dt_eff * A)
constexpr uint32_t CB_DT_B   = tt::CBIndex::c_12;  // dt_eff * B  (broadcast over ssm_state)
constexpr uint32_t CB_INPUT_CONTRIB = tt::CBIndex::c_13; // dt_eff*B*x outer

// Outputs (compute fills, writer reads)
constexpr uint32_t CB_STATE_OUT = tt::CBIndex::c_14; // [head_dim, ssm_state] fp32
constexpr uint32_t CB_Y         = tt::CBIndex::c_15; // [head_dim] bf16
```

Total ~16 CBs (slightly fewer than GDN's 18). All scalar-broadcast
tiles use full TILE_HEIGHT × TILE_WIDTH = 32×32 with the real value
repeated; that pattern is documented in
`qwen36_conv1d_decode_owned/README.md` ("Layout choice") and reused.

------------------------------------------------------------------------

## 5. LLK ops needed (vs GDN's)

GDN's compute kernel uses: `mul_tiles`, `matmul_tiles`,
`transpose_wh_tile`, `add_tiles`, `mul_scalar_tile`, packing /
unpacking, reconfig_data_format, fill_tile, eltwise unary primitives.

Mamba2 SSD additionally needs:
- **`exp_tile`** (compute decay = exp(dt_eff * A)) — standard unary.
- **`log_tile`** or `log1p` — wait, we DON'T need it; A_log is the
  parameter, A = -exp(A_log), so just exp.
- **`softplus_tile`** — typically composed as `log(1 + exp(x))`. Or
  use the stable form `max(x, 0) + log(1 + exp(-|x|))`. Check LLK
  for a direct `softplus_tile`; if absent, decompose.
- **Tile-wise clamp** — compose as `max(min(x, max_val), floor_val)`.
- **`reduce_tile`** along the ssm_state axis for the C·state^T
  reduction (output reduce). GDN uses `matmul_tiles`; here a single
  matmul tile = `[head_dim, ssm_state] · [ssm_state, 1] = [head_dim, 1]`
  fits in one `matmul_tile_with_dt` call per head_dim chunk.

Decision: try `matmul_tile` for the C-reduce first (cleanest); fall
back to manual `mul_tiles` + sum if precision suffers (the 35B
DeltaNet kernel's matmul-tile precision was fine for `cos ≥ 0.999`,
no reason to expect worse here).

------------------------------------------------------------------------

## 6. Compute kernel pseudocode (the meat)

```cpp
void MAIN {
    // ── runtime args ──
    uint32_t batch_id, head_id, n_state_tiles;
    // (per-block runtime args injected by program factory)

    // ── 1. discretization (one-shot scalar work) ──
    cb_wait_front(CB_DT, 1);
    cb_wait_front(CB_DT_BIAS, 1);
    cb_wait_front(CB_A_LOG, 1);

    // dt_eff = clamp(softplus(dt + dt_bias), floor, max)
    add_tiles(CB_DT, CB_DT_BIAS, 0, 0, CB_TMP_A);
    softplus_tile(CB_TMP_A, 0, CB_TMP_B);
    clamp_tile(CB_TMP_B, 0, time_step_floor, time_step_max, CB_DT_EFF);

    // A = -exp(A_log);  decay = exp(dt_eff * A)
    exp_tile(CB_A_LOG, 0, CB_TMP_A);
    neg_tile_inplace(CB_TMP_A, 0);             // CB_A
    mul_tiles(CB_DT_EFF, CB_TMP_A, 0, 0, CB_TMP_B);
    exp_tile(CB_TMP_B, 0, CB_DECAY);

    // dt_B = dt_eff * B   (broadcast scalar over ssm_state vector tile)
    bcast_mul_tile(CB_DT_EFF, CB_B, 0, 0, CB_DT_B);

    cb_pop_front(CB_DT, 1);
    cb_pop_front(CB_DT_BIAS, 1);
    cb_pop_front(CB_A_LOG, 1);

    // ── 2. state update + output reduce (loop over head_dim chunks) ──
    cb_wait_front(CB_X, 1);
    cb_wait_front(CB_DECAY, 1);
    cb_wait_front(CB_DT_B, 1);

    for (uint32_t d_tile = 0; d_tile < n_dim_tiles; ++d_tile) {
        cb_wait_front(CB_STATE_IN, 1);

        // ssm_state_new = decay * ssm_state + dt_B * x
        // (outer product over d × s; bf16 accumulator OK, fp32 dest)
        mul_tiles(CB_STATE_IN, CB_DECAY, 0, 0, CB_TMP_A);  // decay * state
        bcast_mul_tile(CB_X, CB_DT_B, d_tile, 0, CB_TMP_B); // x[d] * dt_B
        add_tiles(CB_TMP_A, CB_TMP_B, 0, 0, CB_STATE_OUT);

        // y[d] = sum_s(C * state_new) + D * x[d]
        matmul_tile(CB_STATE_OUT, CB_C, 0, 0, CB_TMP_A);   // [d,s] · [s,1] = [d,1]
        bcast_mul_scalar_tile(CB_X, CB_D, d_tile, 0, CB_TMP_B);
        add_tiles(CB_TMP_A, CB_TMP_B, 0, 0, CB_Y);

        cb_pop_front(CB_STATE_IN, 1);
        cb_push_back(CB_STATE_OUT, 1);
        cb_push_back(CB_Y, 1);
    }

    cb_pop_front(CB_X, 1);
    cb_pop_front(CB_DECAY, 1);
    cb_pop_front(CB_DT_B, 1);
}
```

(Pseudocode — the real implementation needs `pack_reconfig_data_format`,
`tile_regs_acquire`/`commit`/`wait`/`release` around every tile op, and
`reconfig_data_format` whenever the input CB format changes. Forking
the GDN compute file gives all that boilerplate.)

------------------------------------------------------------------------

## 7. Validation gates (in order)

1. **Build sanity**: cmake compiles the new directory; nanobind
   registers `ttnn.experimental.nemotron3_mamba2_decode_owned`. Gate:
   `python3 -c "import ttnn; print(ttnn.experimental.nemotron3_mamba2_decode_owned)"`
   prints a callable.

2. **Debug-fill smoke**: call with `debug_fill=True`; verify `y` is
   the known constant. Gate: y is constant, no NaN/Inf, state unchanged
   (since compute is bypassed in debug_fill).

3. **Single-tile correctness**: B=1, single head, fp32 state of zeros,
   random inputs. Call the kernel; compare against the numpy oracle.
   Gate: per-element max|Δ| ≤ 1e-3, per-tile cos ≥ 0.999. Run via the
   G0a isolation harness:
   ```
   ssh qb1 'cd ~/tt-xla && .venv/bin/python \
     experiments/utils/test_mamba2_decode_isolated.py \
     --batch 1 \
     --kernel-callable my_ttnn_wrapper:mamba2_decode_step'
   ```

4. **Multi-step replay correctness**: same harness, n_steps=8. Verifies
   recurrence (not just one isolated step). Gate: per-head cos ≥ 0.999
   at every step.

The kernel is correct when gate 3 + gate 4 both pass.

------------------------------------------------------------------------

## 8. Order of operations (the next ~5 days)

| Day | Deliverable | Gate |
|---|---|---|
| 1 | Fork the GDN directory → `nemotron3_mamba2_decode_owned/`; rename everywhere; build skeleton with `debug_fill=True` writing a constant. | Build + nanobind import + debug_fill smoke |
| 2 | Implement reader (load x/dt/B/C/A_log/dt_bias/D/state into CBs) + writer (write y + state_out). Compute kernel is still debug_fill. | Build + ssm_state correctly mutated by writer pass-through; output is debug-constant |
| 3 | Implement compute kernel: discretization stage (dt_eff, A, decay, dt_B). Compute writes a partial result (just decay * state) so we can verify the decay math. | Per-element diff vs oracle's decay * state portion (with input_contribution forced to 0) ≤ 1e-3 |
| 4 | Implement state update + output reduce. Full math. | Gate 3 PASS (single-tile single-step cos ≥ 0.999) |
| 5 | Multi-step harness; bug-fix; document. | Gate 4 PASS (multi-step replay cos ≥ 0.999) + memory entry written |

If a stage stalls >2 days, fall back to **manual ttnn composite via
existing primitives** (path-A bridge) so Phase 1 isn't blocked. That
would slow Phase 1 step time by ~10× per Mamba layer but unblocks
ladder progress.

------------------------------------------------------------------------

## 8.5 Structural finding (added 2026-06-05 post research)

Day-4 implementation surfaced a load-bearing math-structural difference
between GDN and Mamba2 SSD that the original §6 pseudocode missed:

- GDN's delta-rule math requires `pred = k · s_prev` BEFORE the state
  update. The kernel runs this as `matmul_reduce(k, s_scaled, ...)` at
  `qwen36_gdn_decode_owned.cpp:491`, BEFORE the transpose+matmul_outer
  inner loop. That isolated real matmul implicitly initializes the LLK
  matmul-unpacker state, so the subsequent loop's `mm_init` calls work
  cleanly.
- Mamba2 SSD is linear (no `pred` term), so its first matmul lands
  INSIDE the transpose+matmul loop. On Blackhole this hangs the TRISC
  pipeline starting at the second iteration — `transpose_wh_init_short`
  leaves a sticky unpacker bit (tt-metal #15930, no `transpose_wh_uninit`)
  that the in-loop `mm_init` doesn't clear sufficiently.

**Two fixes**, both informed by research:
- **(A) Restructure: pull C·state reduce earlier.** Compute
  `y_partial = C · state_in^T` as a `matmul_reduce` BEFORE the
  state-update loop. Structurally matches GDN's pattern, drops the
  bare-mm_init workaround, makes the kernel's pipeline shape
  comprehensible. **Recommended.**
- **(B) Drop `transpose_wh_tile` entirely.** Use matmul's `transpose=1`
  B-operand flag (per SDPA's pattern) to fold the col-vector transform
  into the matmul itself. The sticky-bit class of hang disappears.
  Less code, less init churn.

Memory entries: [[feedback-gdn-vs-mamba2-kernel-delta]],
[[feedback-sdpa-transpose-b-flag-escape-hatch]],
[[feedback-mm-init-prime-required]] (the original workaround).

------------------------------------------------------------------------

## 9. Open questions to resolve during implementation

1. Does `softplus_tile` exist as an LLK primitive, or do we need to
   compose? (Check via `ssh qb1 'grep -r "softplus" /home/aditya/tenstorrent/tt-metal/tt_metal/llk_api/'`).
2. `bcast_mul_tile` vs `bcast_mul_scalar_tile` — what's the LLK call
   for "scalar tile × vector tile, broadcast scalar across all
   positions"? GDN uses both flavours; reuse the same idiom.
3. fp32 accumulator placement — should `CB_STATE_OUT` be fp32 native,
   or do we keep bf16 + use `tile_regs` for the fp32 accumulator?
   GDN uses bf16 CBs throughout with fp32_dest_acc; same idiom should
   work here.
4. Per-tile shape for `ssm_state`: head_dim=64 fits 2 TILE_WIDTH
   (32). ssm_state=128 fits 4 TILE_WIDTH. So `[head_dim, ssm_state]`
   = 2×4 = 8 tiles per (batch, head). Loop bounds in the compute
   kernel set by these.

------------------------------------------------------------------------

## 10. Memory entries to write during implementation (predicted)

- `feedback_mamba2_softplus_tile_decomposition` — what LLK ops compose
  softplus stably on Blackhole
- `feedback_mamba2_kernel_fp32_acc_placement` — fp32_dest_acc usage in
  the SSD compute kernel
- `feedback_mamba2_decay_state_writeback` — gotchas writing fp32 state
  back to L1 from the writer

------------------------------------------------------------------------

## Related

- Architecture brief §4.3: `research/nemotron3_nano_architecture_brief.md`
- Plan §3a: `research/nemotron3_nano_30b_a3b_bringup_plan.md`
- Mamba primer §3 (math): `wiki/65_mamba_state_space_models.md`
- Numpy oracle: `experiments/utils/mamba2_numpy_oracle.py`
- Isolation harness: `experiments/utils/test_mamba2_decode_isolated.py`
- Fork base: `experiments/owned_ops/qwen36_gdn_decode_owned/`
- Conv1d step (reuse, no new kernel): `experiments/owned_ops/qwen36_conv1d_decode_owned/`
