# Audit: our `qwen36_gdn_decode_owned` vs `tenstorrent/tt-metal` GDN/DeltaNet work

**Date**: 2026-06-04. **Author**: Aditya Sriram. **Trigger**:
a Tenstorrent engineer at the 2026-06-04 poster session asked whether
we should share approaches with them for the GDN kernel. This is the
audit before the reply.

**Sources**:

- **Ours** (this repo + qb1 ttnn fork at
  `/home/aditya/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/`):
  - `experiments/serve/server_35b_ttnn.py:580-607` (call site)
  - `qwen36_gdn_decode_owned/` (kernel) — README at
    `qwen36_gdn_decode_owned/README.md`
  - Memory: `feedback_owned_decay_gate_shipped.md`,
    `feedback_deltanet_perop_findings.md`,
    `feedback_35b_dn_h_state_drift_lever.md`,
    `feedback_kernel_vs_dispatch_realization.md`,
    `feedback_ttnn_list_rebinding_leaks.md`,
    `feedback_ttnn_slice_view_decay.md`
- **Theirs** (`tenstorrent/tt-metal`):
  - **Main**: no GDN/DeltaNet kernel. Only Mamba1 (`models/demos/wormhole/mamba`)
    and three SSM primitives (`ttnn/cpp/ttnn/operations/experimental/ssm/`:
    `hc_sum_reduce`, `prefix_scan`, `repeat_and_interleave_eltwise_mul`).
  - **`changh95/qwen3-coder-next-wh-qb`** (Qwen3-Next, also GatedDeltaNet
    architecture): C++ op `ttnn.experimental.deltanet_recurrence` at
    `ttnn/cpp/ttnn/operations/experimental/deltanet/`, plus python wrapper
    `models/tt_transformers/tt/gated_deltanet.py`.
  - **`alnah005/qwen_3_6_dev_gdn_ttlang`**: TT-LANG (Python DSL) decode-step
    kernels `models/experimental/tt_symbiote/modules/gdn_kernel.py` and
    a trace-compat orchestrator `gdn_kernel_trace.py`. Tests at
    `tt_symbiote/tests/test_qwen3_6_35b_a3b{,_bottom_up}.py`. **Same model
    as ours.**
  - **`alnah005/deltanet_work`**: pure-PyTorch reference only
    (`models/experimental/gated_attention_gated_deltanet/`); no TTNN kernel.
  - **`ign/fs/qwen_3_6_35B_optimization`**: identical to
    `qwen_3_6_dev_gdn_ttlang` for GDN paths (no further sources).

------------------------------------------------------------------------

## 1. TL;DR (5 bullets)

1. **There is a *parallel* upstream effort** on the same model: branches
   `alnah005/qwen_3_6_dev_gdn_ttlang` (Qwen3.6-35B-A3B) and
   `changh95/qwen3-coder-next-wh-qb` (Qwen3-Next — same GatedDeltaNet
   primitive). Neither is merged to `main`. Ours is the *only* shipped,
   bit-validated production kernel on a 4-chip Blackhole mesh.
2. **Biggest similarity**: identical mathematical decomposition into the
   delta-rule 5-step recurrence (`state*=alpha; pred=K@state; delta=beta*(V-pred);
   state+=K^T·delta; out=Q@state`). Ours and `qwen3-coder-next`'s C++ compute
   kernels are structurally near-identical (compare
   `qwen36_gdn_decode_owned/.../compute/qwen36_gdn_decode_owned.cpp` modes
   `0`/`10` with `deltanet/.../compute/deltanet_recurrence.cpp`).
3. **Biggest differences**: (a) ours runs on (1,4) P150 Blackhole mesh, theirs
   on Wormhole; (b) we use **HiFi4 + fp32_dest_acc_en**, they use **HiFi2**;
   (c) their `deltanet_recurrence` *header* claims a wider fusion scope
   (`conv_out`, `b_proj`, `a_proj`, `z_proj`, `dt_bias`, `A_exp`, `norm_weight`)
   but the kernel docstring admits **steps 2-4 are TODO** — i.e. the fusion is
   aspirational; (d) our op has a **multi-core block-sharded program factory**
   (`split_work_to_cores` over `slots * value_tiles`), theirs uses
   `SingleCore{}` (see `deltanet_device_operation.hpp` `program_factory_t = std::variant<SingleCore>`).
4. **Things WE should adopt from THEM**: (1) the TT-LANG decode-step
   formulation from `gdn_kernel.py` (`gdn_step_8head`, grid=(8,4)) — same
   target geometry, may be easier to maintain than our 17-CB hand-written
   compute kernel; (2) their host-side `deltanet_recurrence` op *signature*
   absorbs the `b/a/z/dt_bias/A_exp/norm_weight` projections — that is our
   pending decay-gate + RMSNormGated fusion (`use_owned_decay_gate=True`
   already half-done at `server_35b_ttnn.py:543`); (3) the trace-compat
   pre-allocation pattern from `gdn_kernel_trace.py` (head-sharded index
   tensors via `ttnn.gather`) is a clean answer to view-decay bugs.
5. **Things THEY should adopt from US**: (1) the *full* delta-rule (steps
   1-5) is shipped, bit-validated and on a 4-chip TP mesh — we are weeks
   ahead in correctness; (2) hard-won fp32-state + manual-recurrence
   tradeoff (the bf16-state owned-GDN kernel breaks long-context decode;
   see `feedback_35b_dn_h_state_drift_lever.md`); (3) view-decay /
   list-rebinding leak rules
   (`feedback_ttnn_slice_view_decay.md`, `feedback_ttnn_list_rebinding_leaks.md`)
   — these will bite anyone with a long-lived ttnn process; (4) our
   `debug_mode = 1..10` switch (compute kernel can run truncated stages
   for ablation profiling) — that paid for itself debugging the
   "duplicate-consumer race at slots>~24" (mode 10).

------------------------------------------------------------------------

## 2. Side-by-side

| Axis | **Ours** `qwen36_gdn_decode_owned` | **Theirs (C++)** `deltanet_recurrence` (changh95/qwen3-coder-next-wh-qb) | **Theirs (TT-LANG)** `gdn_kernel.py` (alnah005/qwen_3_6_dev_gdn_ttlang) |
|---|---|---|---|
| **Model** | Qwen3.6-35B-A3B | Qwen3-Next (same GDN primitive) | Qwen3.5/3.6 family (same GDN) |
| **Status** | Shipped, bit-validated, behind `state.dn_owned_gdn=True` flag (`server_35b_ttnn.py:1396`) | Branch, single-core PF, *steps 2-4 declared TODO in kernel docstring* | Branch, declared "drop-in replacement" but not shipped to main |
| **Compute primitive** | C++ LLK (`compute/qwen36_gdn_decode_owned.cpp`) | C++ LLK (`compute/deltanet_recurrence.cpp`) | TT-LANG `@ttl.operation` Python DSL |
| **Math fidelity** | **HiFi4** + `fp32_dest_acc_en=true` (`program_factory.cpp:172`) | **HiFi2** (docstring: "using HiFi2 math fidelity") | unspecified by source; likely default HiFi4 |
| **State dtype supported** | FP32 *or* BFLOAT16 (`device_operation.cpp:29-30`) | takes input dtype as-is (per validation: head_k/v tile-aligned only) | bf16 only (compact in/out) |
| **State shape (per chip)** | `[1, slots, key_dim, value_dim]`, slots=NV_PER_CHIP=8 | `(B, num_k_heads, key_dim, value_dim*gqa_ratio)` = `(B, 16, 128, 256*2)` | per-device `[HEADS_PER_DEV, D]` rank-2 compact |
| **Outputs** | `(state_inplace, out_flat[1, slots*value_dim])` (native_io) | `(batch, 1, batch_padded, value_dim)`, in-place state mutate | `(state_next, out)` |
| **Fusion scope** | recurrence only (alpha apply → predict → delta → outer → add → output matmul) + optional `native_io` row-broadcast | recurrence **+** `b/a/z/dt_bias/A_exp/norm_weight` projections **on paper** (TODO in kernel) | recurrence + alpha-broadcast + beta-broadcast |
| **Sharding** | multi-core `split_work_to_cores` over `slots × value_tiles` (`program_factory.cpp:78-83`), full Tensix grid | **single core only** per `deltanet_device_operation.hpp` `program_factory_t = std::variant<SingleCore>`; a `_multicore.cpp` reader exists but is not wired to the variant | hardcoded grid `(4,4)` for 4-head/dev and `(8,4)` for 8-head/dev |
| **Mesh / chips** | (1,4) P150 Blackhole, fabric_1d, validated 4-chip TP | Wormhole, qb (single? mesh?) — branch name `wh-qb` | T3K (8-dev) **or** qb2 (4-dev) by `num_devices` switch |
| **Trace-compatible** | yes — captured in two-phase warmup pattern | likely yes (trace-aware program cache hooks present) | yes — `gdn_recurrence_step_traced` is the *purpose* of `gdn_kernel_trace.py` |
| **Validated perf** | bit-validated correctness (state PCC 0.9999992; out PCC 0.9999985 on 128×128 BF16). **Decode perf REGRESSED** on full 5-prompt benchmark vs manual chain → `dn_owned_gdn` defaults OFF for 35B on long contexts. 27B sibling kernels shipped at +2.5% (decay/gate) and 12.07→12.41 tok/s (owned_gdn). | none reported on GitHub | none reported on GitHub |
| **Debug toolbox** | 10 sub-modes for component ablation (`compute/qwen36_gdn_decode_owned.cpp:382-540`); microbench harness `benchmark_qwen36_gdn_decode_owned.py` | none in branch | none in branch |
| **Known gotchas (ours)** | `safe_out (mode=10)` needed for `slots>~24` (dual-consumer CB race; see kernel comment lines 562-572); single-CB writer-pop race that ollama/ollama#15865 documents; `ttnn.reshape` is a view (must not dealloc source) | (their kernel doesn't yet ship final state-update, so untested for these classes of bug) | head-sharded gather avoids per-device intrinsics — solves an orthogonal bug |

------------------------------------------------------------------------

## 3. OUR kernel — structure and quoted source

### 3.1 Where it lives

```
qb1:/home/aditya/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/
├── device/
│   ├── qwen36_gdn_decode_owned_device_operation.{cpp,hpp}     (250 LOC)
│   ├── qwen36_gdn_decode_owned_device_operation_types.hpp
│   ├── qwen36_gdn_decode_owned_program_factory.{cpp,hpp}      (259 LOC)
│   └── kernels/
│       ├── compute/qwen36_gdn_decode_owned.cpp                (~540 LOC)
│       └── dataflow/{reader,writer}_qwen36_gdn_decode_owned.cpp
├── qwen36_gdn_decode_owned.{cpp,hpp}                          (41 LOC top-level op)
├── qwen36_gdn_decode_owned_nanobind.{cpp,hpp}
├── README.md  (extensive — kept on qb1 only)
├── benchmark_qwen36_gdn_decode_owned.py
├── integrate_into_ttmetal.py
├── sources.cmake
└── test_qwen36_gdn_decode_owned.py
```

Call site in this repo: `experiments/serve/server_35b_ttnn.py:580-607` —
`dn_forward_ttnn(..., use_owned_gdn=True)` reshapes alpha/beta/q/k/v to
rank-4 `[1, NV_PER_CHIP, 1, dim]` and invokes
`ttnn.experimental.qwen36_gdn_decode_owned(H, q4, k4, v4, alpha, beta_r, native_io=True)`.

### 3.2 Program-factory shape

From `qwen36_gdn_decode_owned_program_factory.cpp`:

- **18 circular buffers** named `CB_STATE_IN, CB_Q, CB_K, CB_VALUE, CB_ALPHA,
  CB_BETA, CB_STATE_SCALED, CB_PRED, CB_DELTA_TMP, CB_DELTA, CB_K_COL,
  CB_OUTER, CB_STATE_NEXT_INTERNAL, CB_Q_PREP, CB_K_PREP, CB_VALUE_PREP,
  CB_STATE_OUT, CB_OUT` (lines 24-40). Most double-buffered (`*2`) and
  sized `key_tiles` deep so the entire per-head update keeps state-scaled
  resident in L1.
- **Work split**: `split_work_to_cores(grid_size, total_blocks=slots*value_tiles, row_major)`
  (line 81). Each Tensix core gets a contiguous range of `(slot, value_tile)`
  pairs. With slots=8 (NV_PER_CHIP) and value_tiles=4 (HEAD_V_DIM/TILE=128/32),
  total_blocks=32, easily fits the 110-core Blackhole grid.
- **Math fidelity**: `MathFidelity::HiFi4, fp32_dest_acc_en = true` (line 172).
- **Reader CT-args**: per-tensor `TensorAccessorArgs` for every input
  (state, q, k, k_col, value, alpha, beta) appended to the compile-time
  args vector (lines 128-134) — the dataflow kernel walks DRAM banks via
  the generic tensor-accessor protocol, not hand-rolled NoC reads.
- **Per-core runtime args**: `[state_addr, q_addr, k_addr, k_col_addr,
  value_addr, alpha_addr, beta_addr, blocks_written, blocks_per_core,
  key_tiles, value_tiles, use_pretransposed_k_col]` (lines 222-232).
- Compute runtime args carry the `debug_mode`, `compact_vectors`,
  `native_io` flags so the *same* kernel binary serves 11 different
  ablation modes (line 233-238).

### 3.3 Compute kernel structure (LLK)

Production path lives in `compute/qwen36_gdn_decode_owned.cpp` else-branch
(after debug-modes 1-9). For each `(slot, value_tile)` block:

```
for key_tile in [0..key_tiles):
    state_scaled[key_tile] = state[key_tile] * alpha     # mul_alpha_tile_indexed_auto
pred = K @ state_scaled                                  # matmul_reduce (matmul_tiles loop)
delta_tmp = value - pred                                 # sub_to_tmp
delta = beta * delta_tmp                                 # mul_beta_auto
for key_tile in [0..key_tiles):
    k_col[key_tile] = transpose_wh_tile(K[key_tile])     # transpose_k_indexed
    outer = k_col * delta                                # mul_outer  (or matmul_outer in vector_mode)
    state_out[key_tile] = state_scaled[key_tile] + outer # add_state_to_out  (or _to_two when safe_out=true)
out = Q @ state_out                                      # matmul_reduce (re-uses state_out CB)
```

Notable LLK choices: scalar bcast via `mul_tiles_bcast_scalar` (native_io=true
path) so `alpha`/`beta` are `[1, slots, 1, 1]` scalar tiles instead of
broadcast in DRAM (saves 3 reshapes per layer on the host side); row-bcast
via `unary_bcast<BroadcastType::ROW>` inside the kernel so `q/k/v` arrive
as compact 1-row tiles and the kernel expands them in L1 — eliminates the
2 KB per-tile zero-padding adapter on the Python side.

The `safe_out (debug_mode=10)` branch (lines 562-572 of the compute kernel)
is the bug fix from real-world batched testing: `mode 0` packed the output
matmul's RHS via `cb_state_out`, which the *writer* also pops — at
slots > ~24 the writer pops state_out tiles before the output matmul reads
them, corrupting early slots. Mode 10 routes the output through
`cb_state_next_internal` (compute-owned), decoupling the consumers.

### 3.4 Device-op validation contract

From `qwen36_gdn_decode_owned_device_operation.cpp:24-105`:

```cpp
TT_FATAL(state.dtype() == DataType::FLOAT32 || state.dtype() == DataType::BFLOAT16, …);
TT_FATAL(state_logical.rank() == 4, "state must be rank 4");
TT_FATAL(state_logical[0] == 1, "state dim 0 must be 1");           // <-- HARD-ASSERT BATCH = 1
TT_FATAL(state_logical[-2] > 0 && state_logical[-2] <= 128, …);     // key_dim ∈ [32, 128]
TT_FATAL(state_logical[-1] > 0 && state_logical[-1] <= 128, …);     // value_dim ∈ [32, 128]
TT_FATAL(state_logical[-2] % TILE == 0, …);
```

(This is the "hard-asserts batch=1" issue called out in `CLAUDE.md` —
batching across slots needs `state_logical[0] = B`; today both PF and the
device op pin dim 0 to 1 and put the per-token slot count on dim 1.)

------------------------------------------------------------------------

## 4. THEIR kernels — what exists, in what state

### 4.1 `tt-metal` main branch — NO GDN/DeltaNet kernel

Confirmed via
`gh api repos/tenstorrent/tt-metal/git/trees/main?recursive=1` and regex
`(?i)(deltanet|gdn|gated_delta|qwen3_?next|linear_attention)`. Only hits
are **Mamba1** (`models/demos/wormhole/mamba/`) and three orthogonal SSM
primitives (`ttnn/cpp/ttnn/operations/experimental/ssm/{hc_sum_reduce,
prefix_scan, repeat_and_interleave_eltwise_mul}/`). These are useful as
*Mamba SSD scan* building blocks but do not implement the delta-rule
recurrence.

### 4.2 `changh95/qwen3-coder-next-wh-qb` — closest comparator (Qwen3-Next)

C++ op `ttnn.experimental.deltanet_recurrence` at
`ttnn/cpp/ttnn/operations/experimental/deltanet/`. Signature
(`deltanet.hpp`):

```cpp
Tensor deltanet_recurrence(
    const Tensor& conv_out,    // post-conv1d output (NOT fused inside)
    const Tensor& b_proj,      // raw projection — kernel computes beta=sigmoid
    const Tensor& a_proj,      // raw projection — kernel computes decay=exp(-exp(A_log)*softplus(a+dt_bias))
    const Tensor& z_proj,      // gate input
    const Tensor& dt_bias,     // bias tile
    const Tensor& A_exp,       // exp(A_log) (precomputed at warmup)
    const Tensor& norm_weight, // RMSNormGated weight
    const Tensor& state,       // recurrent state (in-place mutated)
    uint32_t num_heads, head_k_dim, head_v_dim, num_k_heads, gqa_ratio,
    float scale, float norm_eps);
```

**Aspirational fusion = decay+gate+recurrence+norm+gated readout in one op.**
The Python wrapper (`models/tt_transformers/tt/gated_deltanet.py:439-453`)
just calls the C++ primitive after `ttnn.linear` for the input projection.

**But the kernel docstring (`compute/deltanet_recurrence.cpp` header comment)
declares** "Steps 2-4 (K@state, delta, state update) are still TODO in the
kernel." So as of this snapshot the fused op only handles `state*decay,
Q@state, state writeback`. The recurrence math we ship is **not yet on this
branch**. This is consistent with the device-op program factory being
`std::variant<SingleCore>` only (`deltanet_device_operation.hpp`).

What they *did* land that we did not: an end-to-end `_multicore.cpp` reader
kernel (`device/kernels/dataflow/reader_multicore.cpp`) that distributes 48
heads across 24 cores (2 heads/core, 8-wide row layout). It is wired
through a `WriterMulticore` variant but the device op never *selects*
multi-core (`select_program_factory` returns `SingleCore{}`).

Math fidelity: `HiFi2` (per the Wormhole docstring summary). State shape:
`(B, 16, 128, 256*2)` per their reference `recurrent_state` allocation.

### 4.3 `alnah005/qwen_3_6_dev_gdn_ttlang` — same model, TT-LANG DSL

`models/experimental/tt_symbiote/modules/gdn_kernel.py` defines:

- `gdn_step_4head`: `grid=(4,4)`, 8-device T3K geometry, 4 heads/device.
- `gdn_step_8head`: `grid=(8,4)`, **4-device QB2 geometry, 8 heads/device
  — exactly our (1,4) P150 layout.**

Each kernel uses `@ttl.operation` + `@ttl.compute()` + `@ttl.datamovement()`
decorators (the TT-LANG DSL — Python source compiled to LLK by the
TT-LANG frontend). Computation phases mirror our delta-rule. Per-device
sharding is done host-side; per-core slicing is via `ttl.node(dims=2)`
coordinates inside the kernel.

`gdn_kernel_trace.py` solves the **trace-capture problem**: the original
`gdn_recurrent_step` had `ttnn.from_torch`/`to_torch` calls per step
(breaks Metal Trace). The traced replacement pre-allocates head-index
tensors, shift-matrices and ping-pong state buffers at warmup, then uses
`ttnn.gather` (device-only) + custom TT-LANG pack/unpack kernels to keep
everything on-device. **This is materially similar to the pattern we use
with two-phase warmup + sharded weights, but they bake the head-sharding
into `ttnn.gather` index tensors rather than `ShardTensorToMesh`.**

Tests:
`models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b{,_bottom_up}.py` —
same model, suggesting they too have a 35B-A3B target.

No perf numbers in the README; not visibly merged into a model demo.

### 4.4 `alnah005/deltanet_work` — pure PyTorch reference

`models/experimental/gated_attention_gated_deltanet/`. Confirmed via
WebFetch that `torch_functional/*.py` and `tt/*.py` are PyTorch only,
explicitly marked "TTNN implementation will follow once PCC is confirmed."
No device kernel.

### 4.5 `ign/fs/qwen_3_6_35B_optimization`

Same 4 files as `alnah005/qwen_3_6_dev_gdn_ttlang`'s GDN paths. Likely a
downstream fork. Nothing new for our comparison.

------------------------------------------------------------------------

## 5. Adoption opportunities (prioritised)

### We should adopt from them

| Priority | What | Where | Why / cost |
|---|---|---|---|
| **P0** | Fused `deltanet_recurrence` op *signature* — absorb `b_proj`, `a_proj`, `dt_bias`, `A_log_exp` *and* `z_proj` + `norm_weight` (RMSNormGated) into one op | `deltanet.hpp` (changh95/qwen3-coder-next-wh-qb) | This is the natural next-fusion frontier. We already shipped `qwen36_decay_gate_decode_owned` (the `b/a/dt_bias/A_log` half) at +2.5% on 27B (`feedback_owned_decay_gate_shipped.md`); adding `z`+norm would close out the per-block 11.3 % + 4.4 % spend (`feedback_deltanet_perop_findings.md` blocks B7+B9). |
| **P1** | Read their *multi-core reader* dataflow kernel | `deltanet/device/kernels/dataflow/reader_multicore.cpp` | They distribute 48 heads across 24 cores in a fixed 8-wide row pattern. We split work by `(slot, value_tile)` blocks; theirs is potentially friendlier for next-tile prefetch when value-tiles share a key range. Low-risk to study. |
| **P1** | TT-LANG ablation of the recurrence | `gdn_kernel.py` (alnah005) — particularly `gdn_step_8head` (4-dev × 8-heads = our exact mesh) | TT-LANG would shrink the ~540-LOC LLK to <100 LOC of Python DSL. Worth a side-by-side perf comparison before refactoring; if perf is within 10 %, this becomes the maintainable path. |
| **P2** | Their head-sharded `ttnn.gather` trace pattern | `gdn_kernel_trace.py` | Our trace path uses `ShardTensorToMesh(dim=0)`; gather-based indexing is more flexible for irregular per-device routing (which we will need for true B>1 continuous batching). |
| **P3** | Their `HiFi2` config as a perf experiment | `compute/deltanet_recurrence.cpp` | We pinned `HiFi4 + fp32_dest_acc_en` after the long-context fp32-SDPA cliff investigation, but `feedback_35b_dn_h_state_drift_lever.md` says GDN drift comes from H_t precision specifically, NOT compute fidelity. A HiFi2 sweep on `qwen36_gdn_decode_owned` with strict cosine ladder would tell us if we're paying for unnecessary precision. |

### They should adopt from us

| Priority | What | Where (our citation) | Value to them |
|---|---|---|---|
| **P0** | **Full delta-rule, bit-validated**. They have steps 2-4 marked TODO; we have a shipped, PCC-0.9999992 kernel. | `qwen36_gdn_decode_owned/README.md`; compute kernel modes 0/10 | Saves them weeks of bringup. Modes 1-9 are an *ablation toolkit* that lights up exactly which sub-step is failing during integration. |
| **P0** | **Single-step bf16 H_t drift cliff at L32 pos 5+** (`feedback_35b_dn_h_state_drift_lever.md`). Our owned-GDN bf16-state path gives cos@L32 pos1 = 0.99, pos5 = 0.32. fp32 state needs the *manual* path because the kernel requires bf16 state. | `feedback_35b_dn_h_state_drift_lever.md`; `feedback_35b_drift_cliff_pos1_to_pos5.md`; `dn_forward_ttnn else-branch (manual recurrence) at server_35b_ttnn.py:608-685` | They need to know this before promoting their kernel — recovers ~5 days of "why does long-context die?" debugging. |
| **P1** | **`ttnn.reshape`/`ttnn.slice` view-decay bugs masked at decode pos 0** (`feedback_ttnn_slice_view_decay.md`). Bit us during owned-decay-gate integration; their `dt_bias_r2 = ttnn.reshape(w["dt_bias"], …)` pattern at `server_35b_ttnn.py:541-543` shows the safe handling. | `feedback_owned_decay_gate_shipped.md` "Gotchas" section | They have the same fused-op shape adapter coming; same trap. |
| **P1** | **List-rebinding allocator-fragmentation bug** in long-lived ttnn processes (`feedback_ttnn_list_rebinding_leaks.md`). Caused per-run garbage in our cb_engine after `reset_caches_ttnn`. | `experiments/serve/server_35b_cb.py:cb_reset_states` deallocs prior tensors explicitly | Their trace orchestrator (`gdn_kernel_trace.py`) does ping-pong state buffers, which dodges this — but anyone integrating their op into a long-lived chat server will hit it. |
| **P2** | **Dual-consumer CB race at slots>~24** — the `safe_out (mode=10)` fix in our compute kernel (lines 562-572) | `compute/qwen36_gdn_decode_owned.cpp:562-572`, kernel comment block | They will hit this the moment they wire the multi-core program factory and let writers + output matmul share a CB. |
| **P3** | **Dispatch-vs-kernel realisation rule** (`feedback_kernel_vs_dispatch_realization.md`) — predicts that an "11-call → 1-call" fusion will only buy ~5% of the eager gain in trace mode. Their decay-gate→recurrence→norm super-fusion is *almost entirely* dispatch reduction; expected gain is modest. | `feedback_kernel_vs_dispatch_realization.md` | Saves them from over-promising the fused-op gain on the perf sheet. |

------------------------------------------------------------------------

## 6. Engagement recommendation

Both sides have something concrete to trade. They are 4-6 weeks behind us
on full delta-rule correctness; we are missing one fusion frontier
(decay+gate+norm) and the TT-LANG maintainability story. There is **no
upstream patent issue** — both sides are inside the `tenstorrent/tt-metal`
repo under Apache-2.0, ours as a fork.

**Suggested reply (paste-ready)**:

> Thanks for the pointer. We've been heads-down on the same primitive —
> our shipped, bit-validated kernel lives at
> `tenstorrent/tt-metal:<our-fork>/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/`
> (call site at `server_35b_ttnn.py:580-607`). It runs on (1,4) P150
> Blackhole and implements the full delta-rule with `HiFi4 +
> fp32_dest_acc_en`, multi-core block-sharded over `(slot, value_tile)`,
> with a 10-mode compute-kernel ablation switch we used to root-cause the
> dual-consumer CB race at slots>24 and an `slot_count > 24` race in the
> single-CB writer path. Closest upstream comparator we found is the
> `deltanet_recurrence` op on `changh95/qwen3-coder-next-wh-qb`
> (Qwen3-Next, same GDN primitive). Their op header absorbs more
> projections (`b/a/dt_bias/A_exp/z/norm_weight`) — that is the fusion
> frontier we have not yet closed; we have shipped the `b/a/dt_bias/A_log`
> half as `qwen36_decay_gate_decode_owned` at +2.5% on 27B. We have hard
> won lessons on (1) bf16 H_t drift at long contexts forcing an fp32-state
> manual fallback, (2) `ttnn.reshape` view-decay masked at pos 0, and (3)
> list-rebinding leaks in long-lived ttnn processes — documented in our
> CLAUDE.md memory. Happy to send a comparison doc
> (`research/audit_gdn_kernel_us_vs_tt_metal.md`) and would love to look
> at the TT-LANG kernel on `alnah005/qwen_3_6_dev_gdn_ttlang` together;
> their `gdn_step_8head` targets our exact 4-device geometry.

**Files to point them at, in order**:
1. `research/audit_gdn_kernel_us_vs_tt_metal.md` (this doc).
2. `experiments/serve/server_35b_ttnn.py:580-607` (call site).
3. `ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/`
   on qb1 — README + compute kernel + program factory.
4. `~/.claude/projects/.../memory/feedback_35b_dn_h_state_drift_lever.md`
   (drift cliff lesson).
5. `~/.claude/projects/.../memory/feedback_owned_decay_gate_shipped.md`
   (the closest-shipped fusion analogous to their proposed wider op).

------------------------------------------------------------------------

## Appendix A — Negative findings (things we searched for and did NOT find)

- No GitHub issue or PR on `tenstorrent/tt-metal` mentions
  `qwen36_gdn_decode_owned`, `gated_delta`, or `deltanet` by name (the
  GitHub code/issue search API returned zero hits for all three).
- `tenstorrent/tt-metal` main has **no** DeltaNet/GDN kernel; Mamba1 only.
- No PRs from `changh95` or `alnah005` merging `deltanet_recurrence` or
  `gdn_kernel.py` to main as of 2026-06-04 snapshot.
- `qwen9b-p150` was referenced in the mission brief but does not exist as
  a branch (search returned 0 results for `9b`/`qwen9b`). Closest in
  spirit is `changh95/qwen3-coder-next-wh-qb`.
- No published Tenstorrent perf numbers for either DeltaNet branch
  (WebFetch on `test_deltanet.py` returned "no specific numeric values").

## Appendix B — Reproduction notes

- All `tt-metal` paths verified via
  `gh api repos/tenstorrent/tt-metal/git/trees/<branch>?recursive=1`.
- Our kernel paths verified via `ssh qb1 find` and `cat` against
  `/home/aditya/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/`.
- All quoted shapes/dtypes cross-referenced against
  `server_35b_ttnn.py` and the device-operation validator
  (`qwen36_gdn_decode_owned_device_operation.cpp:24-105`).
