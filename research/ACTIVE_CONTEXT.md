# Active Context — TT-XLA Qwen3.6-27B TP

Read this first after context compaction. Do not re-summarize the whole
`HANDOFF.md` unless the user asks for a full audit.

## Current Status (2026-05-18 evening) — owned_gdn DEFAULTED

The custom single-device GDN/DeltaNet recurrence kernel
(`ttnn.experimental.qwen36_gdn_decode_owned`) is now the production-default
decode path on qb2 (commit `26cad39`). `MAX_POS` was bumped from 256 to 512
(`b4c62ab`) so the trace re-capture matches the qb1 long-context bar.

Measured production decode after the default flip (fresh qb2 cold bootstrap,
`generate_tp` on two canonical prompts × 30 tokens each):
- "The capital of France is" → **80.26 ms/tok = 12.46 tok/s**
- "Implement a JSON parser combinator in Rust" → **80.44 ms/tok = 12.43 tok/s**
- Manual baseline same prompt earlier: 82.92 ms/tok = 12.06 tok/s
- Net delta: **−2.66 ms/tok = +0.40 tok/s = +3.2%**

Gate evidence that justified the default flip (full trail in
`research/owned_gdn_diagnosis_2026_05_18.md`):
- ULP-aware tensor gate at layer 0: kernel is strictly more accurate than the
  manual TTNN broadcast-reduce reference at the prediction step (manual
  internally bf16-quantizes per-element products via `binary_ng` with
  `fp32_dest_acc_en=false`; owned keeps full fp32 dst across the matmul).
  Divergence is exactly 1 BF16 ULP at prediction, intrinsic.
- Teacher-forced argmax coherence on 3 prompts / 236 positions: 230/236
  matches (the 6 flips are all sub-quarter-logit razor ties the manual
  reference itself has at adjacent positions).
- Tier 1 long-context (200 positions, JSON parser prompt): **0/200**
  argmax disagreements (`cosine_ladder_tp_compare_20260518_2105.json`).
- Tier 3 long-context (**500 positions** = qb1 P21 bar): **10/500 = 2.0%**
  argmax disagreements, median cosine 0.9995, **NO cliff** (rolling 50-step
  medians flat at 0.999 from step 0 to step 499)
  (`cosine_ladder_tp_compare_500_20260518.json`).

This is the project's first end-to-end owned TT-Metal custom op landed in
production. The kernel-fusion pattern (Python `binary_ng + reduce` →
fused MMA in a custom op) is now proven and reusable for the other op-heavy
regions of the decode graph.

Known follow-up tracked, not promotion-blocking:
- Eager owned_gdn slowdown on the 2nd+ invocation per server lifetime
  (commit `2905470`). Production decode is traced and unaffected; eager
  probes that toggle modes need a server restart between owned_gdn runs
  until root-caused.

## Current Priority

- Main thread: qb2 multi-chip TP optimization for Qwen3.6-27B on one P150 host
  with `(1,4)` mesh. Custom-kernel fusion is now the active strategy.
- qb1 is reserved by the user right now. Do not run qb1 background agents or
  qb1 device commands until the user says it is free.
- PJRT work is parked.

## Non-Negotiables

- Device work only over `ssh qb1` or `ssh qb2`; never local `ttnn`.
- Persistent server owns the chips. Do not run a second raw `ttnn` process on a
  host while its server is up.
- Validate every optimization. No claimed speedup without measured full decode
  or a defensible component/roofline calculation.
- Sync-bounded timing only.
- Use `.cache/`, `research/`, or `experiments/`; no `/tmp`.
- If qb2 fabric wedges after SIGTERM/SIGKILL: `ssh qb2 'tt-smi -r 0,1,2,3'`.
- **rsync server_tp.py to qb2 BEFORE restarting the server** — local-only
  edits won't take effect after restart; cost of forgetting is one full
  17-min cold bootstrap (2026-05-18 verify-after-flip incident).

## Measured Anchors (post owned_gdn default, 2026-05-18 evening)

- qb2 production full decode: **`80.26-80.44 ms/tok` / `12.43-12.46 tok/s`**
  on canonical prompts at `MAX_POS=512`.
- Same-prompt manual baseline (pre default-flip): 82.92 ms/tok.
- Measured win: 2.6-2.7 ms/tok, ~3.2%.

Historical anchors (pre-2026-05-18, for context — superseded but kept for
trajectory understanding):
- qb2 P25 baseline (manual + trace + MAX_POS=256): `82.8 ms/tok` / `12.08 tok/s`.
- qb2 component medians at P25:
  - `update_input_buffers`: about `0.7 ms`
  - `execute_trace`: about `82.2 ms`
  - `update+execute`: about `82.7 ms`
  - argmax readback: about `1.4 ms`
- Interpretation at P25: bottleneck was the replayed device trace, not host
  update. With owned_gdn defaulted the trace got ~2.6 ms cheaper; the
  attribution should be re-profiled before picking the next fusion target.

## Invalidated Or Safe-But-No-Win Paths

- `paged_fused_update_cache`:
  - same-core K/V writer rejected with overlap error.
  - disjoint K/V writer wedged qb2 for >10 min.
  - endpoint is guarded; do not rerun without explicit user coordination.
- explicit `ttnn.all_reduce(cluster_axis=1, topology=Linear, num_links=1)`:
  - correctness-safe, timing unchanged. Not a standalone win.
- `rotary_embedding_llama_fused_qk`:
  - resident-server probe wedged qb2. Guarded by default.

## Current RoPE Result

- Safer RoPE candidate: slice-first `ttnn.experimental.rotary_embedding` on
  only the 64 rotary dims, then concat 192 pass-through dims.
- Endpoint added locally/remotely: `probe_rope_native_partial_tp`.
- Correctness gate passed on qb2:
  - artifact: `.cache/qb2_tp_rope/results_native_partial_pass_20260515_0030.json`
  - all seven positions accepted.
  - min Q PCC `0.9999975134451065`.
  - min K PCC `0.9999975515805819`.
  - max tail diff `0.000914454460144043`.
- This is not a speedup claim. It only permits a measured trace/full-decode
  variant of native partial RoPE.
- Guarded production trace variant also passed:
  - artifact: `.cache/qb2_tp_rope/results_native_partial_trace_20260515_0041.json`
  - same-session manual baseline:
    `.cache/qb2_tp_rope/results_manual_baseline_after_native_20260515_0042.json`
  - manual median execute/update+execute: `82.20996847376227 ms` /
    `82.74531294591725 ms`.
  - native median execute/update+execute: `81.79361710790545 ms` /
    `82.47091504745185 ms`.
  - measured component deltas: `0.4163513658568263 ms` execute and
    `0.27439789846539497 ms` update+execute. This is not yet a full-decode
    speedup claim.

## Current DeltaNet Results

- Profile note: `research/qb2_decode_profile_2026_05_15.md`.
- Custom-op plan: `research/deltanet_gdn_custom_op_plan_2026_05_15.md`.
- Owned-kernel bring-up handoff:
  `research/gdn_custom_kernel_bringup_2026_05_15.md`.
- Hardware mapping for owned GDN kernel:
  `research/gdn_kernel_hardware_mapping_2026_05_15.md`.
- Important correction from user: `experiments/.refs/tt-qwen-36` is a friend's
  implementation and is known to have GDN recurrence errors. Use that repo only
  as a dataflow/API/reference-implementation source, never as ground truth.
  Correctness ground truth must come from our explicit recurrence formula,
  CPU oracle, and eventually HF/model-level checks.
- Arithmetic-intensity tool:
  `experiments/utils/deltanet_gdn_arithmetic_intensity.py`.
  Default Qwen36 TP recurrence estimate: `0.869 FLOP/byte`,
  `1.588 MB/chip/token` lower-bound traffic, `1.381 MFLOP/chip/token`.
  Interpretation: a GDN native op is valuable for cutting kernel count,
  layout churn, and intermediate traffic, not because the recurrence body is
  large dense compute.
- Profile artifacts:
  `.cache/qb2_tp_profile/results_decode_op_counts_20260515_0129.json` and
  `.cache/qb2_tp_profile/results_decode_op_timed_20260515_0130.json`.
- Limitation: resident-server eager proxy, not true Tracy timing inside
  `execute_trace`.
- Native TTNN Qwen GDN fused ops exist in refs:
  `qwen36_gdn_prepare_decode` and `qwen36_gdn_decode`. These are now only a
  design reference because the referenced implementation is not ground truth.
- qb2 TTNN Tracy build now exposes both native GDN symbols after porting the
  reference op directories and updating CMake/nanobind wiring under
  `~/tenstorrent/tt-metal`.
- Native GDN synthetic one-device recurrence gate passed:
  `.cache/qb2_tp_deltanet/results_native_gdn_synthetic_20260515_0658.json`.
  Shape: `slots=12`, state `[1,12,128,128]`, Q/K/V `[1,12,1,128]`.
  State PCC `0.9999923945472474`, max diff `0.0018737204372882843`;
  output PCC `0.9999854277820642`, max diff `0.0003914386034011841`.
  Synthetic median timing: manual `2.2546573309227824 ms`, native
  `0.06879458669573069 ms`. This is component-only timing, not a full-decode
  speedup claim.
- Real server tensor native-GDN gate currently fails:
  `.cache/qb2_tp_deltanet/results_native_gdn_real_tensors_fp32cast_20260515_0743.json`.
  Native op accepts the mesh tensors, but does not match manual recurrence:
  state PCC `0.08921031954116858`, output PCC `0.21901998019382157`.
  Do not integrate native GDN into production decode yet. Next isolate whether
  this is a MeshDevice/native-op issue or a real-tensor layout/order issue.
- `experiments/serve/scripts/serve_tp.sh` now starts qb2 server with the rebuilt
  TTNN source extension path so the resident process can see the GDN symbols.
  qb2 server is ready after this change.
- No-rebuild recurrence matmul probe:
  `.cache/qb2_tp_deltanet/results_recurrence_matmul_20260515_0236.json`.
  Shape accepted and synthetic median was lower (`1.202 ms` vs `1.294 ms`),
  but output PCC was `0.9998950430336081`, below the strict `0.9999` gate.
  Do not promote.
- Native `ttnn.softplus` for DeltaNet decay/gate:
  `.cache/qb2_tp_deltanet/results_softplus_decay_full_decode_20260515_0324.json`.
  Tensor gate passed, one-step argmax matched, 20 generated IDs matched.
  Same-session full-decode result on prompt `"The capital of France is"`:
  manual `82.78564305510372 ms/tok`, native softplus
  `82.22602661699057 ms/tok`. This is a measured one-prompt/20-token result,
  not yet a broad benchmark.
- CPU oracle / deterministic fixture generator:
  `experiments/utils/gdn_kernel_oracle.py`.
  Local fixture:
  `.cache/qb2_tp_deltanet/gdn_cpu_oracle_fixture_20260515.npz`.
  Summary:
  `.cache/qb2_tp_deltanet/gdn_cpu_oracle_summary_20260515.json`.
  Default component reports all pass against an independent loop reference; the
  tile summary is 4 key tiles, 4 value tiles, and 48 work blocks for TP4.
- Owned first component op source tree:
  `experiments/owned_ops/qwen36_gdn_decay_state/`.
  Current contract is a single-device tiled FP32 or BF16 component op for
  `state_scaled = alpha * state`, with one work block per `(slot, value_tile)`.
  `state` is `[1, slots, key_dim, value_dim]`; `alpha` is a full tiled tensor
  `[1, slots, 32, 32]` with the slot scalar repeated across the tile and the
  same dtype as `state`. Do not pass logical `[1, slots, 1, 1]` alpha to this
  bring-up op.
  It includes reader/compute/writer kernels, TTNN device operation scaffolding,
  nanobind scaffolding, `INTEGRATION.md`, and the remote-only correctness gate
  `test_qwen36_gdn_decay_state.py`.
  qb2 TTNN build integration compiles and links in
  `~/tenstorrent/tt-metal/build_tracy_gcc12_nodist`; the source-package
  `_ttnn.so` / `_ttnncpp.so` were refreshed from that build, and Python import
  confirms `ttnn.experimental.qwen36_gdn_decay_state` is registered.
  Debug-fill 32x32 passes exactly. Debug-copy moves data but shows expected
  pack/readback quantization (`max_abs_diff` about `6.1e-5`).
  The decay ladder passes on qb2 with explicit `--max-abs-diff-threshold
  0.0002` and PCC around `0.9999999`:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decay_state_32x32_decay_hifi4_tol_20260515.json`,
  `.cache/qb2_tp_deltanet/qwen36_gdn_decay_state_32x128_decay_hifi4_tol_20260515.json`,
  `.cache/qb2_tp_deltanet/qwen36_gdn_decay_state_128x32_decay_hifi4_tol_20260515.json`,
  `.cache/qb2_tp_deltanet/qwen36_gdn_decay_state_128x128_decay_hifi4_tol_20260515.json`.
  The strict `1e-5` FP32 max-error gate does not pass; do not describe this as
  FP32-exact. BF16 native-mode ladder passed on qb2 with
  `--dtype bfloat16 --oracle-mode native --max-abs-diff-threshold 0.0003`:
  32x32 PCC `0.9999997731`, max `0.000244140625`; 32x128 PCC `1.0`, max `0`;
  128x32 PCC `0.9999999081`, max `0.000244140625`; 128x128 PCC `1.0`, max `0`.
  This is still only the first component op, not the full GDN recurrence and not
  a decode speedup.
- Owned second component op source tree:
  `experiments/owned_ops/qwen36_gdn_prediction/`.
  Current contract is a single-device tiled FP32 or BF16 component op for
  `prediction = k @ state_scaled`. `state_scaled` is
  `[1, slots, key_dim, value_dim]`; `k` is `[1, slots, 32, key_dim]` with the
  vector repeated across all 32 tile rows; output is `[1, slots, 32, value_dim]`
  with repeated prediction rows. Work blocks are still `(slot, value_tile)`.
  qb2 TTNN build integration compiles and registers
  `ttnn.experimental.qwen36_gdn_prediction`. Initial debug-fill returned zeros
  until `binary_op_init_common(cb_k, cb_state_in, cb_pred_out)` was added,
  matching the earlier decay bring-up failure mode. After that, 32x32 BF16
  debug-fill passed exactly.
  BF16 native prediction ladder passed on qb2 with
  `--dtype bfloat16 --oracle-mode native --max-abs-diff-threshold 0.0005`:
  32x32 PCC `0.9999995677`, max `0.000030517578125`;
  32x128 PCC `0.9999994235`, max `0.00006103515625`;
  128x32 PCC `0.9999999384`, max `0.0000152587890625`;
  128x128 PCC `0.9999991642`, max `0.0001220703125`.
  This is only the prediction component, not delta, outer update, full GDN
  recurrence, or a decode speedup.
- Owned third component op source tree:
  `experiments/owned_ops/qwen36_gdn_delta/`.
  Current contract is a single-device tiled FP32 or BF16 component op for
  `delta = beta * (value - prediction)`. `value` and `prediction` are
  `[1, slots, 32, value_dim]`; `beta` is `[1, slots, 32, 32]` with the slot
  scalar repeated across the tile; output matches `value`. Work blocks are
  `(slot, value_tile)`. The compute kernel currently materializes
  `(value - prediction)` to a temporary L1 CB before multiplying by `beta`.
  qb2 TTNN build integration compiles and registers
  `ttnn.experimental.qwen36_gdn_delta`. 32x32 BF16 debug-fill passed exactly.
  BF16 native delta ladder passed on qb2 on 2026-05-16 with
  `--max-abs-diff-threshold 0.0005`:
  32x32 PCC `0.9999970432825998`, max `0.000244140625`;
  32x128 PCC `0.9999957909040874`, max `0.00006103515625`;
  128x32 PCC `0.9999971434654209`, max `0.000030517578125`;
  128x128 PCC `0.9999970846548589`, max `0.00048828125`.
  This is only the delta component, not outer update, full GDN recurrence, or a
  decode speedup.
- Owned fourth component op source tree:
  `experiments/owned_ops/qwen36_gdn_outer_update/`.
  Current contract is a single-device tiled FP32 or BF16 component op for
  `state_next = state_scaled + k_col * delta`. `state_scaled` is
  `[1, slots, key_dim, value_dim]`; `k_col` is `[1, slots, key_dim, 32]` with
  each key scalar repeated across tile columns; `delta` is
  `[1, slots, 32, value_dim]` with the delta vector repeated across tile rows;
  output matches `state_scaled`. Work blocks are `(slot, key_tile, value_tile)`.
  The compute kernel currently materializes `k_col * delta` to a temporary L1
  CB, then adds that temporary tile to `state_scaled`.
  qb2 TTNN build integration compiles and registers
  `ttnn.experimental.qwen36_gdn_outer_update` when
  `PYTHONPATH=~/tenstorrent/tt-metal/ttnn` is set. Without that PYTHONPATH, the
  venv wheel shadows the rebuilt source package and the owned symbols are not
  visible. 32x32 BF16 debug-fill passed exactly.
  BF16 native outer-update ladder passed on qb2 on 2026-05-16 with
  `--max-abs-diff-threshold 0.0005`:
  32x32 PCC `0.9999989604301218`, max `0.000244140625`;
  32x128 PCC `0.9999992769448623`, max `0.00048828125`;
  128x32 PCC `0.9999992849920574`, max `0.00048828125`;
  128x128 PCC `0.9999991214594598`, max `0.00048828125`.
  This is only the outer-update component, not output contraction, full GDN
  recurrence, or a decode speedup.
- Owned fifth component op source tree:
  `experiments/owned_ops/qwen36_gdn_output/`.
  Current contract is a single-device tiled FP32 or BF16 component op for
  `out = q @ state_next`. `state_next` is
  `[1, slots, key_dim, value_dim]`; `q` is `[1, slots, 32, key_dim]` with the
  query vector repeated across tile rows; output is
  `[1, slots, 32, value_dim]` with repeated output rows. Work blocks are
  `(slot, value_tile)`, reading all key tiles for the selected value tile and
  reducing them with the repeated-row query tiles.
  qb2 TTNN build integration compiles and registers
  `ttnn.experimental.qwen36_gdn_output` when
  `PYTHONPATH=~/tenstorrent/tt-metal/ttnn` is set. 32x32 BF16 debug-fill
  passed exactly. BF16 native output ladder passed on qb2 on 2026-05-16 with
  `--max-abs-diff-threshold 0.0005`:
  32x32 PCC `0.9999990198081428`, max `0.000030517578125`;
  32x128 PCC `0.9999999762572402`, max `0.00000762939453125`;
  128x32 PCC `0.9999999990836789`, max `0.0000019073486328125`;
  128x128 PCC `0.9999997676859007`, max `0.00006103515625`.
  This validates the isolated output contraction only, not the full GDN
  recurrence or a decode speedup.
- Owned component-chain harness:
  `experiments/owned_ops/test_qwen36_gdn_component_chain.py`.
  This is not a fused kernel; it chains the five validated owned component ops:
  decay -> prediction -> delta -> outer_update -> output. It validates
  interface compatibility and BF16-native accumulated behavior against the CPU
  oracle. qb2 ladder passed on 2026-05-16 with
  `--max-abs-diff-threshold 0.001`:
  32x32 state PCC `0.9999987949387302`, state max `0.000244140625`, out PCC
  `0.9999946671314629`, out max `0.000030517578125`;
  32x128 state PCC `0.9999992937225979`, state max `0.00048828125`, out PCC
  `0.9999984056702879`, out max `0.00006103515625`;
  128x32 state PCC `0.9999992127972703`, state max `0.00048828125`, out PCC
  `0.9999985261461334`, out max `0.00006103515625`;
  128x128 state PCC `0.999999113477798`, state max `0.00048828125`, out PCC
  `0.9999977235415836`, out max `0.0001220703125`.
  This is still a correctness/interface gate only and makes no decode speed
  claim.
- Owned fused single-device decode op source tree:
  `experiments/owned_ops/qwen36_gdn_decode_owned/`.
  Contract: `ttnn.experimental.qwen36_gdn_decode_owned(state, q, k, value,
  alpha, beta, *, k_col=None)` updates `state` in place and returns
  `(state, out)`. Inputs are tiled BF16/FP32 on one device:
  `state [1, slots, key_dim, value_dim]`, `q/k [1, slots, 32, key_dim]`
  repeated across tile rows, `value [1, slots, 32, value_dim]` repeated across
  tile rows, `alpha/beta [1, slots, 32, 32]` scalar tiles, and optional
  `k_col [1, slots, key_dim, 32]` repeated across tile columns. Work blocks are
  currently `(slot, value_tile)`.
  The fused compute path keeps intermediate `state_scaled`, prediction, delta,
  transposed K, outer update, and `state_next` tiles in L1 circular buffers,
  writes updated state tiles back to the input state buffer, and writes one out
  tile per block. It transposes repeated-row K inside the compute kernel with
  `transpose_wh_tile` unless `k_col` is supplied. The current production path
  computes output directly from the state-output CB rather than packing a
  duplicate internal `state_next` CB for the output matmul.
  qb2 TTNN build integration compiles and registers the op when
  `PYTHONPATH=~/tenstorrent/tt-metal/ttnn` is set. BF16-native fused ladder
  passed on qb2 on 2026-05-16 with `--max-abs-diff-threshold 0.001`:
  32x32 state PCC `0.9999987949387302`, state max `0.000244140625`, out PCC
  `0.9999946671314629`, out max `0.000030517578125`;
  32x128 state PCC `0.9999992937225979`, state max `0.00048828125`, out PCC
  `0.9999984056702879`, out max `0.00006103515625`;
  128x32 state PCC `0.9999992127972703`, state max `0.00048828125`, out PCC
  `0.9999985261461334`, out max `0.00006103515625`;
  128x128 state PCC `0.999999113477798`, state max `0.00048828125`, out PCC
  `0.9999977235415836`, out max `0.0001220703125`.
  Preallocated output also passed for 32x32 with identical reports:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_32x32_prealloc_bf16_native_20260516.json`.
  This is still a single-device correctness gate and not a decode speedup
  claim.
- Current fused-op optimization debt to preserve across compactions:
  work blocks still reread Q/K for every value tile; Q/K/value are still
  uploaded as repeated-row tiles and alpha/beta as full scalar tiles; the kernel
  still writes state and repeated-row output to DRAM; `transpose_wh_init_short`
  is called per K tile unless `k_col` is supplied; subtract, beta multiply, K
  transpose, outer multiply, and state add use separate L1 CB stages. Optional
  `k_col` is not a standalone measured win unless upstream prepare/QKV can
  produce or keep both row-K and column-K without extra DRAM traffic. Later
  optimize after profiling by grouping multiple value tiles per slot where L1
  permits, keeping Q/K resident longer, avoiding repeated-row vector materialization,
  and reducing temporary CB stages.
- Fused single-device microbench harness:
  `experiments/owned_ops/qwen36_gdn_decode_owned/benchmark_qwen36_gdn_decode_owned.py`.
  It compares the five-op owned component chain against the fused owned op on
  the same 128x128 BF16 fixture, with correctness checked against the CPU
  oracle. Artifact:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_microbench_trace_128x128_20260516.json`.
  Correctness passed: fused output exactly matched component output, and both
  matched the BF16-native oracle at the existing `0.001` max-error gate.
  Sync-only median was `0.016889534890651703 ms`.
  Eager sync-bounded medians: component chain `0.1550390152260661 ms`, fused
  `0.05385500844568014 ms`; measured isolated delta `0.10118400678038597 ms`,
  ratio `2.8788225960904517x`.
  Trace replay medians: component chain `0.026614987291395664 ms`, fused
  `0.021064537577331066 ms`; measured isolated trace delta
  `0.005550449714064598 ms`, ratio about `1.2635x`.
  Interpretation: fusion clearly removes eager dispatch/intermediate overhead,
  but inside trace replay the isolated recurrence saving is much smaller. This
  is component-only timing, not a full-decode speedup claim.
- Fused single-device ablation harness:
  `benchmark_qwen36_gdn_decode_owned.py --slots 12 --include-ablation-modes`.
  Latest qb2 artifact:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_ablation_trace_slots12_128x128_20260516.json`.
  Correctness passed against the BF16-native oracle: component out PCC
  `0.9999967841079703`, fused state PCC `0.9999989660533721`, fused out PCC
  `0.9999967841079703`, and fused out matched component out exactly.
  Trace replay medians:
  - component chain: `0.041463994421064854 ms`
  - full fused op: `0.025850022211670876 ms`
  - mode1 skeleton/read/write/fill: `0.021854997612535954 ms`
  - mode2 decay/write-state: `0.022180029191076756 ms`
  - mode3 decay+prediction: `0.022369669750332832 ms`
  - mode4 decay+prediction+delta: `0.0228099524974823 ms`
  - mode5 update-without-output: `0.02554489765316248 ms`
  - mode6 output-only: `0.021864427253603935 ms`
  Measured trace-only component-vs-fused delta was
  `0.015613972209393978 ms`, ratio `1.6040216167529835x`.
  Interpretation: slots=12 gets above the one-slot trace floor enough to
  localize the fused-kernel cost. Decay, prediction, delta, and output-only are
  nearly at the replay/read-write floor; mode5 is the first substantial jump.
  The first optimization target is therefore the update path: internal K
  transpose, outer multiply, state_next/state_out production, and temporary CB
  traffic. Eager component-chain timing was bimodal in this run, so use the
  trace ladder for bottleneck conclusions. This is still a component microbench,
  not a full-decode speedup claim.
- Follow-up ablation modes 7-9 were added and validated on qb2:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_ablation_modes789_slots12_128x128_20260516.json`.
  Same correctness gate passed: component out PCC `0.9999967841079703`,
  fused state PCC `0.9999989660533721`, fused out PCC `0.9999967841079703`,
  fused out vs component max diff `0.0`.
  Same-run trace medians:
  - component chain: `0.045708962716162205 ms`
  - full fused op: `0.030479510314762592 ms`
  - mode1 skeleton/read/write/fill: `0.02623000182211399 ms`
  - mode2 decay/write-state: `0.026679481379687786 ms`
  - mode3 decay+prediction: `0.027024419978260994 ms`
  - mode4 decay+prediction+delta: `0.02733443398028612 ms`
  - mode5 update-without-output: `0.03026460763067007 ms`
  - mode6 output-only: `0.02637517172843218 ms`
  - mode7 transpose-only-after-delta: `0.02888450399041176 ms`
  - mode8 transpose+outer-no-add: `0.029569491744041443 ms`
  - mode9 update-state-out-only: `0.03016891423612833 ms`
  Interpretation within this same run: K transpose adds about `1.55 us` over
  mode4; outer multiply adds about `0.68 us`; state update/write adds about
  `0.60 us`; duplicate internal state_next staging is negligible because mode5
  is only about `0.096 us` above mode9. This makes pre-transposed K / fused
  QKV-prep producing both row and column K a better next optimization target
  than trying to remove the second state_next CB.
  A direct `transpose_k_all` hoist attempt wedged during correctness sync and
  was reverted. The likely issue is CB reservation/visibility semantics for
  packing multiple transposed tiles into one CB reservation; do not reattempt
  without an isolated microkernel proving the multi-pack pattern.
- Follow-up GDN bottleneck work on qb2, 2026-05-16:
  optional keyword-only `k_col` was added and validated. Internal-transpose
  artifact:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_128x128_kcol_api_internal_20260516.json`.
  Pre-transposed-K artifact:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_128x128_kcol_api_pretransposed_20260516.json`.
  Both matched the 128x128 BF16 oracle with state PCC `0.999999113477798`, state
  max `0.00048828125`, out PCC `0.9999977235415836`, out max
  `0.0001220703125`. Slots=12 timing with `--include-pretransposed-k` showed
  internal-K trace median `0.030665076337754726 ms` and pre-transposed-K trace
  median `0.03065506462007761 ms`; this is correctness-safe but not a standalone
  win because row-K is still needed for prediction and K-col adds a read.
  Artifact:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_pretransposed_k_slots12_128x128_20260516.json`.
  Production compute was changed to use `cb_state_out` directly for the output
  matmul, removing duplicate internal `state_next` packing. Correctness artifact:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_128x128_direct_stateout_internal_20260516.json`.
  A clean timing run showed component-chain trace `0.04136492498219013 ms`,
  fused trace `0.02614909317344427 ms`, and pre-transposed-K trace
  `0.026340014301240444 ms`; artifact:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_direct_stateout_slots12_128x128_20260516.json`.
  A later ablation run with a higher sync floor reported fused trace
  `0.030564493... ms` and mode9 update-state-out-only `0.030194991... ms`;
  artifact:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_direct_stateout_ablation_slots12_128x128_20260516.json`.
  Do not overclaim the absolute 26 us median; compare same-run deltas. Stable
  conclusion: fused GDN removes about 15 us vs the component-chain trace on this
  synthetic recurrence fixture, and the remaining fused-op time is close to the
  skeleton/output/update floor. This is still not a full-decode speedup claim.
- Real resident tensor gate for the owned GDN op, qb2 2026-05-16:
  `probe_deltanet_owned_gdn_real_tensors_tp` was added to the resident server
  and runs through the Unix socket, not as a second raw TTNN process. It builds
  real q/k/v/decay/beta from the live DeltaNet path, broadcasts to the owned
  op's repeated-row/scalar-tile contract, and compares against the current
  manual recurrence without mutating resident SSM state. Layer-0 internal and
  pre-transposed-K artifacts:
  `.cache/qb2_tp_deltanet/results_owned_gdn_real_tensors_layer0_20260516.json`
  and
  `.cache/qb2_tp_deltanet/results_owned_gdn_real_tensors_layer0_pretransposed_20260516.json`.
  Rich diff artifact:
  `.cache/qb2_tp_deltanet/results_owned_gdn_real_tensors_layer0_richdiff_20260516.json`.
  Layer 0 accepted the op and output was tight: output PCC
  `0.999999195685178`, max `0.0009765625`, `0` elements above `0.001`.
  State PCC was `0.9999999792358177`; state max was `0.00390625`, but this was
  a sparse tail: mean `1.1274e-7`, p99 `2.38e-7`, p999 `1.5259e-5`, and only
  `8 / 786432` state elements above `0.001`.
  Full first-token layer sweep artifact:
  `.cache/qb2_tp_deltanet/results_owned_gdn_real_tensor_layer_sweep_20260516.json`.
  All 48 DeltaNet layers accepted the owned op. Original strict gate passed
  for 46/48. The only strict failures were layer 0 as above and layer 60 with
  state max `0.001953125`, state mean `1.6224e-7`, one state element above
  `0.001`, and output max `1.52587890625e-05`. Across all 48 layers, output
  max was at most `0.0009765625` and no output element exceeded `0.001`.
- Guarded production recurrence mode `state.deltanet_recurrence_mode =
  "owned_gdn"` was added but remains disabled by default. Probe endpoint:
  `probe_deltanet_owned_gdn_trace_tp`. Artifact:
  `.cache/qb2_tp_deltanet/results_owned_gdn_trace_probe_20260516.json`.
  Correctness gate: one-step argmax matched (`2614` vs `2614`) and 20-token
  generated IDs matched exactly. Timing was worse, so do not promote:
  same-session manual execute_trace median `82.2640989208594 ms`; owned-GDN
  execute_trace median `83.36730755399913 ms`; manual update+execute median
  `83.0496409907937 ms`; owned update+execute median `84.1441850643605 ms`.
  Full decode loop on the same prompt produced matching IDs, but manual was
  `82.94731070054695 ms/tok` and owned-GDN was `83.76180131454021 ms/tok`.
  Interpretation: correctness-first owned op is compatible with real resident
  tensors, but the current production wrapper adds q/k/value/decay/beta
  tile-broadcast/repeat ops to satisfy the repeated-row contract, more than
  erasing the fused-recurrence component win. Next kernel target is a compact
  vector/scalar owned recurrence contract or a prepare/QKV path that produces
  row-K/col-K and tiled scalar/vector inputs without extra DRAM/read/repeat
  traffic. Do not enable `owned_gdn` in production until a guarded trace is
  both ID-equivalent and faster in same-session full-decode timing.
- Compact-vector follow-up for owned GDN, qb2 2026-05-17:
  `qwen36_gdn_decode_owned(..., compact_vectors=True)` was added to reduce
  q/k/value repeated-row adapter cost. The compute kernel uses a matmul outer
  path for compact q/k/value rows while alpha/beta are still full scalar tiles.
  Do not pass production tensors via a plain `ttnn.reshape([1,NV,1,D])` as the
  compact contract: that failed badly because padded tile rows contained packed
  data, not zeros. Explicitly zero-padding q/k/value rows to 32 fixed the real
  tensor semantics. Artifact:
  `.cache/qb2_tp_deltanet/results_owned_gdn_real_tensors_layer0_compact_zeropad_20260517.json`.
  Layer 0 matched the previous repeated-row behavior: output PCC
  `0.999999195685178`, output max `0.0009765625`, zero output elements above
  `0.001`; state PCC `0.9999999792358177` with the same sparse tail
  (`8 / 786432` state elements above `0.001`, max `0.00390625`).
  Guarded production trace artifact:
  `.cache/qb2_tp_deltanet/results_owned_gdn_compact_zeropad_trace_probe_20260517.json`.
  One-step argmax matched (`2614`), and 20 generated IDs matched exactly.
  Same-session timing was still slower than manual but much closer than the
  repeated-row wrapper: manual execute_trace median `82.22798747010529 ms`,
  compact owned execute_trace median `82.4081456521526 ms`; manual full decode
  `82.88642710540444 ms/tok`, compact owned full decode
  `83.05341660743579 ms/tok`. Do not promote: this is ID-equivalent but still
  not faster. Current bottleneck is integration/contract shaping, especially
  zero-padding q/k/value plus alpha/beta full-tile repeats and output slicing,
  not the fused recurrence arithmetic.
- Native-IO owned GDN follow-up, qb2 2026-05-17:
  `qwen36_gdn_decode_owned(..., native_io=True)` was added. It keeps the proven
  recurrence math but moves adapter work into the compute kernel: q/k/value
  compact row tiles are row-broadcast in L1 with `unary_bcast<ROW>`, alpha/beta
  `[1, slots, 1, 1]` scalar tiles use `mul_tiles_bcast_scalar`, and output is
  written flat as `[1, slots * value_dim]`. This removes the resident
  production path's q/k/value pad, alpha/beta full-tile repeat, and output
  slice for the owned candidate. Synthetic single-device gate:
  `.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_native_io_slots12_128x128_20260517.json`.
  State PCC `0.9999989660533721`, max `0.0009765625`; output PCC
  `0.9999967841079703`, max `0.0001220703125`.
  Resident real-tensor layer-0 gate:
  `.cache/qb2_tp_deltanet/results_owned_gdn_real_tensors_layer0_native_io_20260517.json`.
  Output PCC `0.999999195685178`, max `0.0009765625`, zero output elements
  above `0.001`; state tail matches prior owned runs (`8 / 786432` state
  elements above `0.001`, max `0.00390625`).
  Guarded trace/full decode artifact:
  `.cache/qb2_tp_deltanet/results_owned_gdn_native_io_trace_probe_20260517.json`.
  One-step argmax matched (`2614`), and 20 generated IDs matched exactly.
  Same-session timings on prompt `"The capital of France is"`:
  manual execute_trace median `82.2632450144738 ms`; native-IO owned
  execute_trace median `80.33466408960521 ms`; manual full decode
  `82.90520827285945 ms/tok`; native-IO owned full decode
  `80.99161736899987 ms/tok`. This is a measured one-prompt/20-token guarded
  result, not a broad benchmark. Default production decode still remains
  manual until this is rerun on the standard benchmark set.
- Native-IO owned GDN standard benchmark, qb2 2026-05-17:
  added resident socket command `probe_deltanet_owned_gdn_benchmark_tp` plus
  CLI support. It runs the guarded manual-vs-owned trace/decode probe across a
  prompt set and only accepts prompts whose one-step argmax and generated token
  streams match. Artifacts:
  `.cache/qb2_tp_deltanet/results_owned_gdn_native_io_benchmark_smoke_20260517.json`,
  `.cache/qb2_tp_deltanet/results_owned_gdn_native_io_benchmark_5prompt_20tok_20260517.json`,
  `.cache/qb2_tp_deltanet/results_owned_gdn_native_io_benchmark_5prompt_64tok_20260517.json`.
  The 5-prompt/64-token run did not pass strict token identity:
  only `1 / 5` prompts matched all 64 generated IDs, although all 5 one-step
  argmax checks matched. Aggregate measured timing over those runs was stable
  but is not promotable: manual decode mean `82.92718172015157 ms/tok`,
  native-IO owned mean `80.97573915947578 ms/tok`, mean delta
  `1.9514425606757868 ms/tok` (`2.3531987041836517%`). First divergence
  positions: France prompt step 41, tensor-parallelism prompt step 43, Python
  prompt step 10, bottleneck prompt no divergence through 64, hybrid prompt
  step 14. The 5-prompt/20-token run also failed strict identity on the Python
  and hybrid prompts. Therefore `owned_gdn` remains experimental and default
  production decode remains manual.
- Native-IO owned GDN divergence diagnostics, qb2 2026-05-17:
  added resident socket command `probe_deltanet_owned_gdn_divergence_tp`.
  It is diagnostic only: eager forwards return logits and read top-k values;
  it is not a timing benchmark. Artifacts:
  `.cache/qb2_tp_deltanet/results_owned_gdn_divergence_python_20260517.json`,
  `.cache/qb2_tp_deltanet/results_owned_gdn_divergence_hybrid_20260517.json`.
  Python prompt: eager manual and eager owned matched through 16 tokens; at the
  trace-divergence step 10 the owned eager top two were an exact bf16 tie
  (`2784` and `2007` both `20.375`), while manual had a small `0.25` logit
  margin. Hybrid prompt: eager manual and eager owned diverged at step 14 with
  tiny margins: manual picked `16099` over `8751` by `0.125`; owned picked
  `8751` over `16099` by `0.25`. Current read: the owned native-IO path is
  numerically very close but not strict-token-equivalent over longer greedy
  decode; next correctness work should compare recurrence/state/logit error
  accumulation under a fixed teacher-forced token stream before optimizing more.
- Teacher-forced native-IO owned GDN diagnostics, qb2 2026-05-17:
  added resident socket command `probe_deltanet_owned_gdn_teacher_forced_tp`.
  It first builds a manual-greedy teacher stream, then runs manual recurrence
  and owned GDN on the exact same input tokens/positions. It compares logits
  at each generated step and selected final DeltaNet SSM state buffers. This
  separates real numeric drift from autoregressive trajectory drift. Artifacts:
  `.cache/qb2_tp_deltanet/results_owned_gdn_teacher_forced_python_20260517.json`
  and
  `.cache/qb2_tp_deltanet/results_owned_gdn_teacher_forced_hybrid_20260517.json`.
  Python prompt (`16` forced tokens): no manual-vs-owned argmax difference and
  no forced-token rank change. Worst per-step logit max abs was `0.59375`;
  worst logit PCC was `0.999207043665397`. Final state comparison for selected
  layers: layer 0 PCC `0.9999915620386256`, max abs `0.25`; layer 1 PCC
  `0.9999133964220422`, max abs `0.015625`; layer 2 PCC
  `0.9998986647201842`, max abs `0.005126953125`.
  Hybrid prompt (`20` forced tokens): first manual-vs-owned argmax difference
  and forced-token rank change both occurred at step `14`. Manual picked
  teacher token `16099` with logit `23.5`; owned picked `8751` and assigned
  teacher token `16099` logit `23.25`. The step-14 logit comparison was still
  close overall: max abs `0.4375`, mean abs `0.0691443532705307`, RMS
  `0.08731238543987274`, PCC `0.9993653486862898`. There was an earlier
  non-argmax logit-vector outlier at step `4` (max abs `7.14453125`, PCC
  `0.9613907258289726`), so the next diagnostic should record top absolute
  logit-diff tokens per step. Final selected state comparisons remained close:
  layer 0 PCC `0.9999943059140779`, max abs `0.125`; layer 1 PCC
  `0.9999174040075741`, max abs `0.015625`; layer 2 PCC
  `0.9998951026135912`, max abs `0.00830078125`.
  Current interpretation: this is not a gross GDN-state corruption; it is
  small bf16-level drift that can flip near-tie logits. Do not promote
  `owned_gdn`; next step is top-absolute-diff logit diagnostics and, if needed,
  per-layer teacher-forced state snapshots around hybrid steps 3-5 and 13-14.
- Resident eager profile comparison after native-IO owned GDN, qb2
  2026-05-17:
  added `--deltanet-recurrence-mode {manual,owned_gdn}` to
  `profile_decode_tp_ops` and included
  `ttnn.experimental.qwen36_gdn_decode_owned` in the monkey-patched op list.
  Note: after rsync, qb2 initially served stale bytecode; forcing remote mtimes
  forward with `touch experiments/serve/server_tp.py experiments/serve/client_tp.py`
  and restarting fixed it. Use only the `mode2` artifacts:
  `.cache/qb2_tp_profile/profile_decode_tp_ops_manual_timed_mode2_20260517.json`
  and
  `.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_timed_mode2_20260517.json`.
  This is sync-bounded eager profiling, not direct trace replay timing. It is
  valid for op/category attribution, not as a throughput benchmark.
  Manual profile: `4268` profiled TTNN calls; owned-GDN profile: `3788` calls.
  The owned path removed `480` recurrence-category op calls. DeltaNet recurrence
  category went from `816` calls / `128.445 ms` sync-bounded to `336` calls /
  `51.069 ms`; the owned custom op itself appeared `48` times and summed
  `9.540 ms` sync-bounded. This explains the measured ~`1.9-2.0 ms/token`
  trace/decode improvement: recurrence fusion worked, but it only removes part
  of the total graph.
  Remaining owned-profile shape: `3788` TTNN calls still execute in the eager
  body, including DeltaNet decay/gate (`480` calls), matmul (`321`), DeltaNet
  conv (`288`), DeltaNet qkv repeat (`336`), RoPE (`320`), attention plumbing
  (`320`), RMSNorm (`305`), collectives (`129`), DeltaNet output gate (`240`),
  MLP plumbing (`128`), cache update (`32`), SDPA (`16`), and LM-head/io (`5`).
  Top remaining owned ops by count were reshape `899`, slice `593`, linear
  `321`, rms_norm `305`, add `304`, and mul `288`. Current answer to the
  bottleneck question: DeltaNet recurrence was a bottleneck and the fused kernel
  removed a real chunk of it, but it was not the dominant whole-token bottleneck;
  the post-fusion graph is still dominated by thousands of small layout/elementwise
  ops, skinny matmuls, collectives, and non-recurrence DeltaNet/attention/MLP
  work.
- Current GDN component optimization debt to preserve across compactions:
  component ops intentionally materialize intermediates to DRAM; final fused
  op should keep `state_scaled`, prediction, delta, and state-update dataflow in
  L1/CBs where feasible. Decay repeats scalar `alpha` as a full tile.
  Prediction repeats K rows and rereads K per value tile. Delta uses a temporary
  CB for subtract. Outer-update repeats K across columns, repeats delta across
  rows, and uses a temporary CB for `k_col * delta`. Output repeats Q rows,
  writes repeated output rows, and rereads Q per value tile. Current component
  work blocks are simple and parallel but reread vector tiles; later evaluate
  slot-level or multi-value-tile work grouping if L1 budget allows.
- Resident-server synthetic mesh control endpoint added:
  `probe_deltanet_native_gdn_synthetic_mesh_tp`.
  This isolates whether `qwen36_gdn_decode` fails on MeshDevice tensors in
  general, mesh-sharded tensors, or only on real DeltaNet tensor layout/order.
- Synthetic replicated mesh gate passed:
  `.cache/qb2_tp_deltanet/results_native_gdn_synthetic_mesh_20260515.json`.
  State PCC `0.9999964644317922`, output PCC `0.9999903074626416`,
  pass gate true. This rules out a generic MeshDevice failure.
- Synthetic `ShardTensorToMesh(dim=0)` gate failed:
  `.cache/qb2_tp_deltanet/results_native_gdn_synthetic_mesh_sharded_dim0_20260515.json`.
  State PCC `0.9927821563373678`, max diff `0.053370650857686996`;
  output PCC `0.9917728381167387`, max diff `0.0067208558320999146`.
  This points at mesh-sharded topology/local tile addressing as a real issue.
- Real tensor rerun still failed in `fp32_cast` mode:
  `.cache/qb2_tp_deltanet/results_native_gdn_real_tensors_fp32cast_rerun_20260515.json`.
  State PCC `0.08921031954116858`, output PCC `0.21901998019382157`.
  `current_dtype` is rejected because the native op requires FP32 state:
  `.cache/qb2_tp_deltanet/results_native_gdn_real_tensors_current_dtype_rerun_20260515.json`.
- The optional resident synthetic timing loop was disabled after a failed
  attempt hit an L1 circular-buffer allocation clash from repeated mutating
  native calls in the resident process. Treat mesh synthetic endpoint as
  correctness-only; use a dedicated timing harness later.

## GDN Adapter-Elimination Probes

- Resident SSM rank-4 shape fix, qb2 2026-05-17:
  production DeltaNet SSM is now allocated and reset as
  `[1, N_V_HEADS, K_DIM, V_DIM]` sharded on mesh dim `1`, so each chip's local
  SSM already has the custom-op state contract `[1, NV_PER_CHIP, K_DIM, V_DIM]`.
  Production recurrence now uses `H_4d = dn['ssm']`, and the safe copy-back path
  copies `H_new` directly into `dn['ssm']` instead of reshaping to/from
  `[NV_PER_CHIP, K_DIM, V_DIM]`. Artifacts:
  `.cache/qb2_tp_deltanet/results_owned_gdn_rank4ssm_trace_20tok_20260517.json`,
  `.cache/qb2_tp_profile/profile_decode_tp_ops_manual_rank4ssm_timed_20260517.json`,
  `.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_rank4ssm_timed_20260517.json`,
  and
  `.cache/qb2_tp_deltanet/results_owned_gdn_rank4ssm_benchmark_5prompt_64tok_20260517.json`.
  The one-prompt/20-token guarded trace matched exactly. Profile counts confirm
  the shape cleanup: manual `4268 -> 4172` profiled calls and safe `owned_gdn`
  `3788 -> 3692`, both from `reshape -96` (`DeltaNet_recurrence -48`,
  `DeltaNet_state_update -48`). The standard 5-prompt/64-token owned-GDN gate
  still failed strict identity (`1/5` prompts matched), so this is a valid
  adapter cleanup but not a correctness fix for owned-GDN numeric drift.
- Owned-mode rank-4 q/k/v shape fix, qb2 2026-05-17:
  when `state.deltanet_recurrence_mode` is `owned_gdn` or
  `owned_gdn_inplace`, `deltanet_step_tp` now produces q/k/v directly as
  `[1, NV_PER_CHIP, 1, K_OR_V_DIM]` before q/k RMSNorm and passes them directly
  to `qwen36_gdn_decode_owned`. Manual recurrence mode still uses the old
  rank-2 q/k/v path. Artifacts:
  `.cache/qb2_tp_deltanet/results_owned_gdn_rank4qkv_trace_20tok_20260517.json`,
  `.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_rank4qkv_timed_20260517.json`,
  and
  `.cache/qb2_tp_deltanet/results_owned_gdn_rank4qkv_benchmark_5prompt_64tok_20260517.json`.
  The one-prompt/20-token guarded trace matched exactly. Profile counts confirm
  the expected owned-path cleanup: versus the pre-shape-fix owned profile,
  safe `owned_gdn` moved `3788 -> 3548` profiled calls, all from
  `reshape 899 -> 659` (`-240`). Relative to the rank-4-SSM-only profile,
  this q/k/v change removed another `144` recurrence reshapes
  (`DeltaNet_recurrence 288 -> 144`). The standard 5-prompt/64-token gate still
  failed strict identity (`1/5` prompts matched), so this remains a safe adapter
  cleanup for the experimental owned path, not a default-promotion result.
- `owned_gdn_inplace` was added as an explicit experimental recurrence mode,
  qb2 2026-05-17. It passes the resident `dn['ssm']` view directly into
  `qwen36_gdn_decode_owned` and skips the returned-state reshape/copy-back.
  Artifacts:
  `.cache/qb2_tp_deltanet/results_owned_gdn_inplace_trace_smoke_20260517.json`,
  `.cache/qb2_tp_deltanet/results_owned_gdn_inplace_trace_20tok_20260517.json`,
  `.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_inplace_timed_20260517.json`,
  and
  `.cache/qb2_tp_deltanet/results_owned_gdn_inplace_benchmark_5prompt_64tok_20260517.json`.
  The 4-token smoke and one-prompt/20-token trace matched exactly. The profile
  showed the intended count drop versus `owned_gdn`: total profiled TTNN calls
  `3788 -> 3644`, with `add -48`, `reshape -48`, and `copy -48`;
  `DeltaNet_recurrence 336 -> 288` and `DeltaNet_state_update 144 -> 48`.
  The standard 5-prompt/64-token benchmark failed strict token identity: only
  1/5 prompts matched all 64 generated IDs, despite all one-step argmax checks
  matching. Do **not** promote `owned_gdn_inplace`; the state-aliasing
  assumption is not strict enough for default decode.
- `profile_decode_tp_ops`, `probe_deltanet_owned_gdn_trace_tp`, and
  `probe_deltanet_owned_gdn_benchmark_tp` now accept
  `--deltanet-decay-mode {manual,native_softplus}`. Combined
  `native_softplus + owned_gdn_inplace` artifacts:
  `.cache/qb2_tp_deltanet/results_owned_gdn_inplace_softplus_trace_20tok_20260517.json`
  and
  `.cache/qb2_tp_profile/profile_decode_tp_ops_owned_gdn_inplace_softplus_timed_20260517.json`.
  Native softplus reduced profiled calls further to `3500`;
  `DeltaNet_decay_gate` dropped `480 -> 336`, specifically `add -48`,
  `exp -48`, and `log -48`. It is not promotable: the one-prompt/20-token
  generated IDs diverged early even though the first-token argmax matched.
  Treat this as attribution only, not a valid speedup.

## Current Tracy / Overlap Result

- Tracy-enabled qb2 tt-metal build exists:
  `~/tenstorrent/tt-metal/build_tracy_gcc12_nodist`.
  It was built `--without-distributed`; the TP4 harness still exercised the
  single-host `(1,4)` mesh and timings matched P25.
- Standalone profiling harness:
  `experiments/utils/qb2_tp_tracy_profile_probe.py`.
  Stop the resident server before running it; it opens all four chips.
- Reusable artifact parser:
  `experiments/utils/analyze_tracy_overlap.py`.
- Synced Tracy artifact:
  `research/probe_logs/qb2_tp_tracy_p25_sync_20260515_0446/.logs/`.
  Summary JSON:
  `.cache/qb2_tp_tracy/p25_manual_sync_summary_20260515_0446.json`.
- Measured synced run:
  - `execute_trace` median `82.29258947540075 ms`
  - `update+execute` median `82.79911905992776 ms`
  - argmax readback median `1.2135275173932314 ms`
- Applying `sync_device_info.csv` scale/shift shows coarse chip-level overlap:
  across six trace runs, the four chips' median `TRACE-FW` starts were within
  about `0.051 ms`, and per-chip trace spans were about `82.18-82.22 ms`.
- Limitation: this proves the chips are active together during replay. It does
  **not** yet label matmul-vs-collective intervals inside `execute_trace`, so
  it does not prove communication/computation overlap. The final TT postprocess
  failed while joining host op `1026` to `cpp_device_perf_report.csv`, and most
  `TT_DNN_DEVICE_OP` timing rows lacked expanded metadata.

## Next Bigger Step

1. For GDN fusion, the correctness-first single-device owned fused op now
   passes the 32/128 BF16 ladder, and the slots=12 ablation points at the
   update path as the next local bottleneck. Duplicate state_next staging has
   been removed from the production path; optional pre-transposed K is
   correctness-safe but not a standalone timing win. Real-tensor/20-token
   guarded integration was initially ID-equivalent but slower because of
   contract-shaping repeats. The native-IO variant removed q/k/value
   pad, alpha/beta repeat, and output slice from the owned production path and
   is now ID-equivalent and faster on a guarded one-prompt/20-token run, but
   both `owned_gdn` and `owned_gdn_inplace` fail strict longer multi-prompt
   decode identity. Do not promote them as defaults. Remaining adapter debt:
   the path still reshapes q/k/v/beta to `[1, slots, 1, dim]` views; removing
   those cleanly likely requires either extending the custom op contract to
   accept the native rank-2 tensors already produced by the model path, or an
   upstream prepare path that emits one compact tile per slot without extra
   TTNN reshapes. For TP, keep designing around mesh-sharded local shards, not
   the reference op's replicated/interleaved happy path.
2. For overlap, decide whether to rerun with NOC/fabric event collection or add
   explicit device-visible annotations around collective-heavy regions. Do not
   claim comm/compute overlap from the current trace alone.
3. Native DeltaNet softplus is also not promotable in the combined path because
   strict 20-token generation diverged. Only revisit if a teacher-forced
   numeric diagnostic shows the manual `log(exp(x)+1)` and native `softplus`
   are interchangeable under the full decode token stream.
4. Decide whether native partial RoPE should be combined with native softplus
   behind a guarded mode; require full-decode comparison for the combined path.

## Latest GDN Stepwise Isolation (qb2, 2026-05-17)

Non-negotiables still apply: qb2 only for TP/multi-chip validation, resident
server owns the devices, no default promotion without strict generated-token
identity, and no performance claim without correctness-valid measurement.

Stepwise real-tensor probes were added to
`probe_deltanet_owned_gdn_real_tensors_tp`:

```text
--stepwise
--seed-state {resident,manual_once}
```

Artifacts:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_nativeio_l0_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_tiled_l0_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_compact_l0_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_nativeio_seeded_l0_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_tiled_seeded_l0_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_tiled_seeded_pretransk_l0_20260517.json
```

Zero resident state, layer 0, production native-IO contract:

- `debug4_delta` matched TTNN manual (`max_abs_diff=0.000244140625`).
- Full output matched the gate (`max_abs_diff=0.0009765625`, no values
  above `0.001`).
- Returned state failed the strict state gate because of eight outliers at
  `max_abs_diff=0.00390625` while PCC stayed `0.9999999792358177`.
- Tiled and compact non-native contracts produced the same signature, so this
  is not a native-IO adapter-only problem.

Seeded nonzero state (`--seed-state manual_once`) on the tiled contract:

- `debug2_state_scaled`: `max_abs_diff=0.00390625`, 5 values above `0.001`.
- `debug3_prediction`: `max_abs_diff=0.0078125`, 5 values above `0.001`.
- `debug4_delta`: `max_abs_diff=0.015625`, 3 values above `0.004`.
- `debug5/debug9/full_state_next`: `max_abs_diff=0.015625`, 20 values above
  `0.001`, 4 values above `0.004`.
- Full output: `max_abs_diff=0.001953125`, 1 value above `0.001`.
- Pretransposed K produced the same state/output signature, so the K transpose
  path is not the root cause.

The native-IO seeded debug modes 2-4 are not reliable as substep diagnostics:
they use older non-auto alpha handling and disagree badly with nonzero state.
Use tiled seeded mode for intermediate debug comparisons; use native-IO for the
production full-op contract.

Conclusion: the custom op is algebraically close but not bit/numerically
equivalent to the manual TTNN recurrence. The first meaningful discrepancy is
BF16-scale rounding/materialization in `state_scaled`/prediction, which is then
amplified at the state update. The strict decode failure is therefore not a
remaining reshape/native-IO compatibility issue. Next correctness work should
make the owned kernel mimic the manual TTNN rounding schedule more closely, or
relax only if teacher-forced/generated-token evidence proves the drift is safe
across the standard prompt gate.

## Strict-Reduction GDN Attempt (rejected)

qb2, 2026-05-18: tested whether replacing the owned kernel's contraction
matmul path with a more TTNN-shaped `mul + REDUCE_COL + top-row accumulate`
path would remove the residual GDN drift.

Artifacts:

```text
.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_strict_32x32_native_20260517.json
.cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_strict_slots12_128x128_native_20260517.json
.cache/qb2_tp_deltanet/results_owned_gdn_strictreduce_fixed_stepwise_nativeio_seeded_l0_20260517.json
```

Results:

- Standalone 32x32 native-IO ran, but worsened state PCC to
  `0.9999794459006638` with state max diff `0.0008544921875`.
- Standalone slots=12 128x128 native-IO was BF16-close in absolute terms but
  worse than the matmul-reduce kernel: state PCC `0.9999768405609735`, state
  max diff `0.002227783203125`, output max diff `0.00030517578125`.
- Resident layer-0 seeded real-tensor probe rejected it decisively:
  state PCC `0.9868074718878412`, state max diff `9.5`, output PCC
  `0.9894557125780156`, output max diff `0.392578125`.

Conclusion: this strict-reduce implementation is semantically wrong for the
current tiled production layout. The active qb2 build was restored to the
matmul-reduce production path after this rejection. Do not revive the
REDUCE_COL accumulation path without first building a tiny standalone
orientation test that proves the tile-level reduction contract against TTNN.

Restored-build sanity artifact:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_restored_matmul_nativeio_seeded_l0_20260517.json
```

That restored production path returned to the prior known signature:
state PCC `0.9999997911075303`, state max diff `0.015625`, output PCC
`0.9999971009389962`, output max diff `0.001953125`. The strict state gate
still fails, so this is a restoration/sanity result, not a default-promotion
result.

## 2026-05-18 Default-Safe Equivalence Kernel Status

Non-negotiable conclusion: no owned GDN path is default-safe yet.

Validated in resident qb2 server after removing the failed debug10 path:

```text
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_component_pred_nativeio_seeded_l0_20260518.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_component_pred_fix1_nativeio_seeded_l0_20260518.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_component_pred_fix2_nativeio_seeded_l0_20260518.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_component_pred_fix3_nativeio_seeded_l0_20260518.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_component_pred_comparefix_nativeio_seeded_l0_20260518.json
.cache/qb2_tp_deltanet/results_owned_gdn_stepwise_component_pred_nativek_nativeio_seeded_l0_20260518.json
```

Findings:

- Full owned GDN restored to the known matmul-reduce signature:
  state PCC `0.9999997911075303`, state max diff `0.015625`; output PCC
  `0.9999971009389962`, output max diff `0.001953125`.
- `owned_copy_vs_input` is exact; the copy path is not the first mismatch.
- `isolated_prediction_state_scaled` is exact when `state_scaled` is passed
  into debug mode 3 with alpha=1. The contraction still differs from TTNN
  manual prediction by max `0.0078125`, so matmul contraction arithmetic is an
  independent mismatch.
- The standalone `qwen36_gdn_prediction` component now has opt-in debug modes:
  mode 10 materializes the first transposed K tile, mode 11 materializes
  `state_scaled * k_col` for that tile, mode 12 reduces that tile, and mode 2
  attempts the full strict contraction.
- Mode 10 passed exactly:
  `.cache/qb2_tp_deltanet/results_owned_gdn_component_mode10_kcol_nativeio_seeded_l0_20260518.json`.
- Mode 11 runs but is not default-safe against a TTNN-generated product
  intermediate: PCC `0.9999995951668267`, max diff `0.00390625`, one element
  `> 0.001`:
  `.cache/qb2_tp_deltanet/results_owned_gdn_component_mode11_product_ttnn_expected_nativeio_seeded_l0_20260518.json`.
  Rechecking with the TTNN expected path changed to full multiply then slice
  produced the same result:
  `.cache/qb2_tp_deltanet/results_owned_gdn_component_mode11_product_fullorder_ttnn_expected_nativeio_seeded_l0_20260518.json`.
- Mode 12 runs but is not default-safe against a TTNN-generated one-tile
  reduction: PCC `0.9999996063019564`, max diff `0.0078125`, one element
  `> 0.004`:
  `.cache/qb2_tp_deltanet/results_owned_gdn_component_mode12_reduce_ttnn_expected_nativeio_seeded_l0_20260518.json`.
  Rechecking with the same full-order expected path produced the same result:
  `.cache/qb2_tp_deltanet/results_owned_gdn_component_mode12_reduce_fullorder_ttnn_expected_nativeio_seeded_l0_20260518.json`.
- Mode 2 full strict contraction is wedge-prone/too slow in the resident
  server path; it tied up the server and required killing/restarting the
  resident process. Do not run all debug modes together.
- Direct matmul contract was tested in isolation:
  `.cache/qb2_tp_deltanet/results_owned_gdn_matmul_contract_nativeio_seeded_l0_20260518.json`.
  `ttnn.matmul(k4, state_scaled)` accepts the native shapes
  `[1,12,1,128] @ [1,12,128,128] -> [1,12,1,128]`, but it is not equivalent
  to the current TTNN broadcast-reduce reference: prediction max diff `0.0625`
  and PCC `0.999993423459592`. The custom component remains aligned with the
  broadcast-reduce path (`component_prediction` max diff `0.0078125` vs
  broadcast) and is worse vs matmul (`component_prediction_vs_matmul` max diff
  `0.0625`). Therefore switching the reference to matmul would change the
  numerical contract; it is not a free simplification.

Do not wire the standalone strict-reduce component into the full GDN kernel.
The next correctness step is to eliminate the rare BF16-scale mode 11/12
mismatch against TTNN intermediates or loosen the promotion gate only with a
documented BF16 equivalence argument and generated-token validation.
