# qb2 Multi-Chip Performance Plan — 2026-05-14

Goal: optimize Qwen3.6-27B TP4 on qb2 toward hardware-ceiling proximity. Use
measured full-decode performance as the source of truth. Do not claim an
optimization win until it passes correctness gates and improves measured
production-path ms/tok.

## Current measured baseline

Server: qb2 persistent `experiments.serve.server_tp`, P25 path active
(`embed_tt`, `cos_table_tt`, `sin_table_tt` in server log).

Command:

```bash
ssh qb2 'cd ~/tt-xla && .venv/bin/python experiments/utils/qb2_tp_generate_bench.py \
  --runs 5 --warmup 1 --max-tokens 30 --chunk-size 30'
```

Result JSON:

```text
~/tt-xla/.cache/qb2_tp_generate_bench/results_20260514_220947.json
~/tt-xla/.cache/qb2_tp_generate_bench/results_20260514_222053.json  # after component endpoint restart
```

Measured, excluding one warmup:

| Metric | Value |
|---|---:|
| median ms/tok | 82.809 |
| mean ms/tok | 82.807 |
| median tok/s | 12.076 |
| min/max tok/s | 12.075 / 12.078 |

Prompt output is coherent ("Paris" continuation). This is the benchmark
baseline for subsequent qb2 changes.

Post-endpoint restart recheck (`results_20260514_222053.json`) matched the
baseline: median 82.808 ms/tok, median 12.076 tok/s. The instrumentation did
not perturb the production generate path.

## Current component decomposition

Server-resident endpoint:

```bash
ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp \
  bench_decode_tp_components --iters 20 --warmup 3'
```

Result JSON:

```text
~/tt-xla/.cache/qb2_tp_components/results_20260514_222202.json
```

Measured medians:

| Component | Median ms |
|---|---:|
| `update_input_buffers` only | 0.718 |
| `execute_trace` only | 82.183 |
| `update_input_buffers + execute_trace` | 82.693 |
| tiny argmax readback | 1.419 |

Interpretation:

- The production timed region is almost entirely trace replay. The remaining
  host update path is less than 1 ms/tok in this sync-bounded probe.
- On-device `plus_one` for position state may still be worth doing for
  cleanliness, but this measurement bounds the immediate opportunity: the
  whole measured update region is ~0.7 ms/tok.
- The next multi-chip optimization should target trace-body work or collective
  layout inside the trace, not host input-buffer overhead.

## Rules for this track

- Main thread owns qb2 multi-chip work.
- Use the persistent server unless a raw ttnn probe is explicitly coordinated.
  `ttnn` opens all chips; concurrent device processes can SIGBUS or hang
  (`feedback_mtp_head_probe.md`).
- Before claiming a win, run the same full-decode benchmark and record JSON.
- Any projected improvement must state the measured component cost it removes
  or the roofline calculation behind it.
- If cosine/top-1 correctness drops below the current path, stop and ablate
  before optimizing (`feedback_correctness_first.md`).

## Immediate hypotheses

### H0: Refresh the P25 breakdown — done

Hypothesis: post-P25 bottlenecks differ enough from the old 7.02 tok/s
breakdown that we need a new decomposition before choosing code changes.

Evidence: P22 removed full-logits readback; P25 moved embedding/cos/sin lookup
on device. Old `feedback_tracy_tp_breakdown.md` still includes 9.4 ms logits
readback and 1.9 ms update_input_buffers from the pre-P22/P25 path.

Validation:

- Implemented server-resident `bench_decode_tp_components` in
  `experiments/serve/server_tp.py` and client command in
  `experiments/serve/client_tp.py`.
- Gate passed: post-endpoint generate benchmark still reproduces ~82.8 ms/tok.
- Result: trace replay is the dominant measured region; update/input overhead
  is small.

Follow-up Tracy/device pass:

- Harness: `experiments/utils/qb2_tp_tracy_profile_probe.py`.
- Synced artifacts:
  `research/probe_logs/qb2_tp_tracy_p25_sync_20260515_0446/.logs/`.
- Summary:
  `.cache/qb2_tp_tracy/p25_manual_sync_summary_20260515_0446.json`.
- Measured replay stayed consistent with H0: `execute_trace` median
  `82.293 ms`, `update+execute` median `82.799 ms`.
- `--sync-host-device` enabled coarse cross-chip timing analysis. After
  applying sync scale/shift, the four devices' median trace starts were within
  about `0.051 ms` across six trace runs, with per-device `TRACE-FW` spans of
  about `82.18-82.22 ms`.
- Interpretation: TP replay is active on all chips over the same 82 ms window.
  This does not yet prove communication/computation overlap inside the token;
  the current logs do not label collective intervals versus matmul intervals
  during replay, and the final TT op/device join failed on missing op `1026`.

### H1: Move remaining position updates on device

Hypothesis: P25 still writes `tok_buf`, `cur_pos_buf`, and `rot_idxs_buf` from
host each step. Moving `cur_pos` / rotary index increment into the trace may
reduce the remaining host update cost, but the measured opportunity is bounded
by the full update bucket.

Evidence:

- P25 benchmark showed net +0.4% after replacing host embedding/cos/sin writes
  with on-device lookup.
- Current code still writes `cur_pos_buf` and `rot_idxs_buf` in
  `server_tp.py:update_input_buffers`.

Validation:

- H0 measured update at ~0.718 ms/tok. Treat this as a cleanup/latency-tail
  candidate, not the main path to hardware-ceiling proximity.
- Correctness gate: generated token sequence matches current top-1 for a short
  deterministic prompt; paged cache positions advance exactly.

### H2: CCL/layout cleanup around attention and MLP exits

Hypothesis: current row-parallel exits may pay avoidable collective/layout cost.
Potential mechanisms include `all_gather_concat`, persistent-buffer async CCL,
or fused matmul+collective APIs.

Evidence:

- Candidate APIs and production references are cataloged in
  `research/multi_chip_optimizations_menu_v2_addendum.md`.
- Current server uses `ttnn.all_reduce` exits in DeltaNet, MLP, and gated
  attention.

Validation:

- Do not start with a broad refactor. Pick one exit path after H0 identifies a
  measurable collective/layout bucket.
- Implement behind a flag or separate helper.
- Correctness gate: inter-chip outputs agree and sequence top-1 matches current
  path.
- Perf gate: same full-decode benchmark improves over 82.809 ms/tok median.

First concrete collective probe:

- Added guarded `probe_explicit_all_reduce_tp`.
- Variant: replace default `ttnn.all_reduce(partial)` exits with
  `ttnn.all_reduce(partial, cluster_axis=1, topology=ttnn.Topology.Linear,
  num_links=1, memory_config=partial.memory_config())`.
- Rationale: qb2 mesh is `(1,4)`, so axis 1 is the populated TP axis. The docs
  scout found persistent-buffer `all_reduce_async` requires width-sharded
  layouts that current `[1, 5120]` partials may not satisfy, so explicit
  `ttnn.all_reduce` is the conservative first test.
- Gate: eager baseline vs explicit top-1 must match before temporary trace
  timing is considered.

Result:

```text
.cache/qb2_tp_collectives/results_explicit_all_reduce_20260514_2318.json
```

The explicit all-reduce variant was accepted and matched baseline top-1
(`2614`). Temporary trace timing was effectively unchanged versus H0:
execute-only median was about `82.13 ms`, and update+execute median was about
`82.65 ms`. A post-probe baseline component sanity check still measured
`execute_trace` at `82.20 ms`. Conclusion: explicit axis/topology kwargs are
safe, but not a standalone performance win. Do not promote this as an
optimization unless it becomes a prerequisite for a later CCL change.

### H2a: Fuse K/V paged cache writes

Hypothesis: the two `paged_update_cache` calls in each full-attention layer can
be replaced by one `paged_fused_update_cache` call. This is trace-body work:
there are 16 full-attention layers, so the operation-count reason to test this
is one fewer cache-write dispatch per full-attention layer per token. No
speedup is assumed until full-decode timing moves.

First validation result:

```text
.cache/qb2_tp_fused_cache/results_overlap_reject_20260514_2236.json
```

The fused op rejected the production writer layout:

```text
input_tensor1 and input_tensor2 must not overlap
```

The current production path uses the same one-core height-sharded L1 writer
config for K and V, so this first variant is invalidated. The next variant is
now server-resident and guarded: keep the default path unchanged, but in probe
mode place K writer input on cores `(0,0)..(7,3)` and V writer input on
`(0,4)..(7,7)` before calling `paged_fused_update_cache`.

Validation:

- Run `probe_fused_paged_update_cache_tp` after restarting the qb2 server with
  the disjoint K/V writer configs.
- Correctness gate: baseline and fused eager argmax match after resetting
  mutable state. If accepted, capture a temporary fused trace and compare
  sync-bounded trace medians against H0.
- Full-decode gate: only if the probe passes, promote to a guarded production
  variant and run the 5-run `qb2_tp_generate_bench.py` benchmark.

Second validation result:

```text
.cache/qb2_tp_fused_cache/results_disjoint_timeout_20260514_155332.json
```

The disjoint K/V writer variant did not return after more than 10 minutes in
the resident server handler. The client was killed and `serve_tp.sh stop`
required SIGTERM after graceful shutdown timed out. Treat this production-path
variant as invalidated for now. A future investigation would need a much
smaller isolated compatibility probe before touching production decode again.
The server endpoint now refuses to run the wedge-prone disjoint path unless
`allow_wedge_prone_disjoint=true` is passed explicitly.

### H3: Reduce manual RoPE dispatches

Hypothesis: current manual partial-RoPE sequence still dispatches many small ops
per full-attention layer. A native/fused RoPE path may reduce that cost if the
current TT-NN API supports Qwen's partial rotary shape.

Evidence:

- Manual rotate-only remains in `server_tp.py:gated_attn_step_tp`.
- Prior native RoPE attempts had shape/padding constraints
  (`feedback_c3_native_rope_abandoned.md`,
  `feedback_native_rope_api_shape.md`), so this is an API-compatibility
  question first.

Validation:

- Read current tt-metal docs/reference before coding.
- Probe Q/K RoPE equivalence at production shapes.
- Only run full decode after the tensor-level gate passes.

First validation result:

```text
.cache/qb2_tp_rope/results_fused_qk_timeout_20260515_0009.json
```

`ttnn.experimental.rotary_embedding_llama_fused_qk` was tested through a
resident-server compatibility endpoint on production-shaped synthetic Q/K
tensors. The handler reached the fused path but did not return after several
minutes. Recovery required killing the client, stopping `server_tp` with
SIGTERM, and resetting qb2 with `tt-smi -r 0,1,2,3`.

Conclusion: invalidate `rotary_embedding_llama_fused_qk` for the production
qb2 path until a smaller isolated probe proves the op cannot wedge on this
mesh/layout. The server endpoint now refuses this path by default unless
`allow_wedge_prone_fused_qk=true` is passed explicitly. Next RoPE attempt, if
any, should use the non-fused slice-first `ttnn.experimental.rotary_embedding`
recipe or a pure tensor-equivalence probe that cannot enter the fused QK kernel.

Second validation result:

```text
.cache/qb2_tp_rope/results_native_partial_pass_20260515_0030.json
```

The slice-first `ttnn.experimental.rotary_embedding` recipe was tested through
the resident qb2 server on production-shaped synthetic Q/K tensors. It applies
native RoPE to only the 64 rotary dims, trims the op's padded head axis back to
the logical Q/K head counts, and concats the 192 pass-through dims.

Result: correctness gate passed for positions `0, 1, 7, 31, 32, 127, 255`.
All positions were accepted, min Q PCC was `0.9999975134451065`, min K PCC was
`0.9999975515805819`, and max tail diff was `0.000914454460144043`.

Conclusion: this candidate is allowed to move to a guarded production trace
variant and measured full-decode comparison. This is not a speedup claim; the
only current evidence is semantic compatibility plus operation-count reduction
inside the RoPE subgraph.

Trace production-variant result:

```text
.cache/qb2_tp_rope/results_native_partial_trace_20260515_0041.json
.cache/qb2_tp_rope/results_manual_baseline_after_native_20260515_0042.json
```

The guarded production variant uses the dynamic on-device cos/sin row with
`token_index=None` rather than baking a Python `token_index` into the trace.
It matched the manual baseline argmax on a reset production forward, captured a
temporary decode trace, and ran a 20-iteration sync-bounded component benchmark.

Same-session 20/3 medians:

| Path | execute_trace ms | update+execute ms |
| --- | ---: | ---: |
| manual P25 | `82.20996847376227` | `82.74531294591725` |
| native partial RoPE | `81.79361710790545` | `82.47091504745185` |
| measured delta | `0.4163513658568263` | `0.27439789846539497` |

Conclusion: native partial RoPE is trace-compatible and has a small measured
component win. It should not become the default on this evidence alone; require
a full-decode comparison before claiming an end-to-end speedup.

## Resident-server decode profile

```text
research/qb2_decode_profile_2026_05_15.md
.cache/qb2_tp_profile/results_decode_op_counts_20260515_0129.json
.cache/qb2_tp_profile/results_decode_op_timed_20260515_0130.json
```

Because qb2 is not currently running a Tracy/profiler-enabled server build, the
fresh profile is a resident-server eager proxy of the same production trace
body, not true per-op device timing inside `execute_trace`.

The count profile records `4268` TTNN calls in one decode body. Largest
categories by count are DeltaNet recurrence (`816`), DeltaNet decay/gate
(`480`), DeltaNet other (`384`), DeltaNet QKV repeat (`336`), matmul (`321`),
attention other (`320`), RoPE (`320`), and RMSNorm (`305`).

The sync-bounded eager timing proxy puts the largest single bucket at DeltaNet
recurrence (`117.053 ms`, `17.42%` of profiled eager-op time), followed by
matmul (`77.583 ms`, `11.54%`), DeltaNet decay/gate (`73.958 ms`, `11.00%`),
and DeltaNet conv (`63.987 ms`, `9.52%`). These percentages rank candidates;
they are not trace replay percentages and are not speedup claims.

Conclusion: the next primary experiment should target DeltaNet recurrence/body
fusion or an equivalent reduction of DeltaNet small-op count. Cache update,
SDPA, and LM-head/IO are not first-order targets in the current measured shape.

## DeltaNet follow-up probes

Artifacts:

```text
.cache/qb2_tp_deltanet/results_recurrence_matmul_20260515_0236.json
.cache/qb2_tp_deltanet/results_softplus_decay_full_decode_20260515_0324.json
```

Reference GDN fused ops (`qwen36_gdn_prepare_decode`,
`qwen36_gdn_decode`, and `qwen36_causal_conv_decode`) exist in the
`experiments/.refs/tt-qwen-36` tree, but qb2's installed `ttnn.experimental`
does not expose them. Treat those as unavailable unless we choose a TTNN rebuild.

The no-rebuild recurrence-as-matmul rewrite was accepted by the API and had a
lower synthetic median on the isolated body (`1.2022379087284207 ms` versus
`1.2938075233250856 ms`), but it failed the strict correctness gate:
`out_vs_manual.pcc = 0.9998950430336081`, below `0.9999`. Conclusion: do not
promote this path.

The lower-risk native softplus candidate replaces the manual
`log(exp(a + dt_bias) + 1)` sequence in the DeltaNet decay/gate path with
`ttnn.softplus(a + dt_bias)`. It passed the tensor gate:

- softplus PCC `0.9999966887762556`, max diff `0.00390625`
- decay PCC `0.9999961572547352`, max diff `0.00390625`
- one-step production argmax matched (`2614`)

Same-session full-decode comparison on prompt `"The capital of France is"`,
20 generated tokens:

| Path | generated IDs match | ms/tok | tok/s |
| --- | ---: | ---: | ---: |
| manual P25 | yes | `82.78564305510372` | `12.079389168172321` |
| native DeltaNet softplus | yes | `82.22602661699057` | `12.161599449016396` |

This is a measured one-prompt/20-token result, so it is valid evidence for this
case, not a broad benchmark. Next validation before default promotion should be
a standard longer/multi-prompt decode run, and then a combined native-softplus
plus native-partial-RoPE run if both remain clean.

## Deprioritized unless new evidence appears

- Distributed RMSNorm Step 1: correct but slower in P24
  (`feedback_distributed_rms_norm_failed.md`).
- DRAM-sharded MLP on single P150/current shape: 2.1x slower in P23
  (`feedback_dram_sharded_mlp_probe.md`).
- Speculative decoding D'3: below break-even acceptance at current evidence
  (`feedback_d3_dont_ship_yet.md`).
