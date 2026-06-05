# tt-metal Contributions Packet — CS440LX / tt-model-bringup (2026-06-05)

**Project**: <https://github.com/aweditya/tt-model-bringup> (Stanford CS440LX,
direct TT-Metal bringup of Qwen3.6-27B dense TP, Qwen3.6-35B-A3B MoE, and
Gemma 4 12B `gemma4_unified` on Blackhole P150 — qb1 / qb2 (1,4) mesh).

**Audience**: tt-metal kernel maintainers, tt-llk maintainers, docs owners,
and dev-blog editors.

## Executive summary

This packet collects 20 reproducible findings from ~6 weeks of bringing up
three production-grade LLMs end-to-end on Blackhole P150 — each one cost us
at least a half-day of debugging and survives as a single-line discipline in
our memory index. Tier 1 contains three undocumented kernel/LLK gotchas that
would have saved days (HiFi4 SDPA cliff, `mm_init` prime recipe, sticky
`transpose_wh` unpacker bit). Tier 2 collects high-impact ttnn API bugs in the
"silently returns a view / silently returns garbage" class. Tiers 3-5 catalogue
init/mesh discipline, perf negatives, and model-family precision gotchas that
generalize across Qwen / Gemma. Tier 6 is the one methodology entry — reading
device-op source before forking, and reusing isolation probes — that produced
every entry above.

## What's already in tt-metal main (audits we've already credited)

Before submitting upstream, we audited where our work overlaps with
Tenstorrent's published work-in-progress. These three internal audit docs in
`research/` already credit what came from tt-metal main; the findings below
are explicitly the **delta** that is not yet upstream:

- [`research/audit_gemma4_opts_us_vs_tt_metal_44962.md`](audit_gemma4_opts_us_vs_tt_metal_44962.md)
  — cross-reference of our `gemma4_unified` 12B opts vs tt-metal #44962
  (Gemma 4 text-model optimization umbrella, four E2B/E4B/26B-A4B/31B
  variants). We already shipped 3 of their 18 sub-issues (#44952 embedding,
  #44953 vocab-sharded lm_head, #44957 traced decode). Five of their
  sub-issues we plan to adopt verbatim (paged_fused_update_cache,
  RMSNorm-into-matmul fusion, `concat_heads_decode→o_proj` fusion, MLP
  gate/up fused matmul, redundant `to_memory_config` audit).
- [`research/audit_gdn_kernel_us_vs_tt_metal.md`](audit_gdn_kernel_us_vs_tt_metal.md)
  — audit of our `qwen36_gdn_decode_owned` C++ kernel against tt-metal main
  (no GDN), `changh95/qwen3-coder-next-wh-qb` (their C++
  `deltanet_recurrence`), and `alnah005/qwen_3_6_dev_gdn_ttlang` (TT-LANG
  Python decode kernel, same model as ours).
- [`research/audit_qwen36_us_vs_qwen9b_p150_branch.md`](audit_qwen36_us_vs_qwen9b_p150_branch.md)
  — cross-check of our 27B / 35B-A3B work vs `tenstorrent/tt-metal`
  `qwen9b-p150` branch (their Qwen3.5-9B reference: hybrid attention + GDN,
  masked-bucket prefill, chunk-outer trace prefill, vLLM hooks).

Findings already credited to tt-metal in those audits are not duplicated here.

---

## Tier 1 — Critical undocumented gotchas (would have saved days)

### 1. **SDPA decode HiFi4 + `fp32_dest_acc_en` is broken at length on Blackhole P150**

**What**: Our production paged-SDPA-decode config used
`MathFidelity::HiFi4` with `fp32_dest_acc_en=True`. At
length, the accumulator path produces a hard cliff at ~pos 129
(cos 0.99 → 0.36, top-1 retrieval at L=500 collapses 98% → 28%). Swapping to
HiFi2 + `fp32_dest_acc_en=False` (the "B3" config) eliminates the cliff at
a 1.8% per-token perf cost. `llama70b-galaxy` and `deepseek_v3` references
in tt-metal already use HiFi2 — but it is nowhere documented as
**required** for Blackhole SDPA decode.

**Why it matters**: This is *the* long-context blocker. Took us a week of
binary-searching layer/position pairs to localize. Anyone bringing up >128
token decode on BH will hit it.

**Source**: internal source notes `feedback_fp32_sdpa_cliff_probe.md`,
`feedback_bf16_prefill_drift_cliff.md`,
`feedback_needle_haystack_qb1.md`,
`feedback_qb2_tp_long_context_works.md`; commit `4741253`.

**Suggested form**: tt-metal issue with our needle-haystack reproducer +
docs PR on `SDPAProgramConfig` recommending B3 for Blackhole decode.

### 2. **`mm_init` prime + `mm_init_short` 4-ingredient recipe for transpose+matmul loops**

**What**: Compute kernels that combine `transpose_wh_tile` +
`matmul_tiles` + eltwise inside a loop hit a ~4-iteration TRISC hang on
Blackhole unless **all four** of the following are present:
(a) full `mm_init(in0_cb, in1_cb, out_cb)` ONCE before the loop;
(b) pre-transpose tiles *outside* the loop;
(c) `mm_init_short(in0_cb, in1_cb)` inside the loop after eltwise sections;
(d) explicit `pack_reconfig_data_format(out_cb)` after the pack step.
The tt-llk L3 programming-model doc lists this as `# TODO`.

**Why it matters**: Owned-kernel authors will hang silently and conclude
their algorithm is wrong. Multi-day debug per kernel without the recipe.

**Source**: internal source notes `feedback_mm_init_prime_required.md`,
`reference_tt_llk_frozen_in_tt_metal.md`; commit `d239875`
(`qwen36_decay_gate_decode_owned`).

**Suggested form**: tt-llk docs PR fleshing out the L3 programming-model
TODO + an explanatory comment block in `tt_llk/.../matmul.h`.

### 3. **`transpose_wh_tile` sets a sticky unpacker bit (no `transpose_wh_uninit`)**

**What**: `transpose_wh_init_short` configures `llk_unpack_A_init` with
`transpose=true` and the bit *persists* across subsequent ops in the same
kernel. tt-metal's SDPA dodges this by passing `transpose=1` as a B-flag
to `matmul_tiles` (parameter-scoped, not state-scoped). There is no
documented `transpose_wh_uninit`. tt-metal #15930 has been open since
~2024.

**Why it matters**: Owned kernels chaining transpose into anything other
than that exact `matmul_tiles` overload silently miscompute. Found by
binary-searching tile-level outputs against an HF oracle.

**Source**: internal source notes
`feedback_sdpa_transpose_b_flag_escape_hatch.md`; references tt-metal
#15930.

**Suggested form**: tt-llk PR adding `transpose_wh_uninit` (preferred) OR
docs note on the `matmul_tiles` B-flag escape hatch.

---

## Tier 2 — High-impact ttnn API bugs / view-decay class

### 4. **`ttnn.slice` and `ttnn.reshape` return VIEWS — view-decay corrupts silently, masked at decode pos 0**

**What**: Both ops return views into source tensor memory. Deallocating
the source while the view is live corrupts the view's data. The bug is
*invisible at decode pos 0* because in many attention paths
`attn_out == V` at pos 0, so corruption matches identity. Surfaces as
~0.998-per-layer cosine erosion → terminal cos ~0.49 by layer 63.

**Why it matters**: This is the single most repeated bug class in our
own work. Localizing took multiple "per-layer ladder" sweeps. Any owner
of a fused custom op who frees scratch tensors hits it.

**Source**: internal source notes `feedback_ttnn_slice_view_decay.md`,
`feedback_owned_decay_gate_shipped.md`; commit `a70ce65`.

**Suggested form**: ttnn docs note on `slice`/`reshape` view semantics +
optional `DPrint`/lifetime-warning under `TT_METAL_DPRINT_*`.

### 5. **`ttnn.argmax(keepdim=False)` returns bit-pattern garbage on large `[1, N]`**

**What**: At vocab-sized shapes (`[1, 152064]` for Qwen3.6 dense,
`[1, 248320]` for the sharded variant), `ttnn.argmax(..., keepdim=False)`
returns reinterpreted fp32 bits instead of an integer index. Only
`keepdim=True, use_multicore=True` returns a correct argmax.

**Why it matters**: Vocab-sharded LM-head + on-device argmax is the
single biggest decode-perf win we shipped (P22, +5.1% tok/s on 27B).
Hidden behind this footgun for half a day.

**Source**: internal source notes `feedback_vocab_sharded_lm_head_result.md`;
commit `ef3f336`.

**Suggested form**: tt-metal issue with reproducer.

### 6. **`paged_update_cache`: `input.dim(1)` must equal `page_table.dim(0)` — "dim 1 = batch", not NKV**

**What**: The kernel internally interprets `input.dim(1)` as **batch**,
not as `NKV_PER_CHIP`. This is masked for any model with
`NKV_PER_CHIP=1` (Qwen3.6-35B-A3B) and breaks the moment you fork to a
model with `NKV_PER_CHIP>1` (Gemma 4 sliding KV). The same hidden
contract likely exists in `paged_scaled_dot_product_attention_decode`.
The op also asserts `update_idxs` dtype `INT32` without a user-facing
message.

**Why it matters**: Forking SDPA/cache calls into a new model family
without splitting into N per-NKV-head calls (or relaying out the cache)
costs ~10 minutes of TT_FATAL guessing per regression — exactly the
"read kernel source first" loop Tier 6 #20 is about.

**Source**: internal source notes `feedback_paged_update_cache_nkv_per_chip.md`.

**Suggested form**: tt-metal docs PR clarifying the dim-1 semantics +
better TT_FATAL message for the dtype assertion.

### 7. **`ttnn.rms_norm` shape-drift: `[B,N,D]` vs `[B*N,D]` non-equivalent in bf16**

**What**: Folding rank-3 to rank-2 before `ttnn.rms_norm` introduces
~0.009 mean-abs-difference drift in bf16. The drift then amplifies
through subsequent matmul + all_reduce stages, reaching terminal mad
0.41 vs 0.0 by the lm_head. The two paths *should* be mathematically
equivalent.

**Why it matters**: Code that folds dims for "convenience" silently
poisons downstream layers. Found via per-sub-op cosine + mad ladders;
not obvious from the per-op cosine alone.

**Source**: internal source notes `feedback_ttnn_rms_norm_shape_drift.md`.

**Suggested form**: ttnn docs note on rank-preserving bf16 equivalence
+ optional rank-3 fast path.

### 8. **`ttnn.slice` / `ttnn.concat` on single-row `[1, HEAD_DIM]` TILE_LAYOUT silently broken downstream**

**What**: The single-row case works in isolation but breaks downstream
of `rms_norm` in integration: cosine drops 0.94 → 0.78 with no
single-op cosine drop. Workaround: broadcast K and V to
`[NQ_PER_CHIP, HEAD_DIM]` *before* RoPE / cache update — at 4× KV
memory and ~2.4× perf cost.

**Why it matters**: Multi-day debug, and the workaround taxes every
GQA model with `NKV_PER_CHIP=1`. Integration repro needed because
the micro-probe passes.

**Source**: internal source notes
`feedback_qwen36_attn_rope_single_row_ttnn_bug.md`; commit `c5b0012`.

**Suggested form**: tt-metal issue with the integration reproducer.

### 9. **ttnn batched-expert matmul non-deterministic across calls (~13% rel at small magnitudes)**

**What**: Two sequential `matmul` calls with bit-identical inputs and
bit-identical wrappers produce outputs that differ by max-abs 3.8e-5 —
~13% relative at output magnitudes ~3e-4. Suspected cause:
allocator-state-dependent core scheduling changes the reduction
order, and bf16 is not associative.

**Why it matters**: For per-expert MoE per-slot loops, this rules out a
naive "loop and reuse" implementation; we had to ship a broadcast
variant (v1.4b) at a perf cost. Anyone using ttnn matmul in a
fixed-input regression test will see flaky CIs.

**Source**: internal source notes `feedback_ttnn_moe_per_slot_drift.md`.

**Suggested form**: tt-metal issue (determinism guarantees).

---

## Tier 3 — Init / mesh / shape gotchas

### 10. **`ttnn.ShardTensor2dMesh(dims=(0, None))` keeps leading NCHIPS dim → matmul TT_FATAL `a=1 vs b=4`**

**What**: For the canonical "stack per-chip arrays, shard across chips"
pattern, the 2D sharder keeps a leading 4-dim per chip and a downstream
matmul fails with `TT_FATAL a=1 vs b=4`. The correct API is
`ShardTensorToMesh(mesh, dim=0)`.

**Why it matters**: One-line fix, but the choice of sharder is not
obvious from the API name. We hit this on Gemma 4 v0.1.1.

**Source**: internal source notes `feedback_ttnn_shard_1d_vs_2d.md`.

**Suggested form**: ttnn docs PR on sharder selection.

### 11. **Two-phase warmup discipline REQUIRED for multi-trace coexistence**

**What**: When two paths share a device (e.g. prefill + decode), the
naive "capture prefill → JIT-warm decode → capture decode" order
allocates the decode kernel cache *on top of* prefill's reserved trace
memory. Trace replay then reads garbage or hangs at 99% CPU. Correct
order: compile *all* paths first with `enable_trace=False`, then capture
all back-to-back. Diagnostic env: `TT_METAL_TRACE_ALLOC_TRACKING=1`.

**Why it matters**: Filed upstream as
[tenstorrent/vllm#352](https://github.com/tenstorrent/vllm/pull/352).
Any production stack with prefill + decode hits this.

**Source**: internal source notes `feedback_two_phase_warmup.md`.

**Suggested form**: tt-metal tracing-guide docs PR + a short dev-blog
"how to coexist multiple traces."

### 12. **List/dict rebinding of ttnn tensors leaks device buffers in long-lived processes**

**What**: Python `list[i] = new_tensor` or `dict[k] = new_tensor` drops
the Python reference but the underlying C++ device buffer persists.
Long-running servers fragment the device allocator → forward returns a
*different* garbage on each run with identical inputs. Workaround:
always explicit `ttnn.deallocate(old)` before rebinding.

**Why it matters**: Manifests only after hours of uptime. Took us a
production-server crash to localize.

**Source**: internal source notes `feedback_ttnn_list_rebinding_leaks.md`.

**Suggested form**: ttnn docs note + an optional lifetime-binding
wrapper helper.

---

## Tier 4 — Performance findings

### 13. **P150 firmware v19.5.0+ silently downgrades Tensix 140→120; v19.6.0 ttnn grid = 11×10 = 110**

**What**: P150 firmware silently disables 20 Tensix cores. Roofline math
citing 140 cores or the spec-sheet 745 TFLOPS is wrong on shipping
hardware. Measured numbers on qb1 (fw 19.6.0): DRAM streaming ceiling
404 GB/s (78.9% of 512 GB/s published), per-core L1 user-allocatable
1408 KB, host↔device tilize-bound at 1.4 GB/s, total DRAM 31.83 GB
across 8×3.979 GiB banks.

**Why it matters**: Every "we should hit X tok/s" projection in the
ecosystem is off until people use 110 cores. We re-derived the BH
roofline from scratch.

**Source**: internal source notes `feedback_p150_firmware_core_check.md`,
`feedback_p150_memory_bandwidth_measured.md`,
`reference_p150_roofline_priority.md`.

**Suggested form**: P150 platform-spec docs PR + dev-blog "what your
P150 actually has."

### 14. **`ttnn.experimental.all_reduce_async` is +4% net loss for serial residual-stream decode**

**What**: Async CCL adds ~4% setup overhead and provides no win when
CCLs in the residual stream are serially dependent (no compute window
to overlap with). tt-metal already auto-pipelines sync CCLs on the same
CQ at ~13% natural overlap.

**Why it matters**: We tried this as a "free" win and regressed perf.
A docs note prevents others from the same dead-end.

**Source**: internal source notes `feedback_async_ccl_negative.md`;
commit `1f1eef2`.

**Suggested form**: ttnn docs note "when NOT to use async CCL."

### 15. **ttnn fused-op gap analysis — 9 manual sequences with single-call replacements**

**What**: We audited our decode hot path for places where a fused op
already exists in tt-metal but we'd implemented the manual sequence.
Examples: `rms_norm_pre_all_gather`, `rms_norm_post_all_gather`,
`rotary_embedding_llama_fused_qk`, `paged_fused_update_cache`,
`rotate_half`. These exist but are not surfaced anywhere as a
"patterns" doc. Estimated recoverable: ~30 ms/tok in our stack.

**Why it matters**: Decode authors rediscover these one by one. A
single cheat sheet would save weeks across the ecosystem.

**Source**: internal source notes `feedback_ttnn_fused_ops_gap_analysis.md`.

**Suggested form**: docs PR — "ttnn fused-op cheat sheet for decode."

---

## Tier 5 — Model-family gotchas that generalize

### 16. **Gemma 4 attention has THREE per-head RMSNorms — `q_norm`, `k_norm`, AND `v_norm`**

**What**: `v_norm` is `with_scale=False` (pure x/rms, no learnable
weight). Missing it gives cos=0.95 with V magnitude 3.7× too high. The
correct construction is `ttnn.rms_norm` with an all-ones weight as
identity.

**Why it matters**: Easy to miss in the HF source if you've only
shipped Qwen3.6 (which has q/k_norm but no v_norm). Multi-day debug
before we re-read modeling_gemma3_text.

**Source**: internal source notes `feedback_gemma4_v_norm.md`.

**Suggested form**: tt-metal model-bringup recipe docs PR.

### 17. **Gemma 4 SDPA `scale=1.0`, NOT `1/sqrt(d_k)` — invisible at pos 0**

**What**: HF Gemma 4 sets `self.scaling = 1.0`. Wrong scale is *masked*
at pos 0 because single-token softmax = 1.0 regardless of pre-scale.
Surfaces at pos 1+ as a cliff: cos 0.9995 → 0.26 at the first decode
step.

**Why it matters**: A whole-session debug. Generalizes the lesson "if
your pos-0 cos is great but pos-1 cliffs, suspect a scale that's
invisible to single-token softmax."

**Source**: internal source notes `feedback_gemma4_sdpa_scale_1.md`;
commit `c97bf15`.

**Suggested form**: docs PR for `gemma4_unified` SDPA path + a
diagnostic note in the bringup recipe.

### 18. **Gemma 4 per-layer learned `layer_scalar` multiply at end of decoder**

**What**: Every Gemma 4 decoder layer ends with `h *= layer_scalar`
where `layer_scalar` is a per-layer learned scalar buffer (L0=0.054,
L24=0.82, L47=0.053 in 12B). Missing it makes L0 cosine ~1 (cosine is
scale-invariant) but mad 18× too big, and L1+ collapses.

**Why it matters**: Exact match for the magnitude-bug fingerprint of
Finding #19. Load from safetensors and `ttnn.multiply` at end of layer.

**Source**: internal source notes `feedback_gemma4_layer_scalar.md`.

**Suggested form**: docs PR + diagnostic discipline note.

### 19. **Cosine ≥ 0.999 with large `mad` = magnitude bug — always also check `mad`**

**What**: Cosine is invariant under scalar multiplication, so a
sub-step with cos≥0.999 and a sudden `mad` jump is the fingerprint of
a missing per-tensor scale. Bit us on Qwen 35B GDN Q scaling
(missing `× 1/sqrt(d_k)`, commit `369cb32`) and on Gemma 4
`layer_scalar`. Always report both.

**Why it matters**: A one-line discipline that catches a whole class of
bringup bugs that would otherwise survive a cosine-only ladder.

**Source**: internal source notes `feedback_cos_not_enough_also_check_mad.md`,
`feedback_qwen36_gated_dn_q_scale.md`.

**Suggested form**: dev-blog post on bringup diagnostic discipline.

---

## Tier 6 — Methodology / process

### 20. **Read kernel source FIRST when forking into a new shape regime + reuse existing isolation probes**

**What**: Many ttnn kernel contracts (NKV, head_dim, B, dtype) are only
documented in `TT_FATAL` macros inside the device-op source. Read them
*before* writing the call site. Then fork the isolation probe under
`experiments/cb/isolate/` (paged_sdpa, paged_update_cache, etc.) — a
probe iterates in seconds; a full forward iterates in minutes
(~75s bootstrap).

**Why it matters**: This single discipline produced essentially every
Tier 1-5 finding above. Adopting it as a CONTRIBUTING.md addition
would make community bringup contributions an order of magnitude
faster to land.

**Source**: internal source notes `feedback_read_kernel_source_first.md`,
`feedback_use_existing_isolation_probes.md`.

**Suggested form**: tt-metal `CONTRIBUTING.md` addition on the
kernel-bringup workflow.

---

## Next steps for engagement

**We'd lead with these 3-5 findings** (highest debug-time-saved per
maintainer-hour-to-triage):

1. **#1 SDPA HiFi4 cliff** — single issue with our needle-haystack
   reproducer; blocks any >128-tok decode on Blackhole.
2. **#4 ttnn view-decay** — single issue + docs note; rediscovered by
   every owned-kernel author.
3. **#2 + #3 LLK init recipe + sticky transpose** — combined tt-llk
   docs PR; the L3 doc is already a `TODO` stub.
4. **#13 P150 firmware core downgrade** — short platform-spec docs PR;
   gates every roofline number in the ecosystem.
5. **#11 Two-phase warmup** — tracing-guide docs PR (we already filed
   the symptom against vllm#352; the upstream pattern belongs in
   tt-metal docs).

**Suggested submission format**:

- **Single umbrella issue** ("CS440LX / tt-model-bringup contributions
  packet, 2026-06-05") with one checkbox per finding above. Easier to
  triage than 20 separate issues; we can split out individual issues
  per maintainer request.
- **Two grouped docs PRs**: one against tt-metal docs (Findings #1,
  #6-#8, #10, #11, #13-#15, #16-#19), one against tt-llk docs
  (#2, #3).
- **Two follow-up tt-metal issues** for the determinism / API-bug
  class (#5 argmax garbage, #9 matmul non-determinism).
- **One dev-blog draft** combining #11 (two-phase warmup), #13
  (real P150 numbers), and #19 (cos+mad discipline).

**Our offer**: every finding above has a minimal reproducer in our
`experiments/cb/isolate/` shelf (Read kernel source first → fork an
existing probe → iterate in seconds, per Finding #20). On request we
will:

- Cut each reproducer down to a self-contained `.py` runnable on a
  single P150 with no model weights (where possible).
- File the issues / PRs against tt-metal and tt-llk in the suggested
  format above.
- Provide our HF-vs-TT per-layer cosine + mad ladder harness
  (`experiments/utils/cosine_ladder_*.py`) for any finding whose repro
  benefits from a reference oracle.

Contact: <https://github.com/aweditya/tt-model-bringup> (issues + PRs
welcome there; happy to mirror upstream).
