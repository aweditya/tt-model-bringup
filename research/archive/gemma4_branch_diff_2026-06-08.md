# Tenstorrent `arg/gemma4_optimizations` vs our Gemma 4 12B — diff (2026-06-08)

Pinned commit read: `tenstorrent/tt-metal@f7d0161` (`models/demos/gemma4/`).

Their tree targets the **Gemma 4 MoE family** (E2B / E4B dense + 26B-A4B / 31B
MoE). They do NOT have a 12B variant in the model card list, but the
`precision_overrides.json` ships an explicit `gemma-4-12B-it` entry, so the
code is meant to load the same checkpoint we work with. Our `12B IT` shape
maps onto their **E4B-with-MoE-off / no PLI** code path (head_dim 256 sliding
/ 512 global, GeGLU dense MLP, partial-rotary 0.25 on global, K=V tying on
global, layer_scalar, final softcap=30).

## 1. TL;DR

- **~6 novel decode-side wins they ship that we don't.** Highest single-ROI
  for our 47 ms/tok traced baseline (B=1, qb1) is **(B3) width-sharded
  decode `rms_norm` via `LayerNormShardedMultiCoreProgramConfig`** — they
  claim ~76 us → <10 us per rms_norm on a single-tile-height activation,
  built lazily on first decode call. We currently fire 337 plain rms_norm
  ops / forward (7/layer × 48) and Round-8 left this entire class unattacked.
  Projected band: **-2 to -8 ms / tok traced** depending on how much of the
  per-norm wall-time is reducible (their probe data and the LLama-Galaxy
  precedent both suggest the upper half of that range).
- Honourable mentions (next-best): **fused QKV matmul** (single matmul +
  `nlp_create_qkv_heads_decode` instead of three separate Q/K/V matmuls);
  **`nlp_concat_heads_decode`** (multi-core sharded concat replacing
  transpose+concat); **`num_links=2` for Blackhole CCL**.
- They do NOT ship: vocab-sharded lm_head + softcap with the `tanh` chain we
  ship; `ttnn.roll`-based RoPE fold (Round 5); `add(..) + mul_scalar` SFPU
  fusion for `layer_scalar` (Round 6); bfp8-on-MLP+lm_head as a 12B default
  (we have it env-gated post Round 9). Several of these were our wins to
  flag upstream.
- They ship **`bfp8` as the precision default for `shared_mlp + attention`
  on Gemma 4 12B IT** — vindicates our Round 8 lever and argues for making
  bfp8 the source default once we re-validate the long-context behaviour.

## 2. Branch overview (key files)

Path = `models/demos/gemma4/`. Sizes in lines.

| File | LoC | What it does |
|---|---:|---|
| `README.md` | 131 | Variants (E2B/E4B/26B-A4B/31B), CI perf table (12.24 / 7.95 / 11.68 / 9.48 tok/s) |
| `config.py` (MeshConfig) | 155 | mesh shape + per-mode (TP/EP/SP/DP) parallelization; experimental async `reduce_scatter_minimal_async + all_gather_async` `allreduce` helper, NOT used in CCL hot path |
| `precision_overrides.json` | 17 | per-model + per-mesh `bfp8` overrides; `12B-it` default = bfp8 MLP + bfp8 attention |
| `tt/model.py` | 1210 | top-level Gemma4Model: embedding, lm_head, per-layer-input (PLI), KV sharing, on-device sampling. Generator-compatible (`ttnn_prefill_forward`, `ttnn_decode_forward`, `prepare_decode_inputs_host`) |
| `tt/layer.py` | 310 | DecoderLayer with attention + dense MLP + (MoE block) + 4–7 RMSNorms + layer_scalar + PLI gate path |
| `tt/attention/decode.py` | 233 | decode SDPA: 2D-cache embedding RoPE, K=V tying, `paged_scaled_dot_product_attention_decode(scale=1.0, sliding_window_size=…)` |
| `tt/attention/prefill.py` | 175 | causal+sliding prefill, chunked-prefill workarounds for the >32k cliff |
| `tt/attention/operations.py` | 430 | `apply_rope`, `apply_rope_decode_peruser`, `nlp_concat_heads_decode`, chunked prefill helpers, `effective_block_size` for HMA cross-group KV |
| `tt/attention/kv_cache.py` + `kv_cache_hybrid.py` | 98 + 152 | per-layer KV alloc + vLLM-style hybrid kv_cache_groups page table (sliding-only short pool, full-long pool) |
| `tt/attention/weights.py` | 190 | **fused Q+K+V column-parallel** weight (single matmul), K=V tying packs K twice for global, o_proj row-parallel |
| `tt/rms_norm.py` | 174 | dual fast path: (a) sharded decode `LayerNormShardedMultiCoreProgramConfig` (lazy, dim-cached), (b) full distributed `rms_norm_pre_all_gather + all_gather + rms_norm_post_all_gather` |
| `tt/shared_mlp.py` | 122 | GeGLU dense MLP, TP'd (col-parallel gate/up, row-parallel down + allreduce); `gelu(fast_and_approximate_mode=True)` |
| `tt/ccl.py` | 156 | CCLManager; **`num_links=2` on Blackhole**; async path **commented out** (TODO sweep); current hot path uses simple `ttnn.all_reduce / all_gather` |
| `tt/common.py` + `tt/generator.py` | 104 + 199 | wires `Gemma4Model` into the shared tt_transformers `Generator`. Prefill trace with deferred lm_head (post-norm tile → host slice → lm_head) |
| `tt/moe.py` + `tt/router.py` + `tt/experts/*` | ~770 | MoE block (relevant to 26B/31B only); top-8 routing + sparse_matmul experts |
| `demo/text_demo.py` | 751 | bench harness: prefill warmup + measured prefill + decode trace + token-by-token loop |

## 3. Per-subsystem diff

Notation: **ADOPT** = clear win for our 47 ms/tok baseline, low scope.
**ADOPT-eval** = looks like a win, needs an isolation probe first.
**DEFER** = blocked by tt_ccl infra or by a larger architectural decision.
**N/A** = applies only to E2B/E4B/MoE, not our 12B IT path.

| Subsystem | Their approach | Our approach | Delta | Verdict |
|---|---|---|---|---|
| Decode trace | `Generator` trace capture, lm_head OUTSIDE the prefill trace (only post-norm hidden returned, host slices last-token tile and runs lm_head); decode trace baked end-to-end including device embedding + on-device argmax | `ensure_decode_trace` + `step_forward_traced`; `_lm_head_argmax` fully inside trace; 100/100 token gate | Their split-prefill lm_head is a TTFT trick (not decode); decode-side trace surface ~equivalent | **N/A for decode**; ADOPT-eval their prefill split when we hit prefill perf |
| Sliding attention | `paged_scaled_dot_product_attention_decode(scale=1.0, sliding_window_size=W)`, head_dim=256, 2 SDPA grids (`(8,4)` if `head_dim≥512` else full), `exp_approx_mode=False` | Same kernel, `sliding_window_size=1024`, custom `SDPAProgramConfig`, HiFi2 + `fp32_dest_acc_en=False` | We match; HiFi2 here is a deliberate B3 recipe with the [[fp32-sdpa-cliff-probe]] guard | parity |
| Global attention (NKV=1, head_dim=512, partial RoPE 0.25) | Same paged SDPA, smaller `(8,4)` SDPA grid for L1 fit; **K=V tying** packed into the fused QKV weight (V = K post-norm); partial RoPE applied via standard `apply_rope` (`ttnn.experimental.rotary_embedding`) since head_dim=512 still uses HF-format cos/sin cache full-width | Two-call paged SDPA pattern (NKV_PER_CHIP>1 workaround); K=V tying via clone in eager path; partial RoPE done in `_apply_partial_rope` | They've eliminated the NKV>1 workaround via K=V tying at WEIGHT-fuse time (no `ttnn.clone(k)` at runtime) | **ADOPT-eval** — K=V weight-fuse saves 8 clones / forward (Round 5 left this) |
| RMSNorm (distributed?) | **Dual fast path**: (a) auto-build `LayerNormShardedMultiCoreProgramConfig` on first decode-shape call (~76 us → <10 us claimed); (b) full Megatron distributed `rms_norm_pre_all_gather + all_gather + rms_norm_post_all_gather` (gated `is_distributed`, off by default on 12B) | Plain `ttnn.rms_norm` everywhere, REPLICATED. 337 calls / forward. Round-8 confirmed LayerNorm ops are MARKER-dropped (un-gauged) but the count alone makes it the biggest unattacked op class | They have the *single-device* sharded norm shipping today AND the distributed primitive ready for the day we go Megatron-TP | **ADOPT (B3)** — their sharded decode-path rms_norm is the single highest-ROI item. Forks `tt/rms_norm.py:42-99` mostly verbatim |
| MLP | GeGLU (col-parallel gate/up + row-parallel down + allreduce), `gelu(fast_and_approximate_mode=True)`, DRAM_MEMORY_CONFIG weight, **bfp8 weight by default for 12B IT** via `precision_overrides.json` | Same TP layout. `gelu(fast_and_approximate_mode=False)` (we sticked with the slower exact variant per Step 0.2 of bringup), Round-4 `activation="gelu"` matmul fusion, bfp8 env-gated (not default). Round 10 DRAM-shard wire-up BLOCKED on `ShardTensor2dMesh` upload contract | They run APPROXIMATE GELU + bfp8 by default; we run EXACT GELU + bf16 default. Their MLP is ~2× cheaper in DRAM bytes and the GELU call is faster | **ADOPT-eval** — flip approximate-GELU after a 100/100 + needle check (we historically refused it per Step 0.2; revisit). bfp8 default safe per our Round 9 ablation |
| LM head | column-parallel along vocab dim, all-gather after softcap; on-device `SamplingGenerator` (top-k / top-p) that consumes sharded logits *without* the all-gather; bfp8 disallowed (their comment: "262k vocab argmax is too lossy") | vocab-shard lm_head (our Round-1 P1, -8% traced) + all_gather + softcap + on-device argmax; bfp8 env-gated. SamplingGenerator NOT used | They have richer sampling (top-k/top-p on device) but their default still gates argmax through all-gather. We win on argmax-shortcut; they win on top-k/top-p without a host roundtrip | **ADOPT-eval (sampling)** — bring `SamplingGenerator` when we add temperature/top-p; defer for greedy chat |
| RoPE (table cache / per-forward hoist) | `rope_caches_2d` stored ROW_MAJOR `[max_seq_len, head_dim]`; per-step `ttnn.embedding(position_idx, cos_cache)` lookup *inside attention*. **2D ROW_MAJOR storage drops the per-call Untilize that TILE-storage forced** (their comment: "240 Untilize ops / decode, ~25 us each") | Our Round 3 hoists the rope lookup out of the per-layer hot path (`_compute_rope_for_forward` → reused across 48 layers). We TILE-store the cos/sin tables | Different angle of attack: they avoid Untilize via ROW_MAJOR storage; we avoid 47/48 redundant lookups via a per-forward cache. **Both wins should compose** | **ADOPT (small)** — convert our `cos_*_tt / sin_*_tt` to ROW_MAJOR storage. Saves ~96 Untilize / forward = same class as Round 5 |
| RoPE (rotate) | Standard `concat([-x2, x1])` via `_rotate_half` + per-user `ttnn.add(mul(x, cos), mul(rotate_half(x), sin))` for batch>1; legacy fused `ttnn.experimental.rotary_embedding` for batch=1 | Round 5 `ttnn.roll(x, half, dim=-1)` + pre-signed sin tables → 3-op chain (`roll + mul + addcmul`), bit-identical | **We win**: 1 fewer op / RoPE call, BIT-identical math. Their per-user path uses 4 ops (mul, mul, rotate_half-which-is-2-ops, add) | (none — flag upstream if they want it) |
| Embedding | On-device `ttnn.embedding` inside the decode trace (token-id input); embed_scale via `ttnn.mul`. Column-parallel embedding for TP>1 | Same shape: on-device embed inside trace. Embed table stays `ttnn.bfloat16` | parity | parity |
| KV cache layout | `[max_num_blocks, num_local_kv_heads, block_size, head_dim]` TILE, REPLICATED across mesh. `effective_block_size` helper handles HMA cross-group sharing (sliding/full sharing one physical buffer) | Same shape; **two caches per sliding layer** (NKV_PER_CHIP>1 hack). Single cache per global layer | They have one cache per layer (K=V tied for global → V = K post-norm at runtime). Cleaner. Our two-cache sliding pattern is a workaround for the kernel contract `cache.padded_shape[1] == NKV_PER_CHIP` | **ADOPT-eval** (medium) — collapse two caches into one if their `paged_update_cache(num_kv_heads=...)` override lets us. Saves cache memory + 1 paged_update_cache per sliding layer (40 ops / forward) |
| Paged kernels | `ttnn.experimental.paged_update_cache(num_kv_heads=…, cache_position_modulo=…)`; SDPA decode with `paged_scaled_dot_product_attention_decode(sliding_window_size=…)` | `paged_fused_update_cache` (Round 1 win) for the combined K+V write; same SDPA decode call | **We win**: paged_fused_update_cache cuts dispatches in half (88 → 88, but kernel does both K and V in one call). They use the unfused `paged_update_cache` x2 | (none — flag upstream) |
| Dtype mix (bf16 / bfp8 / fp32_dest_acc) | bf16 activations, **bfp8 attention + MLP weights** by default on 12B IT, bf16 lm_head + embedding, fp32_dest_acc on every matmul (HiFi4) | Default bf16 weights everywhere (Round-9 revert); env-gates flip bfp8 on MLP + lm_head; HiFi4 + fp32_dest_acc + packer_l1_acc throughout | They ship Round-8's lever as a default. Plus they push attention to bfp8 (Q/K/V/O weights) — we never landed that (only probed in Round 9) | **ADOPT** — re-default bfp8 on MLP + extend to Q/K/V/O attention projections (small Round 9-class follow-on) |
| Other: fused QKV matmul | Single matmul `[hidden] × [hidden, (q+kv+kv)_per_dev]` then `nlp_create_qkv_heads_decode` splits Q/K/V on-device | Three separate matmuls (Q, K, V) | They fire 1/3 the projection ops in attention | **ADOPT-eval** — fork `weights.load_attention_weights:67-95` (fused-on-host) + `operations.split_qkv_heads_decode`. Saves 2 matmuls / layer × 48 = 96 matmul dispatches / forward |
| Other: `nlp_concat_heads_decode` | Multi-core HEIGHT_SHARDED concat replacing `transpose + nlp_concat_heads`; one core per user, batches up to 32; comment: "the old single-core concat ran ~30 us / layer" | We do `transpose + nlp_concat_heads` (single-core) | They eliminate a TM op + spread the concat across cores. Op count: 48 (us) → 48 multi-core (them) — gain is in kernel-time per call | **ADOPT-eval** — small but proven; replace at concat callsite |
| Other: `num_links=2` CCL on Blackhole | `default_num_links()` returns 2 on Blackhole; their comment: "per-layer all-reduces are ~31% of device time, this is the single highest-ROI CCL knob" | We call `ttnn.all_reduce(cluster_axis=1)` with default `num_links` (1 in our path) | They double inter-chip CCL BW; we leave it on the table | **ADOPT** — one-line change in `all_reduce_tt` |
| Other: bounded sliding KV pool | `bounded_sliding_kv_cache` flag + `cache_position_modulo` kwarg → sliding-window layers allocate `sliding_window/block_size` blocks instead of `max_model_len/block_size`. vLLM hybrid `kv_cache_groups`-shaped | We allocate full `MAX_KV=4096` per sliding layer regardless of window | At 4096 it's not memory-bound, but at 32k+ context this becomes the difference between fitting and not | **DEFER** until we want long-context server |

## 4. Specific call-outs (highest ROI first, given our 47 ms/tok traced baseline)

1. **(B3) Sharded decode rms_norm** — `tt/rms_norm.py:42-99` lazy `LayerNormShardedMultiCoreProgramConfig`. 337 LayerNorm / forward, currently un-gauged but Tracy v2's overflow loss means actual kernel time is *unknown*; their comment claims ~10x speedup. Even a 30% norm-time reduction at 5 us / norm = ~0.5 ms / norm-class. Lowball expected: **-1 to -3 ms / tok traced**. **Probe path**: their helper builds the largest core grid whose count divides `dim/32`, with a `block_w = tiles / num_cores` and a `subblock_w` falling from 4. We can drop their `RMSNorm._build_sharded_cfg + _forward_sharded` directly into our `_layer_forward_pos0_paged` rms_norm sites behind an env gate.

2. **`num_links=2` on Blackhole** — single-line edit in `all_reduce_tt` and any `all_gather` site. Round-8 said our matmul = 99.5% of PM-BW so all_reduce is a small share of the budget, but the win here is ~free. **Expected: -0.2 to -0.5 ms / tok traced**.

3. **Fused QKV matmul** — host-side concat of Q+K+V weights into one column-parallel matmul → `nlp_create_qkv_heads_decode` splits at runtime. Saves 2 matmul *dispatches* per attention layer × 48 layers = 96 fewer matmul launches / forward. PM-BW-wise the same bytes get read (we're BW-bound, not dispatch-bound), but Q/K/V projection PM-BW is ~7% of matmul PM-BW combined — collapsing them into one matmul may let the kernel reuse the activation tile across all three outputs, cutting that 7% by some factor. **Expected: -0.3 to -1 ms / tok traced**.

4. **K=V tying fused into the WEIGHT** — global layers do `v_w = k_w` at weight-load time. Eliminates 8 `ttnn.clone(k)` per forward we have (we tried in Round 5; perf was within noise). **Expected: -0 to -0.2 ms / tok traced** — primarily a code-cleanup win.

5. **RoPE table ROW_MAJOR storage** — flip our `cos_*_tt / sin_*_tt` from `TILE_LAYOUT` to `ROW_MAJOR_LAYOUT`. Their comment cites "240 Untilize ops / decode @ ~25 us each" pre-flip. Stacks with our Round-3 cache hoist (we hoist the lookup; they fix the per-lookup Untilize). **Expected: -0.5 to -1.5 ms / tok traced**.

6. **`nlp_concat_heads_decode` (multi-core)** — replace our `transpose + nlp_concat_heads` with their sharded one-core-per-user variant. Comment says single-core was ~30 us / layer × 48 = 1.4 ms / forward; even a 2× speedup ships ~0.7 ms. **Expected: -0.3 to -0.7 ms / tok traced**.

7. **Re-default `bfp8` on MLP (un-revert Round 9) + extend to attention Q/K/V/O** — their `precision_overrides.json` ships `bfp8` for both modules as the 12B IT default. Round 9's needle-haystack failure was traced to prompt shape (IT instruction echo), not bfp8. **Expected: -1.0 to -1.5 ms / tok traced** at the cost of one re-validation pass.

**Cumulative if all 7 land**: ~3-9 ms / tok = 47 ms → 38-44 ms = **22-26 tok/s** (vs today's 21.3).

## 5. Optimizations we have that they don't (flag back to Tenstorrent)

- **`ttnn.roll` + pre-signed sin tables** in `_apply_full_rope` (Round 5) — bit-identical to their `_rotate_half` + concat + neg, one op fewer per RoPE call. Their `tt/attention/operations.py:144-149` still does the 4-op `_rotate_half`.
- **`add(a, b) + mul scalar` SFPU fusion for `layer_scalar`** (Round 6) — their `tt/layer.py:307-308` still does a separate `ttnn.mul(hidden_states, self.layer_scalar)` after the residual add. Their layer would shed one BinaryNg / layer if they fused it via `activations=[MUL_UNARY_SFPU, layer_scalar]` on the residual add.
- **`paged_fused_update_cache` for K+V** (Round 1) — they call `paged_update_cache` twice (K and V separately) in `tt/attention/decode.py:145-165`. We have the fused variant working on disjoint cores.
- **`addcmul` fusion in `_apply_full_rope`** (Round 4) — they do separate `mul + add` after rotate_half. We collapse to one `ttnn.addcmul(value=1.0)`.
- **Vocab-sharded lm_head + on-device argmax** (P1) — their lm_head is column-parallel on vocab, then ALL-GATHER on dim=-1, then they expect the host to argmax. We do `ttnn.argmax` on the sharded logits before the all-gather (saves vocab-wide all-gather bytes when only the argmax is needed). They get this for free when `SamplingGenerator` is enabled but it's gated on `tp > 1 and per_device_vocab ≤ 64K`.
- **`_shard_for_paged_write` minimal reshard** (Round 2) — we simplified the 5-op untile→pad→tile→reshard chain to a 2-op `reshape + to_memory_config`. They still have the 5-op pattern implicit in their `to_memory_config(tt_k, q_sharded_mem)` flow (line 132-134 of `decode.py`). Likely already efficient; worth comparing kernel times.

## 6. Open questions

- **Their CI perf is **9.48 tok/s on T3K 1×8 for 31B** and **11.68 tok/s for 26B-A4B on T3K**. They don't publish a 12B IT number; ours is 21 tok/s on qb1 1×4 P150 (Blackhole). Architecturally Blackhole > Wormhole bandwidth, so this isn't immediately comparable, but the 2× shouldn't be the whole story. Are they targeting correctness + multi-variant generality over single-variant perf?
- **`SamplingGenerator` interaction with our `vocab_sharded` lm_head**: their sampling consumes TP-sharded logits without all-gather; ours requires the all-gather before argmax. If we adopt their sampler, what's the bf16 cliff for argmax on per-chip 16K-vocab logits? (Our 65k all-gather covers 16K×4 chips.)
- **B3 sharded rms_norm at our hidden=3840**: 3840 / 32 = 120 tiles. Their `_build_sharded_cfg` finds the largest core grid dividing 120; candidates are 8×3 (24), 6×4 (24), 8×4 (32 not dividing 120), 10×4 (40 not), 6×5 (30 not), **12×10 (120, gx outside 8-wide WH grid)**. P150 grid is 11×10 = max 110 — so the helper would pick **8×3=24 cores** (block_w=5). Need to verify the actual core grid we'd see at this dim and the realised speedup vs the 76 us → <10 us range they cite for 31B (hidden=5376 = 168 tiles, max 84-core grid).
- **K=V tying weight-fuse interaction with our two-cache-per-sliding-layer workaround** — our two-cache pattern is specifically for NKV_PER_CHIP=2 on sliding layers; global layers have NKV_PER_CHIP=1 already. K=V tying applies to global only in their codebase too. Net: K=V weight-fuse is global-only and our gain is ~8 clones / forward. Recheck Round-5's negative finding on `v_raw` clone removal — the math might differ if V comes from a TIED weight vs a clone.
- **Approximate-GELU** — our Step 0.2 of bringup explicitly recorded `gelu(fast_and_approximate_mode=False)` because `gelu_pytorch_tanh` matched at cos=0.99999803. They use `=True`. Is the approximate kernel actually `gelu_pytorch_tanh` underneath, or a different polynomial that drifts on long context? Probe before flipping.

## 7. Recommended next round (Round 11 brief)

Single round, single lever: **B3 sharded decode rms_norm**. Probe at `_layer_forward_pos0_paged` for the 4 rms_norm sites (input/post_attn/pre_ff/post_ff), gate behind `TT_GM4_SHARDED_RMSNORM=1`. Standard validation stack (100/100 token, L=128 needle, 3-run traced delta). Fork their `tt/rms_norm.py:42-99` `_build_sharded_cfg` + `_forward_sharded` verbatim. Expected: -1 to -3 ms / tok traced.

If B3 doesn't move the needle (norms might be the un-gauged bucket that's actually small kernel work), pivot to the **`num_links=2` + RoPE-ROW_MAJOR + fused-QKV** triple in one round — they share the same validation stack and each ships ~0.5 ms.

---

*Files cached at `/Users/adityasriram/Labs/stanford/cs440lx/tt-model-bringup/.cache/gemma4_branch_diff/` (23 files, full tree under `tt/`). Refresh by re-running `bash .cache/gemma4_branch_diff/fetch_files.sh`.*
