# Gemma 4 31B bringup scope — delta vs the 12B production path

Status: scoping. Research-only doc. Author: agent (session 2026-06-07).
Target host: qb2 (4 P150s, 32 GB/chip). Production reference:
`experiments/serve/server_gemma4_unified_ttnn.py` (2017 lines, 47.5 ms/tok
traced after P1 vocab-shard lm_head).

User assertion to verify: "12B → 31B is mostly config delta, would be really
simple because most of the things stay the same." TL;DR: **mostly true for
the model code path; memory budget is the real risk** (see §3 — it fits on
(1,4) P150 with ~3–4 GB/chip headroom, narrower than 12B's ~22 GB/chip
headroom).

## 1. Confirm the model exists

WebFetch results:

- **`google/gemma-4-31B`** — 30.7B params, BF16, 256K context window, Apache 2.0,
  **not gated** (publicly downloadable).
- **`google/gemma-4-31B-it`** — instruction-tuned variant, also Apache 2.0,
  **not gated**. (Gemma 4 12B IT was previously gated per our memory, so this
  is a documented improvement.)
- Same `Gemma4ForConditionalGeneration` family arch; same `gemma4_text`
  text subconfig.

The "12B IT was gated" memory may now be stale for the 4-family; both 31B
checkpoints download without an access request.

## 2. Config diff: 31B vs 12B (text-only fields)

From `WebFetch` of both `config.json` payloads:

| Field | 12B (`gemma-4-12b-it`) | 31B (`gemma-4-31B`) | Ratio | Note |
|---|---|---|---|---|
| `model_type` (top) | `gemma4_unified` | `gemma4` | — | **DELTA**: 12B is "unified" (text+vision+audio in one config); 31B drops "unified". Vision still present. No audio_config in 31B. |
| `architectures` | `Gemma4UnifiedForConditionalGeneration` | `Gemma4ForConditionalGeneration` | — | Different HF class. Text forward should be identical. |
| `text_config.model_type` | `gemma4_unified_text` | `gemma4_text` | — | See above. |
| `hidden_size` | 3840 | **5376** | 1.40× | Drives all matmul shapes. Still divisible by 4 (1344/chip). |
| `num_hidden_layers` | 48 | **60** | 1.25× | More layers = more per-layer trace memory + longer JIT compile. |
| `num_attention_heads` (NQ) | 16 | **32** | 2.00× | Q heads double. On (1,4) mesh: NQ_PER_CHIP 4 → 8. |
| `num_key_value_heads` (sliding) | 8 | **16** | 2.00× | Doubles. On (1,4) mesh: NKV_PER_CHIP_SLIDING 2 → 4 (still clean). |
| `num_global_key_value_heads` | 1 | **4** | 4.00× | **DELTA**: 12B has NKV_GLOBAL=1 (requires the two-call paged decode workaround per `[[reference-gemma4-two-call-paged-decode]]`); 31B has NKV_GLOBAL=4, which divides cleanly onto (1,4) as NKV_PER_CHIP_GLOBAL=1. **POSSIBLY simplifies** the global-attention path (no two-call workaround needed). |
| `head_dim` (sliding) | 256 | 256 | 1.00× | Same. |
| `global_head_dim` | 512 | 512 | 1.00× | Same. |
| `intermediate_size` | 15360 | **21504** | 1.40× | INTERMEDIATE_PER_CHIP 3840 → 5376. |
| `vocab_size` | 262144 | 262144 | 1.00× | **Same tokenizer**. Cleanly shardable / 4 = 65536. |
| `max_position_embeddings` | 262144 | 262144 | 1.00× | Same. |
| `sliding_window` | 1024 | 1024 | 1.00× | Same. |
| `layer_types` pattern | 5 sliding : 1 global (×8 = 48) | 5 sliding : 1 global (×10 = 60) | — | **Same 5:1 pattern**; just more cycles. |
| `final_logit_softcapping` | 30.0 | 30.0 | 1.00× | Same. |
| `rms_norm_eps` | 1e-06 | 1e-06 | 1.00× | Same. |
| `attention_bias` | false | false | — | Same. |
| `attention_k_eq_v` | true | true | — | Same. |
| `hidden_activation` | gelu_pytorch_tanh | gelu_pytorch_tanh | — | Same. |
| `rope_theta` (sliding) | 10000.0 | 10000.0 | 1.00× | Same. |
| `rope_theta` (global) | 1000000.0 | 1000000.0 | 1.00× | Same. |
| `rope` `partial_rotary_factor` (global) | 0.25 | 0.25 | 1.00× | Same. |
| `tie_word_embeddings` | true | true | — | Same. (lm_head shares embed weight.) |
| `num_kv_shared_layers` | 0 | 0 | — | Same; no cross-layer KV sharing. |
| `enable_moe_block` | (absent) | false | — | 31B not MoE; same as 12B. |
| `use_bidirectional_attention` | (absent) | "vision" | — | Vision-only flag, not text-path. |
| `vocab_size_per_layer_input` | (absent) | 262144 | — | Vision/multimodal field. |

**Highlights — what's structurally different (text path only):**

1. **NQ doubles, NKV doubles** → matmul shapes change, GQA group same (NQ/NKV=2 in both).
2. **`num_global_key_value_heads` 1 → 4**: this is the only field that
   *removes complexity*. 12B's NKV_GLOBAL=1 forced a two-call paged SDPA
   per global-attention layer (because `paged_update_cache.input.dim(1)` must
   match `page_table.dim(0)` and NKV_PER_CHIP=1 on (1,4) breaks the contract
   — `[[feedback-paged-update-cache-nkv-per-chip]]`). 31B with NKV_GLOBAL=4
   → NKV_PER_CHIP_GLOBAL=1 still has NKV_PER_CHIP=1 BUT each chip holds a
   DIFFERENT KV head, not the same one replicated — so paged_update_cache
   on a single chip is naturally B=1, NKV=1 with its own dedicated head,
   no aliasing needed. **Net: 31B may not need the two-call workaround**;
   verify at bringup time.
3. Everything else is scale-only: hidden_size +40%, intermediate +40%,
   layers +25%. Same RoPE, same softcap, same v_norm/q_norm/k_norm pattern
   (`[[feedback-gemma4-v-norm]]`), same per-layer `layer_scalar`
   (`[[feedback-gemma4-layer-scalar]]`). **No new layer types, no new
   normalizations, no new attention flavor.**

The architecture *family* is identical. The user's "mostly config delta"
assertion is **correct for the model code**. Risks are downstream (memory,
trace shapes, ladder bringup time).

## 3. Memory budget on (1,4) P150 mesh

P150: 32 GB/chip nominal, ~30 GB usable after L1 + runtime + trace
(per `[[feedback-p150-memory-bandwidth-measured]]`).

### Weights

```
31B × 2 bytes (bf16)       = 62 GB total
/ 4 chips                  = 15.5 GB/chip          (weight residency)
```

For comparison, 12B is `12 × 2 / 4 = 6 GB/chip`.

### KV cache

8K context, NQ=32, NKV=16 sliding / 4 global, head_dim 256 sliding / 512 global.

Per layer (sliding): `2 × 8192 × NKV_PER_CHIP_SLIDING × head_dim × 2 bytes`
  `= 2 × 8192 × 4 × 256 × 2 = 32 MB/chip/sliding_layer`

Per layer (global): `2 × 8192 × NKV_PER_CHIP_GLOBAL × global_head_dim × 2`
  `= 2 × 8192 × 1 × 512 × 2 = 16 MB/chip/global_layer`

Layers: 50 sliding + 10 global out of 60.

```
KV (8K)  = 50 × 32 + 10 × 16  = 1660 MB/chip ≈ 1.6 GB/chip
KV (32K) = 4× of above        ≈ 6.6 GB/chip
```

### Trace memory

Per `[[feedback-ttnn-trace-region-size]]`: default 50 MB suffices for
non-chunked workloads. 31B with 60 layers vs 12B's 48 may need ~60 MB;
budget 100 MB to be safe.

### Activations / scratch

Bounded by B × HIDDEN × seq budget at trace capture. For B=1 decode:
~50 MB/chip working set.

### Total

```
weights        : 15.5 GB/chip
KV (8K, B=1)   :  1.6 GB/chip   (production target)
KV (32K, B=1)  :  6.6 GB/chip   (long-context option)
trace          :  0.1 GB/chip
activations    :  0.05 GB/chip
─────────────────────────────
working (8K)   : 17.3 GB/chip  ← FITS, ~12.7 GB headroom
working (32K)  : 22.3 GB/chip  ← FITS, ~7.7 GB headroom
working (B=2,8K): 18.9 GB/chip ← FITS
```

**FITS on (1,4) P150**. Headroom is *narrower* than 12B (which has ~22 GB
headroom at 8K B=1), but comfortable at the target's standard 8K context
budget. Long-context (128K-256K) would need the prefix-cache-on-device
analysis from `[[project-prefix-caching-design]]` to plan.

### Per-chip memory equivalence to existing models

| Model | Weight/chip | At-rest working budget |
|---|---|---|
| 27B (production) | ~6.75 GB | ~12 GB |
| Gemma 4 12B (production) | ~6 GB | ~10 GB |
| **Gemma 4 31B (this plan)** | **~15.5 GB** | **~17–22 GB** |
| 35B (production) | ~17.5 GB | ~19–20 GB |

**31B's per-chip weight footprint is ~88% of 35B's** — but 35B is MoE
(A3B active) so most of its working bandwidth pressure is the per-step
3B active. 31B is dense (~31B active per step), so DRAM BW pressure per
step is ~10× higher than 35B's. **Per-step DRAM BW is the real perf
question**, not memory residency. See §6.

## 4. Trace compatibility with current Gemma 4 12B pipeline

The existing pipeline (`server_gemma4_unified_ttnn.py` + `_cb.py`) is built
around 48 layers + the 12B shape constants. To switch to 31B, the trace path
needs:

1. **Re-capture with new shapes** — trivial; trace capture isn't model-aware
   beyond the tensor shapes it records. New B=1 capture happens automatically
   when the bootstrap runs with the 31B config.
2. **Two-phase warmup still applies** (`[[ttnn-multi-trace-two-phase-warmup]]`).
3. **trace_region_size** likely needs bumping from 50 MB to 100 MB. One-line
   config bump.
4. **NKV_GLOBAL=4 means the two-call workaround may be unnecessary** — at
   12B v0.3 we shipped the two-call paged-SDPA for global layers to handle
   NKV_PER_CHIP_GLOBAL=1 + B>1 contract violation. With 31B's NKV_GLOBAL=4
   on (1,4) = NKV_PER_CHIP_GLOBAL=1, the *contract* is technically the same,
   but the underlying KV slot is per-chip-distinct rather than replicated,
   so the existing paged_update_cache call should work in one shot.
   **VERIFY at bringup**: write an isolation probe forking
   `experiments/cb/isolate/paged_sdpa.py` for the 31B global shape; if
   single-call works, drop the two-call workaround → faster + simpler.

The 12B trace pipeline forks to 31B without architectural changes. Just
shape constants + weight loading.

## 5. Scope estimate — what to touch

### NEW files

| File | Source / pattern forked | LOC est. |
|---|---|---|
| **NEW** `experiments/serve/server_gemma4_31b_ttnn.py` | fork `server_gemma4_unified_ttnn.py` | ~2000 (≈ identical structure) |
| **NEW** `experiments/serve/server_gemma4_31b_cb.py` | fork `server_gemma4_unified_cb.py` | ~500 |
| **NEW** `research/gemma4_31b_bringup_plan.md` | fork `research/gemma4_12b_bringup_plan.md` (971 lines) | ~600 (lighter, since arch is known) |

### Edited files

| File | Touch |
|---|---|
| `experiments/serve/cb_api.py:50` | Add `BACKENDS["gemma4_31b"] = "server_gemma4_31b_cb"` |
| `experiments/serve/cb_scheduler.py:50` | Add `_BACKEND_MODULES["gemma4_31b"] = ("server_gemma4_31b_ttnn", "server_gemma4_31b_cb")` |
| `scripts/run_harness_tmux.sh` | Add `gm4_31b` case (forks `gm4` case) |
| `HANDOFF.md` | One-line entry pointing at 31B work-in-flight |

### Inside `server_gemma4_31b_ttnn.py`, the things that change vs the 12B copy

```python
# Top-of-file constants — 6 changes (the entire config delta lives here):
MODEL_ID = "google/gemma-4-31B"      # was "google/gemma-4-12B"
HIDDEN = 5376                         # was 3840
NUM_LAYERS = 60                       # was 48
NUM_Q_HEADS = 32                      # was 16
NUM_KV_HEADS_SLIDING = 8              # was 8 (UNCHANGED!) — wait, NKV_SLIDING is 16 per 31B config
NUM_KV_HEADS_SLIDING = 16             # was 8  (corrected)
NUM_KV_HEADS_GLOBAL = 4               # was 1
INTERMEDIATE = 21504                  # was 15360
# All derived per-chip values recompute automatically.
```

Plus a one-line cache-dir name change (`models--google--gemma-4-31B` vs
`models--google--gemma-4-12B`), plus a possible **simplification** of the
global-attention paged decode (drop the two-call workaround if the probe
confirms NKV_PER_CHIP_GLOBAL=1 + per-chip-distinct path works).

The `layer_scalar` field is per-layer learned scalar; 31B has 60 of them
(vs 12B's 48). Load loop is the same; safetensors key pattern identical.

The `v_norm` (no-scale RMSNorm with all-ones tensor — `[[feedback-gemma4-v-norm]]`)
applies identically.

### Bringup ladder (per `[[reference-model-bringup-recipe]]`)

| Phase | Time at 12B | Time at 31B (est.) | Notes |
|---|---|---|---|
| v0.0 (HF numpy oracle) | 4 hours | **6 hours** | Larger weights → longer numpy fp32 build |
| v0.1.0 (embed + final_norm gate) | 1 hour | 1 hour | Same |
| v0.1.1 (Q/K/V + q_norm/k_norm/v_norm) | 2 hours | 2 hours | Same |
| v0.1.2 (full sliding attention block) | 2 hours | 2 hours | Same |
| v0.1.3 (global attention block) | 3 hours | **2 hours** | *Faster* if NKV_GLOBAL=4 removes the two-call workaround |
| v0.2 (full forward, 1 token, oracle match) | 6 hours | **9 hours** | More layers to ladder; per-layer bootstrap longer |
| v0.3.0 (KV cache + paged SDPA) | 4 hours | 4 hours | Same |
| v0.3.1 (multi-step decode chain match) | 4 hours | 4 hours | Same |
| v0.4 (trace capture, two-phase warmup) | 4 hours | 4 hours | Same |
| v0.5 (CB + HTTP smoke) | 4 hours | 4 hours | Same |
| **Total** | ~36 hours | **~42 hours = ~5.5 days** | |

**Estimate: 5–6 days of focused build for v0.5 production-ready 31B HTTP path.**

That's slightly longer than the 12B's ~36 hours because (a) larger weights
mean longer bootstrap iterations (~3× per cycle), and (b) more layers in the
per-layer ladder validation. Architecture work itself is essentially trivial.

## 6. Honest perf expectation

12B traced: **47.5 ms/tok** (post-P1 vocab-shard), at 9.5 GB/chip weight
working set roughly DRAM-BW-bound.

31B working-set scaling: weights / chip 6 GB → 15.5 GB (2.6×); per-step
working DRAM BW scales similarly. **Naive scaling**: 47.5 × 2.6 ≈ 123 ms/tok
(8.1 tok/s) **on the same trace pipeline at the same perf maturity.**

This assumes perfect linear DRAM-bound scaling (32 GB/chip total BW is
the same across model sizes). Real numbers will likely be 110–140 ms/tok
depending on whether NKV_GLOBAL=4 saves a meaningful per-step ms (probably
not — global layers are 10/60 = 17% of total).

For comparison: 35B (MoE A3B) currently sits at 81 ms/tok with active
3B / step. **31B (dense) will be 1.4–1.7× slower per token than 35B**
on the same hardware, despite being 4 GB smaller — because dense 31B does
~10× the per-step matmul work that MoE 35B's A3B does. This is the
intrinsic cost of "go dense instead of sparse" and isn't a TT-specific
deficiency.

## 7. Risks

1. **NKV_GLOBAL=4 two-call workaround simplification turns out to NOT
   simplify**. Worst case: 31B inherits the same two-call pattern as 12B.
   No regression, just no win. Probe at v0.1.3 to decide.

2. **Memory budget at long context**. 8K B=1 fits with 12.7 GB headroom.
   B=2 at 8K fits. B=2 at 32K fits with ~3 GB headroom. B>2 + long context
   may not fit; the CB engine's TT_CB_SLOTS env will need a lower default
   for the 31B backend (suggest `TT_CB_SLOTS=2` initially; raise after
   measurement).

3. **`gemma4` vs `gemma4_unified` HF class divergence**. The text path is
   identical between the two — they share `Gemma4DecoderLayer` etc. — but
   our numpy oracle (`[[feedback-numpy-reference]]`) builds the HF
   reference from `modeling_gemma4_unified.py`. For 31B we may need to
   read from `modeling_gemma4.py` instead, which could have minor offsets
   (e.g. different residual norm placement). Verify at v0.0.

4. **`v_norm` `with_scale=False` identity trick**. We pre-allocate an
   all-ones HEAD_DIM tensor and pass it as the weight to `ttnn.rms_norm`.
   12B uses HEAD_DIM=256. 31B also uses HEAD_DIM=256. Should reuse cleanly.

5. **`layer_scalar` magnitude**. At 12B, `layer_scalar` ranges from
   0.054 (L0, L47) to 0.82 (L24). For 31B with 60 layers, the curve may
   peak higher. Missing `layer_scalar` is a magnitude bug; `mad` diverges
   but cosine still ~1 (`[[feedback-cos-not-enough-also-check-mad]]`). Use
   the same load loop and same mid-layer mad assertion in the v0.1 ladder.

6. **Tokenizer drift**. Vocab 262144 same as 12B; the tokenizer.json files
   should be byte-identical (same BPE merges, same special tokens). Verify
   with `cmp` of both HF cache dirs at bringup.

7. **Bringup wall-clock**. Per-iter bootstrap at 31B will be ~3× slower
   than 12B due to larger weights to upload to mesh (~62 GB total). Use
   the dev-harness pattern (`[[reference-gm4-dev-harness]]`) to amortize
   the cost — bootstrap once, drop trigger files for ~10s iterations.

## 8. Decision frame

- **Architecture**: identical to 12B family. No new layer types, no new
  norms, no new attention flavor. User's "mostly config delta" assertion is
  correct.
- **Memory**: fits on (1,4) P150 at 8K B=1 with 12.7 GB headroom. Tighter
  than 12B but workable.
- **Dev time**: ~5–6 days for v0.5 HTTP-ready. About 1 day more than 12B
  was, due to per-iter wall-clock penalty (3× bootstrap).
- **Perf ceiling**: ~110–140 ms/tok traced at v0.5 maturity. About 2.5×
  slower than 12B per token (dense scaling), and 1.4–1.7× slower than 35B
  per token (dense vs MoE).

**Recommendation**: pursue 31B bringup AFTER the current Gemma 4 12B perf
work plateaus (target ≤40 ms/tok per the perf ladder), so we don't fork a
moving baseline. Once 12B perf is stable, the fork to 31B is mechanical
and the 5-day spend is justified for the strictly stronger model (Gemma 4
benchmarks: 31B IT is ~10–15 points stronger than 12B IT on MMLU/GPQA per
Google's launch blog).

## 9. Honest limits

- The text-config diff was extracted from a single `WebFetch` of each
  `config.json`. Some fields were summarized rather than verbatim; before
  bringup, do a strict `diff` of both raw JSON files and audit any field
  this doc didn't enumerate (especially anything new in transformers
  5.10.0.dev0 vs 5.5.0.dev0).
- The `layer_scalar` and `v_norm` presence in 31B is *inferred* from
  arch-family inheritance; not verified by reading the 31B safetensors
  key list. v0.0 should `safe_open` a layer's keys and confirm
  `layer_scalar` + `v_norm.weight` patterns exist.
- The memory budget at 32K context (6.6 GB KV/chip) is back-of-envelope;
  in practice page-table indirection and SDPA scratch eat some headroom.
  Run a real `MAX_KV=32768` smoke before claiming long-context support.
- Perf scaling estimate (2.6× weight working set → 2.6× per-step) assumes
  perfect DRAM-BW-bound scaling. In reality, compute-bound segments don't
  scale that way; the 12B's lm_head was lifted from compute-bound to BW-
  bound by P1 sharding. The 31B's lm_head will need the same treatment
  on day one (`server_gemma4_31b_ttnn.py` should ship with the vocab-shard
  pattern already integrated, not as a follow-on).

## 10. Related memory / files

- `experiments/serve/server_gemma4_unified_ttnn.py` — 2017-line target to fork
- `experiments/serve/server_gemma4_unified_cb.py` — 528-line CB shim to fork
- `research/gemma4_12b_bringup_plan.md` — 971-line bringup plan (fork to ~600 for 31B)
- `research/model_bringup_recipe.md` — staged ladder recipe
- `[[reference-model-bringup-recipe]]`
- `[[reference-gm4-dev-harness]]` — fast-iteration harness
- `[[feedback-gemma4-v-norm]]` — v_norm `with_scale=False` identity trick
- `[[feedback-gemma4-layer-scalar]]` — per-layer learned scalar buffer
- `[[feedback-gemma4-sdpa-scale-1]]` — SDPA `scale=1.0` for Gemma 4
- `[[feedback-paged-update-cache-nkv-per-chip]]` — NKV_PER_CHIP=1 contract
  (relevant for 31B NKV_GLOBAL=4 evaluation)
- `[[reference-gemma4-two-call-paged-decode]]` — workaround we may not need
- `[[feedback-p150-memory-bandwidth-measured]]` — 32 GB/chip budget source
- `[[feedback-p22-gm4-vocab-shard-result]]` — P1 vocab-shard pattern to inherit
- `[[ttnn-multi-trace-two-phase-warmup]]` — trace capture protocol
- Google launch blog: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- HF model card: https://huggingface.co/google/gemma-4-31B
