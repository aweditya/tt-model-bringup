# Branch C'0.5 — MAX_POS scale-up to 32k

**Date**: 2026-05-12.
**Scope**: lift KV-cache horizon from 256 → 32 768 in production Qwen3.6-27B path;
build a residual-dtype (fp32 vs bf16) correctness gate at long context.
**Why now**: every later C' phase (bf16 ablation, trace, chunked prefill) is silently
un-gated without long-context numbers. Performance is **not** a goal here — only
correctness + functional 32k operation.
**Out of A6-v1 integration scope** (separate agent).

---

## 1. `MAX_POS` dependencies

Kernel `gated_attn_step_ondevice` (91f L227-305) **does not reference `MAX_POS`** — it
takes pre-allocated K/V caches and reads `cur_pos` from a device tensor. The hot path
is already MAX_POS-agnostic. We only edit allocation sites + module constants.

**Production-path edits**:

| File | Line | Today | Edit to |
|---|---:|---:|---|
| `experiments/91l_fp32_residual_generate.py` | 46 | `MAX_POS = 256` | `32768` |
| `experiments/utils/perf_baseline.py` | 176 | `256` | leave (perf harness) |
| `experiments/demo_qwen36_27b.py` | 56 | `256` | leave (short prompts) |
| `experiments/91f_qwen36_27b_full_ondevice.py` | 32 | `128` | leave (cosine test) |

Allocations that follow from `MAX_POS`:
- `91l` L188-194 — KV cache alloc `np.zeros((1, 4, MAX_POS, 256))` → `ttnn.from_torch`
  bf16 TILE. **Auto-scales** from the constant edit.

**Skip** (deprecated / isolation / probes): 91g L45 (=128), 91e L487 (=128), 91h L52
(=256), 91j L56, 91r L53 (correctness harness, keep at 256), 91i (already
arg-parameterized via `--max-pos`), `utils/scatter_probe.py`.

The C'2 agent edits the same `91l`/`demo`/`perf_baseline` set — coordinate via merge,
not parallel writes.

---

## 2. KV cache memory math

Per attn layer, both K+V, bf16:
`mem_per_layer = 2 × 1 × 4 × MAX_POS × 256 × 2 bytes = 4096 × MAX_POS B`.
Total across 16 attn layers: `65 536 × MAX_POS B = 64 KiB × MAX_POS`.

| MAX_POS | KV total (bf16) | KV total (bf8) | Headroom after 27 GB weights (32 GB chip) |
|---:|---:|---:|---|
| 256 | 16 MB | 8 MB | ~5 GB ✓ |
| 4 096 | 256 MB | 128 MB | ~4.7 GB ✓ |
| 8 192 | 512 MB | 256 MB | ~4.5 GB ✓ |
| **32 768** | **2.0 GB** | 1.0 GB | **~3 GB ✓** (ship target) |
| 65 536 | 4.0 GB | 2.0 GB | ~1 GB tight |
| 131 072 | 8.0 GB | 4.0 GB | **bf16 INFEASIBLE**; bf8 ~1 GB |

Add DeltaNet state ~150 MB (MAX_POS-independent) + ~1-2 GB scratch.

**Ship**: 32k @ bf16. **Defer**: 128k (needs bf8 KV path + scatter-dtype work).

---

## 3. RoPE table sizing

Current `rope_tables_for_pos` (91l L200-206) is host fp32 numpy, fresh per step:
```
half_rot = 32  (= 64 rotary / 2)
freqs = 1.0 / (10_000_000 ** (np.arange(32)/32))
angles = pos * freqs        # scalar broadcast
cos = np.cos(angles).concatenate(...)
```

Numerics check at extremes:
- `pos=32 768`, `freqs[31]≈3.3e-3` → `angles[31]≈3.3e-3` rad. Safe.
- `pos=131 072`, `angles[0]=pos × 1.0 = 131 072 rad`. fp32 `np.cos`/`np.sin` lose
  precision near `|x|~2^23≈8.4e6`. **~60× margin even at 128k.**
- `rope_theta=10⁷` (Qwen3.6) is already long-context-friendly; native context is
  **262K**; no YARN scaling needed below that.

**Verdict**: no code change. Document the fact.

---

## 4. SDPA-decode at large `cur_pos`

Call: `ttnn.transformer.scaled_dot_product_attention_decode` at
`91f_qwen36_27b_full_ondevice.py:292`.
Q `[1,1,24,256]`, KV `[1,4,MAX_POS,256]`, `cur_pos_tensor [1] int32`, `compute_kernel_config=hifi4`.

Phase A4 validated at `cur_pos=32, KV_LEN=128`. Behavior at `cur_pos≈32 000` is
**unknown**. Risks:
- Tile-alignment along seq dim: `32768/32=1024` tiles → aligned ✓.
- Internal seq-chunking caps inside the kernel.
- L1 scratch for Q·Kᵀ scores `[24 × MAX_POS]` → 3 MB bf16 at 32k. Distributed across
  grid; should fit but unverified.

**Probe before any edit**: reuse `experiments/91i_shape_preflight.py --max-pos 32768`
(already arg-parameterized). Extend it to call SDPA at
`cur_pos ∈ {0, 4096, 16384, 32767}`. **Gate**: all pass.
Fallback: ship at highest passing MAX_POS (8k or 16k); file ttnn issue.

---

## 5. `experiments/95_long_context_test.py`

(Filename per user request. `95_moe_partial_trace.py` already exists — if collision is
unacceptable, rename to `91u_long_context_test.py`. Same content.)

**Purpose**: detect residual-dtype drift over long generations. **Variable**:
residual stream dtype. **Fixed**: weights, prompt, seed, sampling (greedy argmax).

**Prompt**: deterministic synthetic 32k document. Default: a 32-token sentence
repeated 1024 times. Tokenizer-stable; exercises positions ≥ 256.

**Architecture**: load model **once** (~10 min); for each config in `["fp32",
"bf16"]`, reset KV caches + DeltaNet states, prefill prompt, decode 200 tokens
(greedy), record per-position diagnostics.

**Per-step record per config** → `~/tt-xla/.cache/c05_long_ctx/{fp32,bf16}.json`:
```
{ "config_id", "prompt_len",
  "decoded_token_ids":     [int × 200],
  "per_pos_top1_logit":    [float × 200],
  "per_pos_top2_logit":    [float × 200],
  "per_pos_top5_ids":      [[int × 5] × 200] }
```

**Implementation notes**:
- Import `_91f.{deltanet_step_ondevice, gated_attn_step_ondevice, mlp_step_ondevice}`
  and `_91l.load_embed_lm_head_weights`. Cannot import 91l's `forward_one_token` —
  it's a closure; rebuild it.
- Wrap the dtype switches in 91l L154-165 (proj/norm/scalar), L173-185 (states),
  L188-194 (KV), L201-206 (RoPE), L210-211 (embed) into one
  `dtype_for_config(name)` selector.
- KV cache stays bf16 across both configs (single-write per slot doesn't compound).
  Only the **residual stream + scalar weights + RoPE tables** flip dtype.

**Comparison report** → `~/tt-xla/.cache/c05_long_ctx/comparison.md`:
- Top-1 agreement: `sum(fp32.top1 == bf16.top1) / 200`.
- Top-5 overlap: `mean(jaccard(fp32.top5[i], bf16.top5[i]))`.
- Logit margin curve: `bf16_margin[i] - fp32_margin[i]`.
- First-divergence position.
- **Pass**: top-1 agreement ≥ 0.90 OR first-divergence ≥ 50.

**Args**: `--prompt-len {1024, 8192, 32768}` default 1024 (C'0.5 ships small-N);
`--configs fp32,bf16` default both; `--tokens 200`.

**Runtime**:
- Small-N (1024 prefill, 200 decode) @ current 562ms prefill, 307ms decode:
  ~11 min/config, **~22 min total**. **C'0.5 exit deliverable.**
- Full 32k @ same: ~5h prefill/config → **infeasible until C'5a**. Re-run as the
  C'2 gate after A6-v1 lands.

---

## 6. Implementation order (sequential gates)

1. **SDPA probe** — run `91i_shape_preflight.py --max-pos 32768` with SDPA calls at
   four cur_pos values. **Gate**: all pass; else ship at highest passing value.
2. **Edit constants** — `91l` L46 → `32768`. **Gate**: `91r_per_layer_diff.py` still
   passes (DeltaNet ≥ 0.9997, full_attn ≥ 0.9998); 91r untouched, used as regression
   sentinel.
3. **Demo sanity** — `demo_qwen36_27b.py` produces " Paris" (still at 256 — guards
   the unchanged path); `91l --tokens 60` produces coherent text @ ≥ 3 tok/s
   (guards the 32k alloc).
4. **Write `95_long_context_test.py`** per §5.
5. **Run small-N** — `--prompt-len 1024 --configs fp32,bf16`. **Gate**: top-1
   agreement ≥ 0.90 OR first-divergence ≥ 50.
6. **Commit + doc** — single commit `C'0.5: MAX_POS → 32k + long-context gate`;
   append small-N numbers to `research/performance_roadmap.md`.
7. **Deferred** (post-C'5a): re-run with `--prompt-len 32768`. Real gate for C'2,
   C'4, and the daily-driver promise.

---

## 7. Risks (top five)

1. **SDPA rejects KV_LEN=32k.** Medium / high impact. Mitigated by §6.1 probe;
   fallback = highest passing MAX_POS.
2. **L1 scratch OOM inside SDPA at long KV.** Low-medium / high. Caught by same
   probe.
3. **`ttnn.scatter` index alloc cost grows with MAX_POS.** Index shape is
   `(1,4,1,256)` — **MAX_POS-independent**. Not a risk; documenting.
4. **bf16 residual fails small-N gate.** Medium / medium. That *is* a signal —
   confirms fp32 needed, informs C'2.
5. **Full 32k unrunnable without C'5a.** Certain / no impact on C'0.5 (small-N
   ships now); only delays the full gate.

---

## 8. Effort estimate

**Half-day** for the engineer who wrote 91l.
- §6.1 probe: 30 min.
- §6.2-3 edit + sanity: 30 min.
- §6.4 write 95-script (copy 91l + comparison report): 2 hr.
- §6.5 run + inspect: 30 min (22 min wall).
- §6.6 commit + doc: 15 min.

Total ≈ 3.5 h productive, ~4-5 h wall. Second half-day after C'5a re-runs at 32k.

---

## Out of scope

bf8 KV cache; 128k context; YARN; A6-v1 integration; perf instrumentation at long
context; trace capture interaction.
