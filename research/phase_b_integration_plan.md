# Phase B — Single-Model Integration (Qwen3.6-35B-A3B on 2-chip TP)

This is where A0–A7 components get wired into a working model. ~15-25 hrs of focused work.

## Pre-conditions (all met after Phase A)

- ✅ A0: MoE regression diagnosed; we know the per-op costs in 0.69
- ✅ A1: every shape pinned (`qwen36_arch_notes.md`)
- ✅ A2: every equation extracted (`qwen36_modeling_excerpts.md`)
- ✅ A3: DeltaNet decode-step isolated, cosine 0.999995
- ✅ A4: Gated Attention isolated, cosine 0.999943
- ✅ A5: MoE block isolated (256 experts), cosine 0.999771
- ⏳ A6: chunked-serial scan v1 (decode is OK; prefill works but slow)
- ⏳ A7: 2-chip mesh + collectives + MoE all-to-all validated

## Phase B substeps

### B1 — Weight loading (~3 hrs)

`experiments/90_qwen36_port.py` shell.

- Use `huggingface_hub.hf_hub_download` for 26 safetensor shards (~72 GB on disk in bf16; ~36 GB in bf8)
- Walk `model.safetensors.index.json` to map parameter names to shards
- Per-layer load: stream weights, quantize to bf8 on upload, place on the right chip per TP plan
- Skip vision tower params entirely (we filter by name prefix)

Sharding strategy across 2 chips (per A7 + the architecture):

| Component | Strategy |
|---|---|
| Embeddings (in + out) | Replicated (read-only, small footprint) |
| DeltaNet layers — projections | Hidden-dim sharded (each chip owns half the cols), all-reduce after |
| DeltaNet recurrent state H | Head-sharded (16 v-heads per chip out of 32) |
| Gated Attention — Q/K/V/O | Hidden-dim sharded, all-reduce after |
| MoE — 256 experts | Expert-parallel: each chip owns 128 experts; ttnn.all_to_all_dispatch routes tokens |
| Shared expert | Replicated on both chips |
| RMSNorm | Replicated |

This is straightforward 2-way TP for everything except MoE, which uses expert parallelism (the right pattern for sparse MoE).

### B2 — Decode-path forward (~5 hrs)

Wire together:
- Embed: lookup from replicated table, scatter result to both chips
- For each of 40 layers:
  - RMSNorm (replicated)
  - DeltaNet OR Gated Attention (per layer_types pattern from A1)
  - all-reduce (TP synchronization)
  - RMSNorm
  - MoE: route via all_to_all_dispatch, compute on local experts, all_to_all_combine
  - Residual
- Final RMSNorm + output projection
- Argmax for greedy token

### B3 — Correctness gate (~3 hrs)

Per `feedback_correctness_first.md` and the A5 finding (don't gate on per-layer cosine through 40 MoE layers):

1. **Per-layer cosine ≥ 0.99 vs numpy fp32 reference for layers 0, 4, 8, 12, … 36** (every 4 layers — catches drift)
2. **8/8 greedy token match** at the model output

Numpy reference: full forward pass in numpy on the host (slow but possible at fp32 — 35B params × 4 bytes = 140 GB doesn't fit RAM, so we'd reload weights per layer). Actually, for the correctness gate, we just need the first 2 layers + final lm_head — that fits easily. The 8/8 token match validates the full chain.

### B4 — First real generation (~1 hr)

```python
prompt = "Write a Python function that takes a list and returns the most frequent element."
greedy_decode(prompt, max_tokens=200)
```

Verify coherent code generation. Expected: working Python function with explanation.

### B5 — Trace capture for performance (~3-5 hrs)

Static parts (attention, DeltaNet, shared expert) can be traced. MoE routing is data-dependent — keep eager with device-side topk + host readback (Qwen1.5-MoE pattern).

Per-layer trace:
- RMSNorm + DeltaNet/Attn + all-reduce: ONE trace per layer type (DeltaNet trace × 30, Attn trace × 10)
- MoE remains eager (256 experts × 3 weights = 768 MB of weights, eager dispatch via host)

### B6 — Performance measurement (~1 hr)

Targets:
| Metric | Target | Reach |
|---|---|---|
| Decode tok/s, single chip eager | ≥ 5 | 7-10 |
| Decode tok/s, 2-chip TP eager | ≥ 8 | 12-15 |
| Decode tok/s, 2-chip TP traced (where possible) | ≥ 15 | 20-25 |
| Per-token memory floor (bf8, 2 chips parallel) | — | ~2 ms (theoretical) |

The "reach" numbers assume the binary mul / topk regressions we found in A0 don't compound further.

## Stopping rules

- If B3 cosine < 0.99 after 2 hrs of debugging: stop, document the specific failing layer, ablate.
- If B4 generation is incoherent (loops, garbage): drop to single-chip first to isolate TP from arch issues.
- If B6 decode is < 5 tok/s with no obvious fix: stop, document, decide whether to push for fusion or move to D (Coder-Next) anyway and accept slow decode.

## Risk register

1. **TT-NN multi-chip op coverage**: not every ttnn op may be implemented for sharded inputs. May hit a wall on a specific op. Mitigation: have single-chip fallback path for each layer.
2. **paged_update_cache sharded-input contract**: the production port has this dance; need to reproduce exactly for our cache shapes (head_dim=256, 2 KV heads).
3. **Partial RoPE on device**: A4 used host-side RoPE as a workaround. For Phase B we need device-side. Options: (a) ttnn.experimental.rotary_embedding_llama (b) custom small kernel (c) reshape head_dim into [4 × 64] to make slicing tile-aligned.
4. **fp32 DeltaNet state with bf8 weights**: A3 verified mixed precision works at the recurrence level, but at scale across 30 layers we may see drift. Monitor per-layer.
5. **Cold JIT cache on multi-chip**: each chip JITs kernels independently. First-decode of a 2-chip model probably builds ~200 kernels = ~100 sec. One-time cost.

## After Phase B passes — Phase C and D

Phase C: extend to 4-chip TP, move to bf16 (head room from 4× memory), test 32K context. ~10-15 hrs.

Phase D: scale model config to Qwen3-Coder-Next (80B/3B-active, 12-pattern × 4-layer = 48 layers, 512 experts/10 active). Mostly a config swap; weights re-load. ~5-10 hrs.

## Files Phase B produces

```
experiments/90_qwen36_port.py             — main port script
experiments/90b_qwen36_correctness.py     — B3 cosine + token-match harness
experiments/90c_qwen36_decode_loop.py     — B4 chat-style decoder
research/phase_b_results.md               — what worked, what didn't
research/qwen36_first_generations.md      — actual text outputs at each milestone
```

## Pre-condition: A6 + A7 must land

Without A6 v1 we can't even prefill the model (or we serial-loop on host, slow). Without A7 we can't fit the model.

Suggested order of remaining A work:
1. **A6 v1** — run as soon as qb1 stabilizes (script ready at `experiments/85_deltanet_scan_v1.py`)
2. **A7** — open 2-chip mesh, validate collectives
3. **Phase B starts**

Phase B is conditional on all of those landing.
