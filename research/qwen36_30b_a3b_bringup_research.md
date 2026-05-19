# Qwen3.6-"30B"-A3B Bringup Research (RESOLVED: model is Qwen3.6-**35B**-A3B)

## 0. Naming clarification (READ FIRST)

There is **no Qwen3.6-30B-A3B**. The user-supplied name conflates two distinct models:

| Name in the wild | What it actually is |
|---|---|
| `Qwen3-30B-A3B` | Qwen **3** (April 2025), MoE, 30B/3B, dense Q/K/V attention, no DeltaNet |
| `Qwen3.6-35B-A3B` | Qwen **3.6** (April 2026), MoE, **35B/3B**, DeltaNet+Attention hybrid (same family as our 27B) |

Friend's `qwen36_moe.py` docstring confirms: "Qwen3.6-**35B**-A3B differences from Gemma4 are called out inline."
HF model card confirms 35B total / 3B active. Everything below is for **Qwen3.6-35B-A3B**, which is the only A3B sibling of our 27B target.

## 1. Executive summary

Qwen3.6-35B-A3B is a 35B-parameter MoE LLM with ~3B active per token, top-8-of-256 routing plus one sigmoid-gated shared expert. The backbone is the **same DeltaNet+Attention hybrid** as our 27B build (10 blocks of 3 GatedDeltaNet + 1 GatedAttention, MoE replacing the dense MLP at every position), so all our DeltaNet kernels, paged SDPA, RoPE, LM head, and TP plumbing port directly; the new work is purely the MoE block. With bf4 routed experts and bf8 shared expert (friend's defaults) the model fits ~6.0 GB / chip on a (1, 4) P150 mesh with comfortable headroom for paged KV.

## 2. Architecture comparison

| Property | Qwen3.6-27B (shipped) | Qwen3.6-35B-A3B |
|---|---|---|
| Total params | 27B | 35B |
| Active / token | 27B (dense) | ~3.1B (MoE) |
| Layers | 64 | **40** |
| Hidden (`dim`) | 5120 | **2048** |
| Layer pattern | 16 × (3 DN + 1 Attn) | **10 × (3 DN + 1 Attn)** |
| DN: V heads / QK heads / head_dim | 32 / 16 / 128 | 32 / 16 / 128 (same) |
| Attn: Q heads / KV heads / head_dim | 24 / 4 / 256 | **16 / 2 / 256** |
| Partial RoPE factor | 0.25 (rotary_dim = 64) | 0.25 (same) |
| RoPE theta | 10,000,000 | 10,000,000 (same) |
| RoPE scaling | default (no YaRN active up to 262k) | default; YaRN factor=4.0 → 1.01M ctx |
| Native context | 262,144 | 262,144 |
| Vocab (padded) | 248,320 | 248,320 (same tokenizer) |
| MLP | dense SwiGLU, intermediate=17408 | **MoE**: 256 experts × intermediate=512 + 1 shared expert (intermediate=512) |
| Routing | n/a | softmax → top-8 → renormalize by sum → scatter |
| MTP head | yes (probed 57.9% top-1) | yes (mentioned in card) |
| Format | bf16 | bf16 |

## 3. Memory budget on (1, 4) P150 mesh (12 GB DRAM / chip)

Per-chip breakdown at friend's production dtypes (bf4 routed experts, bf8 shared + attention, bf16 router/norms/lm_head):

| Component | Per-layer | × layers | bf4/bf8 bytes/param | Per-chip @ TP=4 |
|---|---|---|---|---|
| Router (`gate`) | 2048 × 256 = 524k | × 40 | bf16 (replicated) | ~42 MB |
| Routed experts (gate_up + down) | 3 × 256 × 2048 × 512 = 805M | × 40 | bf4 (0.5 B/param) | (805M × 40 × 0.5) / 4 = **4.0 GB** |
| Shared expert (gate/up/down) | 3 × 2048 × 512 = 3.15M | × 40 | bf8 (1 B/param) | (126M × 1) / 4 = ~32 MB |
| Attention (Q/K/V/O across DN+Attn hybrid mix) | varies | × 40 | bf8 | ~1.0 GB (estimate from 27B ratio) |
| DeltaNet projections + conv1d | varies | × 30 DN layers | bf8 | ~0.6 GB |
| Embed + LM head (248320 × 2048) | 508M params | × 1 (shared if tied; vocab-sharded) | bf16 | (508M × 2) / 4 = 254 MB |
| **Total weights / chip** | | | | **~6.0 GB** |
| Paged KV cache (attn layers only = 10) | 2 KV heads × 256 dim × 10 layers × MAX_POS | | bf16 | ~10 MB @ MAX_POS=1024, 320 MB @ 32k |
| Activations + trace + L1 scratch | | | | ~1-1.5 GB (matches 27B) |
| **Total / chip** | | | | **~7.5 GB at 1k ctx, ~8 GB at 32k** |

**Fits comfortably** in 12 GB. Headroom for >32k context. If we ship bf8 routed experts (instead of bf4), expert footprint doubles to 8 GB / chip — still fits with little margin. The 35B is **smaller per chip than our 27B** in active weight bytes because we sparsify 256:8 (32× expert sparsity) and the routed experts are bf4.

## 4. Infrastructure reuse map

| Component (our 27B) | Carries to 35B-A3B? | Notes |
|---|---|---|
| Multi-chip TP mesh (1, 4), `set_fabric_config(FABRIC_1D)` | YES, unchanged | C'7.1 work directly applies |
| Paged SDPA decode + B3 compute_kernel_config (HiFi2) | YES, unchanged | Same partial RoPE, same head_dim 256 — P21 cliff fix transfers |
| Paged KV cache + `update_cache_for_token_` | YES, unchanged | Only 10 attn layers means cache is **6.4× smaller** at same ctx than 27B |
| RoPE (V2 rotate-only) | YES, unchanged | Same theta, same partial factor; cos/sin tables reusable |
| `qwen36_gdn_decode_owned` (DeltaNet recurrence kernel) | YES, unchanged | DN config bit-identical: 32 V / 16 QK / head_dim 128 |
| `qwen36_decay_gate_decode_owned` | YES, unchanged | Same decay/gate shape |
| QK rms_norm fusion | YES, unchanged | Attention head shape (16/2/256) differs, but op is per-head |
| Vocab-sharded LM head + on-device argmax (P22) | YES, unchanged | Same vocab=248320 |
| Trace capture + replay | YES, mostly | New ops inside the trace = MoE forward; need to validate `sparse_matmul` is trace-safe |
| Per-layer TP wiring (column / row parallel + all_reduce) | YES, port pattern | MLP TP becomes Expert TP; friend's qwen36_moe.py already shows the column-shard for gate_up + row-shard for down |
| Dense MLP TP path (`server_tp.py` mlp block) | **REPLACE** with MoE block | This is the only structural change |
| MTP head probe path | YES, unchanged | Same architecture |

**Bottom line:** ~80% of our `server_tp.py` is reusable as-is. The MoE block is the one new component.

## 5. MoE-specific gotchas (from friend's qwen36_moe.py)

1. **Router math is exact (no DeepSeek-style auxiliary loss inference shim):** `softmax → topk(8) → div by sum → scatter into dense [E]` vector. Renormalization is by sum, **not** by second softmax. ttnn has `topk`, `scatter`, `softmax`, `div` — no custom op needed for routing.

2. **`ttnn.sparse_matmul` is the core MoE op** with `sparsity=[1, 1, M, E]` selector and `nnz=top_k`. Already exists in tt-metal (used by Gemma4 and gpt-oss demos in the same repo). Requires per-shape `_build_sparse_matmul_config()` — non-trivial autotuning.

3. **Batch-1 decode quirk:** sparse_matmul expects one sparsity vector per *expert batch*, not per M row. Friend's `decode_forward()` falls back to per-row Python loop when active_batch > 1. **Our server is B=1 — we avoid this entirely.**

4. **Gate/up fusion + TP interleaving:** friend fuses gate+up into one [E, 2·I, H] weight to halve sparse_matmul calls, then **interleaves per-device chunks** before column-sharding so each device gets matching SwiGLU pairs after split. Easy to mis-order this — copy friend's `load_qwen36_expert_weights()` shape logic verbatim.

5. **Shared expert is dense MLP + sigmoid gate:** `shared = sigmoid(x @ Wgate_scalar) * MLP(x)`. Cheap, no routing. Same shape as one expert.

6. **bf4 routed experts (friend's default):** `moe_expert_dtype = ttnn.bfloat4_b` "unconditionally" because routed experts are bandwidth-bound and decode latency dominates. Worth a precision ablation; bf8 fallback if drift cliff appears.

7. **Capacity / load balancing:** No capacity drop in friend's code. All 8 selected experts always run. Avoids dropped-token correctness questions but no skip-experts speedup.

8. **`mesh_partition` post-MoE:** when TP > 1 and not Galaxy, friend partitions the all-reduced output on dim=3 to match the dense MLP output contract. This is a **TP plumbing detail we cannot skip** — our residual stream expects partitioned output.

## 6. Friend's qwen36_moe.py audit (file is complete + production-grade)

768 LOC. Modules:
- `Qwen36MoERouter` — softmax + topk + renormalize + scatter, returns dense `[B, 1, E]` weights.
- `Qwen36MoEExperts` — wraps `decode_forward` and `prefill_forward` around `ttnn.sparse_matmul`. Decode and prefill paths are separate because prefill chunks into 32-row groups with a different sparsity contract.
- `Qwen36SharedExpert` — dense SwiGLU + sigmoid scalar gate.
- `Qwen36MoEBlock` — composes router + experts + shared + add + (TP partition) + memory-config reshape.

`load_qwen36_expert_weights` handles both HF layouts (fused `[E, 2I, H]` stacked or per-expert separate). Output TP partitioning: gate_up column-parallel, down row-parallel, all_reduce after down. Pad per-device intermediate to TILE_SIZE alignment.

Decode op count per MoE forward: ~14 ttnn ops (router 5, sparse_matmul ×2, slice ×2, swiglu mul, transpose, sum, all_reduce, partition, reshape). Vs dense MLP's ~7 ops — roughly 2× dispatch tax, partially offset by 4× weight-bandwidth reduction.

**Verdict:** copy this file mostly as-is. Two integration steps:
1. Wire into `server_tp.py` in place of the dense MLP block (the decoder caller pattern is already in friend's `decoder.py:72-75`).
2. Validate `ttnn.sparse_matmul` works on (1, 4) mesh + Blackhole (friend's repo targets Galaxy primarily — Blackhole P150 validation is unknown).

## 7. Effort estimate

| Task | Estimate | Risk |
|---|---|---|
| Update `model_config.py`-equivalent constants for 35B | 1 h | Low |
| Port `Qwen36MoEBlock` into our codebase | 4-6 h | Low (copy from friend) |
| Validate `ttnn.sparse_matmul` on (1, 4) P150 mesh | **6-12 h** | **HIGH** (Blackhole, mesh, possible kernel cliffs à la P1 SDPA cliff) |
| Wire MoE block into `gated_attn_step_tp` analog | 4 h | Low |
| End-to-end smoke (greedy "capital of France" probe) | 2 h | Medium (numerical correctness gates) |
| MTP head port | 4 h (already half-probed for 27B) | Low |
| Trace capture validation | 4 h | Medium (sparse_matmul in trace untested) |
| Perf benchmark + first optimization pass | 8 h | Medium |
| **Total bringup to first-token correctness** | **~24-36 h** | sparse_matmul on P150 is the dominant unknown |
| Vs from-scratch (analogous to 27B which took ~3 weeks) | ~3 weeks saved | |

Speedup vs 27B is plausible because active params are 9× smaller (3B vs 27B) — even with 2× MoE dispatch overhead and routing tax, a 3-4× tok/s improvement over our current 12.93 tok/s is a defensible projection (i.e., 35-50 tok/s).

## 8. Open questions (probe these before committing to bringup)

1. **Does `ttnn.sparse_matmul` work on Blackhole P150 + (1, 4) mesh?** Friend's reference targets Galaxy. Half-day probe at production shape `[1, 1, 1, 2048] @ [1, 256, 2048, 1024]`. Equivalent to the P1 SDPA-mesh failure mode — needs early de-risk.
2. **bf4 expert drift cliff?** We saw a HiFi4+fp32_dest_acc cliff at pos 129 on bf16 SDPA. Routed experts at bf4 may have analogous numerical cliff. Run cosine ladder vs HF bf16 reference at L=500 before declaring bringup green.
3. **Is `ttnn.topk(k=8)` cheap at vocab=256?** Trivial K and N — should be fine, but verify.
4. **Does paged SDPA + B3 config carry to head_dim=256 + 2 KV heads (vs our 27B's 4 KV heads)?** The reduction in KV heads from 4 → 2 reduces memory but may hit different SDPA grid configs.
5. **Does the `mesh_partition` post-MoE op exist in our tt-metal pin?** Friend's code uses `ttnn.mesh_partition` — confirm API parity.
6. **YaRN at factor=4.0 (262k → 1.01M):** Out of scope for first bringup; same RoPE infra applies.
7. **MTP head shape:** 35B has its own MTP head — confirm same fc + transformer + norm + lm_head structure as 27B's, or whether the architectural change requires re-probing the top-1 match rate.

## File pointers (absolute paths)

- Friend's MoE reference: `/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/experiments/.refs/tt-qwen-36/models/tt_transformers/tt/qwen36_moe.py`
- Friend's decoder MoE wiring: `/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/experiments/.refs/tt-qwen-36/models/tt_transformers/tt/decoder.py:53-78`
- Friend's MoE config parsing: `/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/experiments/.refs/tt-qwen-36/models/tt_transformers/tt/model_config.py:2696-2735, 3040-3060`
- Friend's standalone Mixtral MoE (older sparse_matmul pattern): `/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/experiments/.refs/tt-metal/models/tt_transformers/tt/mixtral_moe.py`
- DeepSeek V3 MoE reference for verify-style decoding patterns: `/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/experiments/.refs/tt-metal/models/demos/deepseek_v3/tt/moe.py`
- Gemma4 MoE (sparse_matmul template friend forked): `/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/experiments/.refs/tt-metal/models/demos/gemma4/tt/moe.py`
- Our current TP server (replace MLP block): `/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/experiments/serve/server_tp.py`

## Sources

- [Qwen/Qwen3.6-35B-A3B HF model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [QwenLM/Qwen3.6 GitHub](https://github.com/QwenLM/Qwen3.6)
- [Qwen3.6-35B-A3B blog (Alibaba)](https://qwen.ai/blog?id=qwen3.6-35b-a3b)
- [Qwen 3.6 model family overview (BenchGecko)](https://benchgecko.ai/family/qwen-3-6)
- [Qwen3.6 collection on Hugging Face](https://huggingface.co/collections/Qwen/qwen36)
- [Compute Market — Qwen 3.6-35B-A3B hardware guide 2026](https://www.compute-market.com/blog/qwen-3-6-local-hardware-guide-2026)
