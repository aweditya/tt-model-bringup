# Qwen3.6-35B-A3B Implementation Plan (MoE bringup on qb2 (1,4) P150)

Date: 2026-05-19. Companion to `research/qwen36_30b_a3b_bringup_research.md`
(architectural survey) and `research/archive/owned_gdn_diagnosis_2026_05_18.md`
(canonical gate-trail pattern). This is a *planning* document. No code runs
here; the next agent picks this up and starts at the day-1 checklist at the
bottom.

Build-from-scratch principle is in force
(`memory/feedback_build_kernels_from_scratch.md`): every novel kernel ships as
an `owned_*` op we wrote, with `tt-qwen-36/qwen36_moe.py` consulted only as
an API/pattern hint when blocked.

## 1. Architecture diff vs Qwen3.6-27B (shipped)

Backbone is structurally identical: 10 blocks of `3 × GatedDeltaNet + 1 ×
GatedAttention` (vs 16 blocks for 27B, so 40 layers vs 64), same DeltaNet
head config (32 V / 16 QK / head_dim 128), same partial RoPE (factor 0.25,
theta 1e7), same vocab/tokenizer (248320 padded), same MTP head shape.
Attention shrinks to 16 Q / 2 KV / head_dim 256 (vs 24/4/256), but the paged
SDPA + B3 compute_kernel_config recipe carries over unchanged.

The single structural change is the MLP: dense SwiGLU (intermediate=17408,
~270M params/layer) is replaced by a Mixture-of-Experts block — 256 routed
experts at `moe_intermediate_size=512`, top-8 selection, plus one
sigmoid-gated dense shared expert at the same 512 intermediate. Per-chip
memory budget at TP=4 with bf8 routed experts is ~8 GB
(`qwen36_30b_a3b_bringup_research.md:43`); with bf4 it drops to ~6 GB.
Either fits comfortably in 12 GB DRAM with paged KV headroom for 32k+
context. The model is *smaller in active bytes per chip* than our 27B
because expert sparsity is 32× and routed experts are quantized.

## 2. Infrastructure reuse map (vs `experiments/serve/server_tp.py`)

| Component (server_tp.py) | Status for 35B | Notes / file refs |
|---|---|---|
| Mesh open `(1,4)` + `FABRIC_1D` | UNCHANGED | `server_tp.py:177-180` |
| `_tp_all_reduce` (num_links=2, explicit) | UNCHANGED | `state.collective_mode = "explicit_all_reduce"` (`:121`) |
| Paged SDPA decode + B3 compute_kernel_config | UNCHANGED | head_dim 256, 2 KV heads — same family, but de-risk per §7 |
| `update_cache_for_token_` / page_table | UNCHANGED | KV cache is 5× smaller (10 attn layers vs 16) |
| `gated_attn_step_tp` (`:806`) | MINOR (head shape constants) | NQ_PER_CHIP becomes 4 (vs 6), NKV_PER_CHIP stays 0.5 — **needs round-up handling for 2 KV heads on 4 chips** (see §7 risk) |
| `deltanet_step_tp` (`:592`) | UNCHANGED | DN config bit-identical to 27B |
| `ttnn.experimental.qwen36_gdn_decode_owned` | UNCHANGED | Owned kernel, already production-default (`:129`) |
| `ttnn.experimental.qwen36_decay_gate_decode_owned` | UNCHANGED | Owned kernel, default since 2026-05-19 (`:149`) |
| `mlp_step_tp` (`:779`) | **REPLACED** | Becomes `moe_step_tp` — the new work |
| RoPE V2 rotate-only + on-device cos/sin lookup | UNCHANGED | Same theta, same partial factor |
| QK rms_norm fusion | UNCHANGED | Per-head op |
| Vocab-sharded LM head + on-device argmax (P22) | UNCHANGED | Same vocab 248320; `forward_token_tp_inner:1030-1036` |
| On-device embed via `ttnn.embedding` (P25) | UNCHANGED | `:993-998` |
| Trace capture + replay | **MAYBE REPLACED** | dynamic top-K indices and `sparse_matmul` (if we use it) must be trace-safe — unknown until probed |
| HF tokenizer + `generate_tp` + handle_generate | UNCHANGED | Same Qwen tokenizer |
| `MeshServerState.layers[]` schema | EXTENDED | Per-layer dict gains `'moe': {router_w, expert_gate_up_w, expert_down_w, shared_*}` instead of `'mlp'` |

About 80% of the existing server is reusable as-is. The MoE block is the only
structural new work; everything else is either constant-tweak or carried
over.

## 3. MoE block design (the new work)

### 3.1 Router

Input: `x` of shape `[1, HIDDEN]` (HIDDEN=2048). Router weight:
`[HIDDEN, NUM_EXPERTS]` = `[2048, 256]` bf16, replicated on every chip.

```
logits  = x @ W_router               # [1, 256]
probs   = softmax(logits, dim=-1)    # softmax-BEFORE-topk (Qwen choice)
vals, idxs = topk(probs, k=8)        # [1, 8] each
weights = vals / sum(vals, -1)       # renormalize by sum (not by a second softmax)
```

`probs` is the active mass; the top-8 weights are then **divided by their
sum**, not softmaxed again. This matches Qwen's HF reference and friend's
implementation. Output is `(weights[1,8], idxs[1,8])` per token. We do NOT
need to scatter into a dense `[1, 256]` vector if we build our own dispatch
(see §3.3) — friend's code scatters because `ttnn.sparse_matmul` consumes a
dense routing vector.

Stability note: softmax-before-topk is the cheaper variant (one softmax, no
per-K renorm needed for gradients) and is what Qwen ships. Don't reorder
this; the renormalize-by-sum step is mathematically required to keep the
weighted output an interpolation (sum of weights = 1).

### 3.2 Expert weights layout on (1,4) mesh

Three options, with trade-offs:

| Layout | Per-chip routed-expert bytes (bf8) | Token routing pattern | Down-side |
|---|---|---|---|
| (A) **Experts sharded across chips** (64 experts/chip) | ~2 GB | Each token must route to chip(s) holding its top-K experts → cross-chip token shuffle (expensive!) | Routing comm dominates |
| (B) **All experts replicated** | ~8 GB | No cross-chip token movement; each chip independently runs its assigned top-K compute | 4× weight memory; mostly idle weight in DRAM |
| (C) **Intermediate-dim sharded (column-parallel)** | ~2 GB | Each chip holds all 256 experts but only `INTER/4 = 128` of the SwiGLU intermediate; same routing pattern as dense MLP TP, same `all_reduce` at the end | Cleanest |

**Pick (C).** This is what friend does and what our dense MLP already does
for 27B. Each chip's `expert_gate_up_proj` is shape `[256, HIDDEN, 2*INTER/4]`
= `[256, 2048, 256]` bf8 ≈ ~33 MB (negligible), and `expert_down_proj` is
`[256, INTER/4, HIDDEN]` = `[256, 128, 2048]` ≈ 16 MB. Total routed expert
storage per chip: ~50 MB × 40 layers ≈ 2 GB. Shared expert weights are
similarly column/row-parallel sharded at ~tiny footprint.

Layout (C) keeps the existing TP idiom: column-parallel `gate_up`,
row-parallel `down`, one `all_reduce` after `down`. The only "MoE-ness"
lives in *which* of the 256 expert slabs each chip computes — and since each
chip has all 256 expert weight slabs and the routing decision is replicated,
each chip independently computes its top-8 SwiGLU and the all_reduce sums
the per-chip intermediate-dim partials.

### 3.3 Token dispatch (decode B=1)

At B=1 (our decode case), only top-8 of 256 experts are active per token,
and the *same* 8 experts are active on every chip (router is replicated).
So dispatch reduces to: for each of the top-8 expert indices, gather that
expert's weight slab and do an ordinary dense matmul on `x [1, HIDDEN]`.

Two implementation strategies:

**(i) Naive loop (G1 first cut, build-from-scratch).** Python for-loop
over the 8 selected indices: `for e in top8: y_e = silu(x @ W_gate[e]) *
(x @ W_up[e]); out += weights[e] * (y_e @ W_down[e])`. Cost: 8 dispatches
× 3 matmuls = 24 dispatches per layer × 40 layers = 960 matmul dispatches
per token. That's hideous: at ~50 µs/dispatch we'd be at ~50 ms/token *just
in dispatch overhead* before any compute lands. NOT viable as final, but
correct and a great G0/G1 reference.

**(ii) Fused indexed-matmul kernel — `owned_moe_expert_decode`.** Build a
single ttnn op that takes `(x [1, HIDDEN], W_gate_up [E, HIDDEN, 2*I_local],
W_down [E, I_local, HIDDEN], top_idxs [8], top_weights [8])` and emits one
output `[1, HIDDEN]`. Per the build-from-scratch principle, this is the
candidate to *own*. Internally: for each of K=8 selected experts, the
kernel reads the indexed weight tiles directly from DRAM (gather inside the
data-movement kernel), does the SwiGLU+down contraction on Tensix cores,
accumulates weighted into one output tile per core, and reduces. Reference
patterns: our `qwen36_gdn_decode_owned` does similar per-slot indexed weight
reads.

**Decision: start with (i) in G1 for correctness, then build (ii) as the
owned kernel for G2-G3.** Do NOT start by integrating `ttnn.sparse_matmul`.
Friend uses it because they need batch-prefill support; we are B=1 decode
and the loop variant is simpler to validate. If our owned kernel hits a
wall, sparse_matmul is a fallback (§7).

### 3.4 Shared expert (always-on dense path)

Per-token cost: `shared_pre = silu(x @ W_sg) * (x @ W_su); shared = shared_pre @ W_sd;
gate_score = sigmoid(x @ W_gate_scalar); shared = shared * gate_score`. This
is exactly a dense SwiGLU MLP with intermediate=512 (much narrower than the
27B's 17408) plus a single `[HIDDEN → 1]` scalar gate. Reuse `mlp_step_tp`
verbatim, swap intermediate constant, add the sigmoid scalar gate on top.
~10% of MoE block compute, easy reuse.

### 3.5 Output gather and TP boundary

Each chip's routed output is `[1, HIDDEN]` (partial across the
intermediate-dim shard). After the down-proj we already have a partial sum
along the contracted intermediate dim; `_tp_all_reduce` finishes the sum.
Shared expert output goes through the same `_tp_all_reduce`. The two
all-reduced outputs are added, then residual-added back to `x`. Mirrors
`mlp_step_tp:799-801`. One `all_reduce` for routed + one for shared = 2
collectives per MoE layer (vs 1 for dense MLP) — small dispatch tax.

## 4. Build-from-scratch kernel candidates

Per `feedback_build_kernels_from_scratch.md` and our owned_gdn/owned_decay_gate
pattern (`experiments/owned_ops/`), kernels we should consider owning:

1. **`owned_moe_expert_decode`** (HIGH priority). Fused indexed gather +
   per-expert SwiGLU + weighted accumulate. Replaces the naive Python loop.
   This is the *MoE-specific* compute kernel and the one that earns its
   keep. Estimate: ~2 weeks like owned_gdn (lots of reader/dataflow work
   around the indexed weight read).

2. **`owned_topk_router`** (MEDIUM priority). Fused softmax → topk(8) →
   div-by-sum. Optional — `ttnn.softmax + ttnn.topk + ttnn.sum + ttnn.div`
   is correct and already small (~5 ops). Probe first: if dispatch overhead
   on these 5 ops is < 1 ms, defer ownership and just use stock ttnn ops.

3. **Do NOT own**: `ttnn.embedding` (used for router input pass-through),
   `ttnn.silu`, `ttnn.sigmoid`, `ttnn.linear` for shared expert. These are
   integration-only and the no-own clause applies.

4. **Defer entirely until measured**: prefill MoE kernel
   (`owned_moe_expert_prefill`). Our server is decode-dominant; prefill MoE
   can use the naive loop in the first cut (slower but correct), and we
   only revisit if prefill latency becomes the user-visible bottleneck for
   long-context use.

Start simple: G0 numpy reference, G1 with stock ttnn (naive loop), only
then design the owned kernel from a known-correct reference.

## 5. Staged validation gates (G0 → G5)

Same gate ladder as owned_gdn / owned_decay_gate
(`research/archive/owned_gdn_diagnosis_2026_05_18.md`).

| Gate | Scope | Artifact | Pass criterion |
|---|---|---|---|
| G0 | Pure-numpy fp32 MoE block forward (router + routed + shared, single layer 0) vs HF Qwen3.6-35B-A3B layer 0 forward | `experiments/91x_qwen36_35b_moe_numpy_oracle.py` | cosine ≥ 0.9999 element-wise, max\|Δ\| ≤ 1e-5 |
| G1 | Single-chip TTNN MoE block on qb1 (stock ttnn naive expert loop, no mesh) at REAL weights | `experiments/utils/g1_moe_single_chip_probe.py` | cos ≥ 0.999 vs G0 oracle, layer 0 only |
| G2 | TP mesh MoE block on qb2 ((1,4), real weights, naive loop + `_tp_all_reduce`) layer 0 | `experiments/utils/g2_moe_tp_mesh_probe.py` | cos ≥ 0.999 vs G0; inter-chip cos = 1.0 after all_reduce |
| G2.5 | `owned_moe_expert_decode` kernel correctness | `experiments/owned_ops/qwen36_moe_expert_decode_owned/test_*.py` | ULP-aware diff vs G1 reference; ≤ 2 BF8 ULP per output element |
| G3 | Multi-layer end-to-end teacher-forced cosine ladder (500 positions, mirrors qb1 P21 + qb2 owned_gdn gate) | `experiments/utils/cosine_ladder_moe_500pos.py` | ≤ 3% top-1 disag, median cos ≥ 0.998, NO cliff (rolling 50-step medians flat) |
| G4 | Production server integration (`server_tp.py` with `moe_step_tp` in place of `mlp_step_tp`), traced decode | `handle_generate_tp` returns coherent output for "The capital of France is" | "Paris" (or equivalent coherent answer); first-token latency measured |
| G5 | Long-context validation: needle-in-haystack at L=500, L=1024, L=4k | `experiments/utils/needle_haystack_b3_moe.py` | ≥ 3/4 retrievals at L=500 (matches owned_gdn bar); cliff-free |

Gate G2.5 splits because the owned kernel is its own validation chain; the
naive loop in G2 acts as the reference for the owned kernel.

## 6. Effort estimate (days)

Calibrated against owned_gdn (~14 days) and owned_decay_gate (~7 days).
Includes bootstrap waits (~10 min per qb2 cold start).

| Stage | Estimate | Confidence |
|---|---|---|
| G0 (numpy MoE oracle + HF cross-check) | 2-3 days | High — `91w` is template |
| G1 (single-chip naive ttnn loop, qb1) | 2-3 days | High — dense ttnn primitives only |
| G2 (TP mesh, naive loop) | 3-4 days | Medium — TP plumbing for indexed-experts is novel |
| G2.5 (owned_moe_expert_decode kernel) | 10-14 days | **LOW** — gather-from-DRAM in dataflow kernel is new ground |
| G3 (cosine ladder, multi-layer) | 2-3 days | Medium — could reveal numerical surprises (bf8 expert drift?) |
| G4 (server integration + trace capture) | 3-5 days | Medium — trace may break on dynamic top-K (see §7) |
| G5 (long-context validation) | 2 days | High — reuses owned_gdn probes |
| **Total** | **24-34 days** | bringup-to-perf-baseline; could double if owned kernel hits a tt-metal blocker |

Speedup target: with active params ~9× smaller than 27B (3B vs 27B) plus
dispatch overhead from K=8 expert dispatches per layer, a defensible
projection is 25-40 tok/s. Do not cite this as a measurement.

## 7. Risks and open questions

1. **Can we trace a forward with dynamic top-K?** Trace capture records a
   static graph; if `top_idxs` changes per token (it does), the captured
   trace must either (a) read `top_idxs` from an input tensor (data-driven
   gather inside the kernel), or (b) be re-captured per token (defeats
   purpose). Friend uses `ttnn.sparse_matmul` which consumes a dense
   `[1, 256]` routing tensor — that IS trace-safe. Our owned kernel must
   read indices from a tensor input, NOT bake them as kernel arguments.
   **De-risk early in G2.5 design.**

2. **`ttnn.sparse_matmul` on Blackhole P150 + (1,4) mesh.** Friend targets
   Galaxy. Unknown if it works on our hardware. We avoid this by owning
   the kernel, but if our owned kernel hits a wall, sparse_matmul is the
   fallback — and the fallback needs its own probe.

3. **Memory: bf4 vs bf8 routed experts.** Friend defaults to bf4 ("routed
   experts are bandwidth-bound and decode dominates"). We saw a HiFi4
   cliff at pos 129 on bf16 SDPA decode
   (`feedback_fp32_sdpa_cliff_probe.md`). bf4 may have an analogous
   numerical cliff at long context. Start with bf8 in G1-G3 (matches our
   27B baseline), only switch to bf4 after G3 passes — and re-run G3 ladder
   to confirm no cliff appears.

4. **At B=1, only 8 of 256 experts active per token = 96.9% sparse.** Does
   our matmul kernel benefit, or do we waste compute on the unselected
   experts? With the owned kernel doing indexed gather we only touch 8
   experts' weights — bandwidth-efficient. With `ttnn.sparse_matmul`, the
   sparsity vector tells the kernel which experts to skip; should also
   skip the unused DRAM reads. The naive loop variant naturally skips
   unused experts. All three honor the sparsity; the question is dispatch
   overhead, not wasted compute.

5. **Attention head TP at 2 KV heads / 4 chips.** Our 27B has 4 KV heads
   on 4 chips → 1/chip. 35B has 2 KV heads on 4 chips → 0.5/chip, which
   is illegal. Options: (a) replicate the 2 KV heads (2 chips compute
   identical KV, others mirror) — needs `gated_attn_step_tp` rewiring;
   (b) only TP along Q heads and replicate KV cache. **Decide before G4.**
   Both options preserve correctness; (b) is simpler and what TT-Metal
   Galaxy llama3_70b does for similar GQA shapes.

6. **MTP head on 35B.** Architecture probably matches the 27B's MTP head
   shape (`feedback_mtp_head_probe.md`), but new model = re-probe top-1
   match rate before committing to D' speculative decoding.

7. **Per-layer weight load time.** 40 layers × ~50 MB routed experts +
   ~16 MB shared per chip + ~16 MB attn/DN = ~3.4 GB / chip transferred at
   bootstrap. With qb2 bootstrap at ~5-10 min today, expect ~10-15 min
   cold start. Cache to disk like 27B does.

## 8. Order of operations (first concrete commits)

1. **First commit:** `research/qwen36_35b_a3b_implementation_plan.md` (this
   doc).
2. **Second commit:** `experiments/91x_qwen36_35b_moe_numpy_oracle.py` —
   pure-numpy MoE forward using `safe_open` weights from HF
   Qwen3.6-35B-A3B layer 0. Validates against directly-instantiated HF
   `Qwen36MoeBlock` if importable; otherwise against per-component HF refs
   (router, expert, shared expert). Target: G0 pass at single forward.
3. **Third commit:** `experiments/utils/g1_moe_single_chip_probe.py` — qb1
   only, single-chip stock-ttnn naive expert loop at real layer 0 weights.
   Cosine vs G0 oracle.
4. **Fourth commit:** `experiments/utils/g2_moe_tp_mesh_probe.py` — qb2,
   `(1,4)` mesh, naive loop with column/row sharding + `_tp_all_reduce`.
5. **Fifth commit (only after G2 green):** start
   `experiments/owned_ops/qwen36_moe_expert_decode_owned/` skeleton —
   follow `qwen36_gdn_decode_owned/` directory layout (compute kernel,
   dataflow, program factory, nanobind, tests, README, INTEGRATION.md).
6. Drive G2.5 → G3 → G4 → G5 in order. Each gate gets its own commit
   with the gate artifact JSON saved in `.cache/qb2_35b_moe/`.

## 9. Explicitly NOT in scope for this plan

- Speculative decoding (D' branch) on MoE. Defer until baseline G4 ships.
  MTP head probe re-run is also deferred to post-G4.
- Cross-host expert parallelism (e.g., spreading 256 experts across qb1+qb2
  for compute parallelism). Single-host (1,4) mesh only.
- Prefill MoE owned kernel. Naive loop suffices for G3-G5 prefill; revisit
  only if long-context prefill latency becomes user-visible.
- YaRN rope scaling beyond 262k. Same RoPE infrastructure applies if we
  ever need it; not a bringup blocker.
- Routed expert capacity / load balancing dropouts. Friend does NOT drop
  tokens; we follow that to avoid dropped-token correctness questions.
- Switching from our owned GDN/decay-gate kernels back to ttnn primitives.
  These ship across both 27B and 35B unchanged.
- Tracing the MoE forward in the **first** integration. G4 can ship with
  *eager* MoE inside an otherwise-traced backbone if dynamic top-K breaks
  trace; perf optimization comes after correctness.

## 10. Day-1 checklist for the next agent picking this up

1. **Re-read this doc end-to-end, plus `research/archive/owned_gdn_diagnosis_2026_05_18.md`
   for the gate-trail pattern, plus `experiments/91w_qwen36_27b_dn_attn_numpy_oracle.py`
   (or whatever the current 27B numpy oracle file is) as the G0 template.**
2. **Download Qwen3.6-35B-A3B weights to qb2** (`huggingface-cli download
   Qwen/Qwen3.6-35B-A3B --local-dir ~/models/qwen36_35b_a3b`). Verify
   `config.json` exposes `num_experts=256`, `num_experts_per_tok=8`,
   `moe_intermediate_size=512`. Confirm `shared_expert_intermediate_size`
   field exists (Qwen sometimes inlines this).
3. **Write `experiments/91x_qwen36_35b_moe_numpy_oracle.py`** with the
   G0 forward and assert vs an HF reference. Commit when cosine ≥ 0.9999.
4. **Do NOT start the owned kernel until G2 passes.** Resist the temptation.
   The naive loop is the reference; the owned kernel optimizes a known-correct
   target.
5. **Before any kernel work, probe whether `ttnn.embedding +
   ttnn.linear(W_router) + ttnn.softmax + ttnn.topk(k=8) + ttnn.sum +
   ttnn.div` is trace-safe on qb2 (1,4) mesh** with replicated weights.
   That answers the central trace-vs-dynamic-routing question (§7 risk 1)
   for free, before any expert compute lands. 1-day probe.
