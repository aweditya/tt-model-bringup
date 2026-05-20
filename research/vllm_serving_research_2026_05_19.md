# vLLM-style Serving & Disaggregated Prefill/Decode — Research Note

Date: 2026-05-19
Scope: research only. No code changes proposed here; this is the planning substrate for a future
serving-layer decision on our qb1/qb2 (4× P150) Qwen3.6-27B stack.

---

## 1. Executive summary

**Disaggregated prefill/decode (PD)** runs the prefill phase of LLM inference on one pool of
accelerators and the decode phase on a *different* pool, shipping the KV cache between them over
the network so the two phases stop fighting each other for the same compute/memory mix.
For us — a single user, single host, ~12 tok/s daily-driver — PD is overkill; the work that
actually matters is **continuous batching on a single host, a real prefill kernel, and an HTTP
front-end**. PD becomes interesting only if we ever try to serve concurrent users at low TTFT.

## 2. Why prefill and decode are different beasts

| Aspect | Prefill | Decode |
|---|---|---|
| Input shape | `[B, prompt_len, H]` (long seq) | `[B, 1, H]` (single token) |
| Bottleneck | Compute / matmul FLOPs | DRAM bandwidth (KV cache reads) |
| Arithmetic intensity | High (reuses each weight `prompt_len` times) | Very low (each weight read once per step) |
| Latency SLO | TTFT (time-to-first-token), seconds OK | TPOT (per-token), tens of ms |
| Batch friendliness | Saturates one chip on one request | Needs many concurrent users to amortize weights |
| Our cost on qb1 | 0 (we don't have a true prefill kernel) | ~83 ms/tok at MAX_POS=256, TP4 traced |

This is the same observation `tt-qwen-36/tech_reports/LLMs/llms.md §3.2` and DistServe/Splitwise
both make. Prefill wants compute-rich chips and big batches; decode wants memory-rich chips and
many concurrent users. Co-locating them means any prefill burst stalls every decoder in its
batch (the "tail ITL" problem vLLM cites), and any decode-heavy moment leaves the matmul engine
idle. Disaggregation cuts that interference.

## 3. vLLM continuous batching + PagedAttention recap

- **PagedAttention (Kwon et al., SOSP 2023):** KV cache stored in fixed-size blocks (e.g.
  16 tokens), looked up via a per-request *block table*. Kills fragmentation, enables prefix
  sharing and copy-on-write. Each attention call gathers non-contiguous blocks in a custom
  kernel.
- **Continuous batching (iteration-level scheduling):** Scheduler runs *one decode step* at a
  time across all live requests; when any finishes, a new one is admitted into the freed slot
  mid-batch. Requires (a) single-user prefill that lands KV into a specific slot, (b) batched
  decode tolerant of per-slot positions.
- **Chunked prefill:** Long-prompt prefill broken into chunks small enough to interleave with
  decode steps in the same forward pass. This is the *intra-instance* answer to the same
  prefill/decode interference problem PD-disagg solves *across* instances. Default in vLLM V1;
  reach for PD only when chunked prefill can't hold your tail ITL.

## 4. vLLM disaggregated PD architecture

```
   [HTTP/OpenAI API]
          │
   ┌──────┴──────────┐
   │   router/proxy  │  (e.g. vLLM production-stack, llm-d, NVIDIA Dynamo)
   └──┬───────────┬──┘
      │ prompt    │ token stream back
      ▼           ▲
 ┌─────────────┐  │
 │ Prefill     │  │
 │ instance(s) │  │   tensors over NIXL/UCX/RDMA (or NVLink within node)
 │ (TP=N)      │──┼──▶ KV cache pages + first token
 └─────────────┘  │
                  │
            ┌─────┴───────┐
            │ Decode      │
            │ instance(s) │   continuous batching, paged attention,
            │ (TP=M)      │   streams tokens out
            └─────────────┘
```

Implementation pointers (per vLLM v1 docs + source):

- `vllm/distributed/kv_transfer/` holds Connector + LookupBuffer abstractions for KV movement.
- `NIXLConnector` is the current production transport (NVIDIA Inference Xfer; UCX/libfabric/EFA).
- A router (vLLM `production-stack`, llm-d, Dynamo, Ray Serve LLM) picks a prefill worker per
  request, awaits KV transfer, enqueues into a decode worker's batch.
- vLLM marks PD as **experimental** and notes it does *not* improve throughput — only tail
  latency / SLO control. Throughput wins come from continuous batching + paged attention.

## 5. Alternatives and variants

- **DistServe (UCSD, Jan 2024):** Co-optimizes phase placement, per-phase parallelism (TP/PP),
  and batching against per-phase SLOs. Reports 4.48× higher *goodput* and 10.2× tighter SLO.
  "Goodput, not throughput" is the right framing for serving.
- **Splitwise (Microsoft, ISCA 2024):** Same core idea + *heterogeneous* hardware — H100 for
  prefill, A100 for decode → 1.4× throughput at 20% lower cost.
- **SGLang:** EPD (Encode-Prefill-Decode) disaggregation. Uses Mooncake's RDMA Transfer Engine
  for KV (Dec 2025; same engine vLLM v1 also integrates). Three roles: proxy + prefill + decode.
- **TGI (HuggingFace):** Historically monolithic continuous batching; recent versions ride NIXL
  backends. More "single instance with chunked prefill" than PD-native.
- **DeepSeek-V3 production:** PD-disagg by default; LMSYS blog cites 3 prefill : 9 decode nodes
  (8 H100 each). Big-MoE expert-parallelism and PD are linked (different dispatch modes per
  phase).
- **llm-d (CNCF Sandbox, Mar 2026):** Kubernetes-native orchestration over vLLM + PD-disagg;
  Red Hat / Google / IBM / NVIDIA / CoreWeave. Reports 10–30% throughput gain on identical infra.
  This is the "industry standard" packaging the user heard about.

## 6. Tenstorrent reality check

What exists in tt-metal (verified locally under `experiments/.refs/tt-metal/`):

- **vLLM fork:** `tenstorrent/vllm` `dev` branch. Integration contract documented at
  `experiments/.refs/tt-qwen-36/tech_reports/LLMs/vLLM_integration.md`:
  `initialize_vllm_model`, `allocate_kv_cache(max_num_blocks, n_kv_heads, block_size, head_dim)`,
  `prefill_forward`, `decode_forward`, `warmup_model_prefill`, `model_capabilities`. Worker glue:
  `vllm/v1/worker/{tt_worker,tt_model_runner,tt_loader}.py`. Reference impl:
  `tt-metal/models/tt_transformers/tt/{generator,generator_vllm,attention}.py`.
- **Paged-attention ops:** `ttnn.experimental.paged_fill_cache` (prefill),
  `paged_update_cache` (decode), `paged_scaled_dot_product_attention_decode` — exactly what
  vLLM requires.
- **Continuous batching demo:** `models/demos/t3000/llama2_70b/demo/demo_continuous_batching.py`
  (cited in `llms.md §3.4`). Pattern: single-user prefill into a freed slot, batched B=32 decode
  with per-slot positions, hot-swap on completion. Tracing is decode-only; prefill is eager.
- **PD-disaggregation:** **none** in tt-metal or the tt-vllm fork. The model is "continuous
  batching inside one server".

Our `experiments/serve/server_tp.py` today:

- Paged KV with `paged_update_cache` + `paged_scaled_dot_product_attention_decode` on a (1,4)
  mesh (B3 SDPA config) — the primitives vLLM-tt needs.
- `_traced_forward` is a single-token decode trace. "Prefill" in `handle_generate_tp`
  (line 5412) is `for tid in prompt_ids: _traced_forward(tid, pos)` — i.e. **N decode steps**,
  not a real parallel prefill. An 8k prompt × ~83 ms/tok ≈ **11 min of TTFT**.
- Single-request Unix-socket server; no batching infra, no admission queue.

The "split 2 chips for prefill, 2 for decode" question: **don't**.

- TP=2 loses arithmetic intensity vs TP=4. `feedback_c76_tp_alone_slower.md` already shows
  DeltaNet TP only 1.13× at 4 chips because dispatch dominates; halving makes it worse.
- One node, no inter-mesh fabric → KV transfer between two (1,2) meshes would go via host DRAM
  over PCIe.
- DistServe notes PD only wins when you saturate prefill workers; for one user, never.
- Right Tenstorrent answer: **temporal sharing on the same (1,4) mesh** (chunked-prefill style),
  not spatial split.

Missing for serious serving:

1. Real **parallel prefill** (`paged_fill_cache` with prompt as the sequence dim). Single
   biggest TTFT lever. Without it, 8k prompts are unusable.
2. Continuous-batching scheduler over a `B>1` decode trace (today `B=1`).
3. HTTP front-end (FastAPI/uvicorn) + SSE/chunked-transfer streaming.
4. Admission queue + per-slot state (positions, page tables, EOS).
5. Tracing is fixed-shape → one captured trace per `(B, seqlen_bucket)`, or eager prefill
   (tt-metal's choice).

## 7. Minimum viable path (single-user, daily-driver, 8k context)

Ranked by ROI. **Do not** start with disaggregation.

1. **Real parallel prefill using `paged_fill_cache`** (~1–2 weeks). Replace the per-token loop
   in `handle_generate_tp` with one (or chunked) prefill processing the whole prompt in
   parallel. Turns 8k prompts from minutes into seconds. Reuse `tt-transformers`
   `_prefill_forward_single_user` as recipe.
2. **HTTP server + SSE streaming** (~2–3 days). Replace Unix-socket RPC with FastAPI; map
   `/v1/chat/completions` (OpenAI-compatible) onto the existing generator so VS Code / Cursor /
   Continue can hit it. No model changes.
3. **Long-context KV growth** (~1 week). MAX_POS=256 today; paged SDPA already validated to 32k
   in isolation (`feedback_paged_sdpa_decode_works_at_32k.md`). Grow `max_num_blocks`, re-verify
   B3 SDPA config at 8k+ (cliff probe was 500 tok — need an 8k repro).
4. **Single-user chunked prefill** (~1 week). Once prefill is a real op, chunk long prompts
   into ~1k-token pieces so a second user's request doesn't wait minutes — cheap PD-disagg.

That's a **~1 month** project, all on the existing (1,4) TP path, no new hardware, no KV
transport, no extra processes. Friend's vLLM-style numbers fall out at *B=1* the moment prefill
is parallel.

## 8. Big-vision path: full continuous-batching + (later) disaggregation

Effort tiers, assuming §7 is done first:

- **Continuous batching, single instance (B=8–16 decode), no disagg:** ~1 quarter.
  New `B>1` decode traces per bucket (B=1,2,4,8,16); scheduler with per-slot positions, page
  tables, EOS; admission/eviction. tt-transformers `generator.py` is the reference. Risk: per-
  slot DeltaNet recurrent state in L1 may not fit at large B; probe before committing. Expected
  win: 3–10× *aggregate* throughput at saturation, ~0 per-user tok/s. Daily-driver doesn't see it.

- **PD-disagg on a second qb host:** +1 quarter, only if real concurrency materializes.
  KV-transfer between two (1,4) meshes in different processes/hosts; no Tenstorrent NIXL
  equivalent → ship KV blocks via host DRAM over the NIC (TCP/RDMA), not the chip fabric. Block
  layout `(max_num_blocks, n_kv_heads, block_size, head_dim)` already matches vLLM's. Riding the
  upstream `tenstorrent/vllm` fork — implement the `generator_vllm.py` contract for
  Qwen3.6-27B and let vLLM/production-stack/llm-d do the orchestration — means we write a
  model adapter, not a serving stack.

**Honest assessment:** A single daily-driver coding assistant doesn't generate the multi-tenant
load that motivates PD-disagg. Continuous batching alone buys throughput we won't consume. The
real unlock for "VS Code at 8k context" is items 1–3 in §7. PD-disagg goes on the someday shelf,
behind "actually get a second concurrent user".

---

## Sources

- vLLM disaggregated prefilling: <https://docs.vllm.ai/en/latest/features/disagg_prefill/>
- vLLM PagedAttention design: <https://docs.vllm.ai/en/latest/design/paged_attention/>
- PagedAttention paper (Kwon et al., SOSP 2023): <https://arxiv.org/abs/2309.06180>
- DistServe (Zhong et al., 2024): <https://arxiv.org/abs/2401.09670>;
  retro / "18 months later": <https://haoailab.com/blogs/distserve-retro/>
- Splitwise (Patel et al., Microsoft Research, ISCA 2024): <https://arxiv.org/abs/2311.18677>;
  blog: <https://www.microsoft.com/en-us/research/blog/splitwise-improves-gpu-usage-by-splitting-llm-inference-phases/>
- DeepSeek-V3 PD-disagg in production (LMSYS):
  <https://www.lmsys.org/blog/2025-05-05-large-scale-ep/>
- SGLang × Mooncake EPD integration:
  <https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration-v1.html>
- llm-d project: <https://llm-d.ai/> · GitHub: <https://github.com/llm-d/llm-d>
- vLLM production-stack PD use case:
  <https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/disaggregated-prefill.html>
- Anyscale continuous batching primer:
  <https://www.anyscale.com/blog/continuous-batching-llm-inference>
- Local: `experiments/.refs/tt-qwen-36/tech_reports/LLMs/llms.md` §3.2, §3.4, §3.5
- Local: `experiments/.refs/tt-qwen-36/tech_reports/LLMs/vLLM_integration.md`
- Local: `experiments/.refs/tt-metal/models/tt_transformers/tt/generator.py`,
  `generator_vllm.py`, `attention.py`
- Local: `experiments/.refs/tt-metal/models/demos/t3000/llama2_70b/demo/demo_continuous_batching.py`
  (referenced by `llms.md §3.4`; tt-metal continuous batching reference impl)
- Local: `experiments/serve/server_tp.py` (`handle_generate_tp` at line 5412 — current
  single-user, single-token-prefill, decode-trace-only server)
