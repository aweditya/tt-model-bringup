# Web research v2: Tenstorrent multi-chip + single-chip (2026-05-14, Agent Y)

Agent: Y (web research, follow-up to Agent P). All claims sourced via WebSearch summaries (WebFetch was permission-denied on most URLs). Every fact carries the URL it came from so the user can verify in a browser.

Focus areas P did NOT hit: tt-metal pull requests (vs issues), tt-forge / tt-xla / tt-inference-server repos, vLLM Qwen3.5/3.6 hybrid-attention community work, 2026 academic papers, Galaxy Blackhole GA news from April/May 2026, MTP speculative decoding empirical data on Qwen3.5/3.6-27B, P150 core-count downgrade firmware story.

---

## Summary findings (top 7 things, in priority order)

1. **`tt-xla` already has an open Qwen3-Next issue (#1389) AND tensor-parallel JAX models for Gemma3-27B (PR #675) and Llama-3.1-8B (PR #605, draft).** Our work parallels official Tenstorrent JAX-TP bringup. Worth reading their TP wiring patterns even though Qwen3-Next != Qwen3.6-27B. URLs: https://github.com/tenstorrent/tt-xla/issues/1389, https://github.com/tenstorrent/tt-xla/pulls
2. **Empirical MTP acceptance on Qwen3.5-27B (llama.cpp NodeNestor benchmark) is only ~47.5% — and that's measured as a NET NEGATIVE because the hybrid recurrent architecture forces 150 MiB of recurrent-state checkpoint/restore per draft batch.** This directly contradicts the optimistic "MTP probe shipping → 1.5-2× speedup" framing in our D'3 memo. Speculative decoding requires acceptance >70% to break even on Qwen3.5/3.6 hybrid arch. URL: https://github.com/NodeNestor/qwen3.5-27b-mtp-llamacpp
3. **`Llama-3.1-70B BFP8 on Galaxy scales 3.8× from QuietBox→Galaxy as REPLICAS are tiled across the mesh (NOT as TP4 inside one box).** This is a key El Reg quote P missed. Tensor parallelism is sub-linear; data-parallel replica tiling IS near-linear. For our (1,4) box, replica-tiling isn't an option but pipeline parallelism (Plan B from P's research) might give us similar headroom. URL: https://www.theregister.com/2025/11/27/tenstorrent_quietbox_review/
4. **Galaxy Blackhole went GA April 28, 2026 → 350+ tok/s/user DeepSeek-R1-0528 671B in "Blitz mode," sub-4 s TTFT on 100K ctx. EE Times measured 255 tok/s/user pre-launch.** New marketing number: the Galaxy advertising peak. Reinforces that Tenstorrent has shipped Galaxy-class performance optimizations (subdevices + dram_prefetcher + fused CCL) that have not yet trickled down to a single QuietBox. URLs: https://www.theregister.com/2026/04/28/tenstorrent_galaxy_blackhole_ai_servers_ga/, https://www.eetimes.com/tenstorrent-unveils-next-gen-servers-for-fast-tokens-no-disaggregation-needed/
5. **The P150 core count was secretly downgraded 140→120 Tensix cores via firmware v19.5.0 (January 2026), claimed 1-2% perf hit.** If qb1/qb2 are running firmware ≥ 19.5.0, our chip has 120 cores not 140. Worth verifying — pretty likely the El Reg 1.78× ceiling was measured at 140-core, while we're at 120. URLs: https://www.tomshardware.com/tech-industry/semiconductors/jim-kellers-tenstorrent-is-downgrading-blackhole-p150-cards-from-140-to-120-tensor-cores-via-firmware-update-will-ship-cards-with-120-tensor-cores-going-forward-company-claims-existing-users-should-expect-1-2-percent-performance-drop, https://github.com/tenstorrent/tt-system-firmware/blob/main/doc/release/release-notes-19.5.md
6. **vLLM Qwen3.5/3.6 had multiple known critical bugs: Marlin MIN_THREAD_N=64 fails at TP>=4, FLA tensor format mismatch causes gibberish, dtype mismatch in DeltaNet during torch.compile, GDN CUDA illegal memory access.** Cross-validation of how immature the Qwen3.5/3.6 stack is on EVEN NVIDIA. Don't blame TT for issues that are upstream architectural friction. URL: https://github.com/vllm-project/vllm/issues/35924
7. **`tt-metal` GEMM_FLOPS tech report quotes Blackhole achieving 580 TFLOPs matmul.** This is the public "best achieved" number — set our matmul micro-bench ceiling at 580 not 745 (theoretical FP8). URL: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md

---

## Tenstorrent ecosystem (other repos P didn't dig into)

### https://github.com/tenstorrent/tt-xla — Official PJRT plugin (sibling to our research fork)
- Issue **#1389 "Qwen3-Next"** (Sept 15, 2025, milestone "Get 50 Top Models Running E2E"). Worth reading their bring-up plan; Qwen3-Next ≈ Qwen3.5 architecture (closer cousin than Qwen3 dense).
- Active TP JAX work:
  - **PR #675**: "Add a Tensor Parallel JAX Gemma3-27B Model" (June 2025) — sibling-sized model with TP wired up in JAX. Probably the closest reference for our planned PJRT TP4 path.
  - **PR #605**: "Add tensor parallel LLaMA 3.1-8B JAX model" (May 2025, draft) — earlier TP wiring template.
- Issue #1537 documents that the PJRT plugin can't be used by torch and jax in the same process. Not our problem now, but if we mix backends later it bites.
- URL: https://github.com/tenstorrent/tt-xla/pulls

### https://github.com/tenstorrent/tt-forge — MLIR-based compiler
- TT-Forge supports Qwen 2.5/3 models 0.5B–32B. NOT yet on the hybrid-attention models. URL: https://github.com/tenstorrent/tt-forge
- TT-Forge can run models from PyTorch, ONNX, TensorFlow, JAX, PaddlePaddle directly to hardware (per QuietBox 2 announcement). URL: https://tenstorrent.com/newsroom/tenstorrent-launches-blackhole-developer-products-at-tenstorrent-dev-day

### https://github.com/tenstorrent/tt-mlir — MLIR dialects targeting TTNN
- Just abstraction layer; no Qwen-specific code surfaced.

### https://github.com/tenstorrent/tt-inference-server — Production vLLM wrapper
- Releases include "Qwen2.5 72B" in v0.0.2 RC. Active work on DeepSeek V3.2 sparse attention via vLLM. URL: https://github.com/tenstorrent/tt-inference-server/releases
- Open issue **#607 "[Model Readiness Support]: gemma-3-27b-it"** — sibling-sized model in their bring-up queue. URL: https://github.com/tenstorrent/tt-inference-server/issues/607
- Galaxy Llama70B docs: https://github.com/tenstorrent/tt-inference-server/blob/main/docs/model_support/llm/Llama-3.3-70B-Instruct_galaxy.md — uses vLLM tt-metal integration v0.10.0; canonical production wiring example.

### https://github.com/tenstorrent/vllm — Tenstorrent's vLLM fork
- Surfaced via DeepWiki: "vLLM tt-metal integration registers TT-specific model implementations with vLLM's model registry, allowing vLLM to use TT-Metal for model inference." URL: https://deepwiki.com/tenstorrent/tt-inference-server/2-vllm-tt-metal-integration

### https://github.com/tenstorrent/tt-npe — NoC performance estimator
- "A simple network-on-chip performance estimator (NPE) for Tenstorrent Tensix-based devices." Worth a look as a tool for predicting CCL latency on (1,4) topology BEFORE running on metal. URL: https://github.com/tenstorrent/tt-npe

---

## tt-metal issues/PRs P didn't hit

### https://github.com/tenstorrent/tt-metal/issues/12102 — "Implementing Speculative Decode on Llama 3.1 8B: Master Issue"
- **Speculative decode is mainline in tt-metal for Llama 3.1 8B** (commit b6c8cc4 fixed PCC for long-seq decode). Reference for our D'3 work — there IS a working TT speculative decoder, just not for hybrid arch.

### https://github.com/tenstorrent/tt-metal/issues/12330 — "Flash Attention and Flash Decode GQA Improvements: Master Issue"
- Active master tracker for flash-decode work. Sub-issues:
  - #13365 paged SDPA Decode page table program caching
  - #14103 multi-core parallelization for small-batch flash decode
  - #29951 sliding window flash decode test failures
- Worth subscribing/scanning if we hit decode-attention perf walls.

### https://github.com/tenstorrent/tt-metal/issues/28102 — "Hang on Qwen/Qwen3-32B at long input prompts (~3000 tokens)"
- Discovered via tt-inference-server benchmark_serving.py, vLLM backend, 32 concurrent requests, 128 prompts × 3000 tokens. Prefill completes ("Finished prefill for all users up to 2999 tokens, Starting decode") then **hangs at decode**.
- Direct hit on our daily-driver concern about long context. Confirms: TT decode at 3k+ ctx is fragile even for Qwen3-32B (simpler than Qwen3.6-27B). Worth watching this issue for the fix.

### https://github.com/tenstorrent/tt-metal/issues/34167 — "Qwen produces garbage output after several decode steps" (closed Dec 2025)
- Closed but symptoms match our drift findings (`<|im_start|>` tokens emitted mid-generation). Related closed issue: #35555. Worth git-archaeologying the fix commit on main to see if our bf16-drift fix is upstream.

### https://github.com/tenstorrent/tt-metal/issues/27100 — "Qwen3-8B Accuracy Degradation in Performance Mode on N300"
- Qwen3-8B at default `DecodersPrecision.performance` gets only 70-80% of GPU reference scores on N300. Validation that the Wormhole→Blackhole performance/precision tradeoffs are unresolved upstream.

### https://github.com/tenstorrent/tt-metal/issues/26977 — "Qwen2.5-7B failing at performance & long context setup"
- LoudBox + tt-transformers crashes OOM. Another long-context fragility datapoint.

### https://github.com/tenstorrent/tt-metal/issues/30224 — "vLLM QwenVL test failing since Sept 29"
- 536MB DRAM allocation failure on QuietBox. Background fragility on multi-device + Qwen.

### https://github.com/tenstorrent/tt-metal/issues/24681 — "DRAM-sharded matmul hanging when invoked twice"
- Caveat for Agent O candidate #4 (DRAM-sharded matmul). Known hang. Have to validate fixed on our build before relying on it.

### https://github.com/tenstorrent/tt-metal/issues/27859 — "All to all dispatch not functional on blackhole"
- Critical: "both 2D and 1D fabric fail, either with a hang or with an error reporting that the node does not contain any neighbors." Foundational Blackhole-CCL fragility, intersects MoE roadmap but also any all-to-all use we might wire later.

### https://github.com/tenstorrent/tt-metal/issues/41827 — "MoE: BH (Blackhole) support"
- "Getting the op running expected to be relatively straightforward, though achieving optimal performance will require additional work." Not our path (Qwen3.6-27B is dense), but signals where Blackhole CCL effort is concentrated.

### https://github.com/tenstorrent/tt-metal/issues/40739 — "[CI] Blackhole post-commit: BH-LoudBox fabric fast unit tests – Ethernet health checks fail after 10 attempts"
- Recent (early 2026) CI failures on Blackhole fabric. Foundation flakiness — when our experiments fail mysteriously, this is a candidate root cause.

### https://github.com/tenstorrent/tt-metal/issues/15597 — "Dram pre-fetcher OP"
- The dram_prefetcher op spec: "asynchronously runs on dram-connected cores, loads data from dram, and stores it to neighbouring cores' L1." This is Agent O candidate #5's official op. Use this issue as the design contract.

### https://github.com/tenstorrent/tt-metal/issues/12637 — "Add DRAM sharded matmuls to Llama3.1-8B"
- Implementation reference for how DRAM-sharded matmul gets wired into a real LLM. Llama3.1-8B template; map onto Qwen3.6-27B dense matmuls.

---

## Tenstorrent tech reports P missed (or under-cited)

### https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md
- **Blackhole headline: 580 TFLOPs matmul achieved.** Theoretical FP8 is 745 — so 580/745 = 78% of theoretical for isolated matmul. This is the rooftop for any matmul-bound kernel; our 27% full-decode reflects everything-else tax.
- Benchmarks BFLOAT16 (HiFi4) / BFLOAT8_B (HiFi2) / BFLOAT4_B (LoFi) precision, 512x512x512 → 16384x16384x16384.

### https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md
- **Two-CQ trace pattern: one CQ writes inputs, another runs programs + reads outputs.** This unlocks I/O overlap on top of trace. We currently use 1 CQ. Worth checking if MeshDevice supports 2 CQs (Wormhole T3K does — open question for our Blackhole mesh).

### https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/FlashAttention/FlashDecode.md
- **Direct quote: "supporting small sequence length decoding (e.g., seqlen=4 or 8) would be useful for applications such as speculative decoding, where draft tokens are fed into the model for verification."** Tenstorrent acknowledges the gap. If/when this lands, D'3 B=2 verify becomes free.

### https://deepwiki.com/tenstorrent/tt-metal/7.5-performance-optimization-techniques — DeepWiki: "Multi-Device and Distributed Execution"
- Confirms MeshDevice optimizations: kernel compilation broadcasting, data broadcasting for replicated tensors, unified command dispatch, MeshEvent synchronization, trace-on-mesh.
- **Sharding patterns by topology: Galaxy (8×4) shards along dims (3, 2), T3000 (1×8) shards along dims (2, 3).** Our (1,4) box — pick whichever maps cleanest.

---

## Galaxy Blackhole GA (April 28 - May 1, 2026) — fresh news P missed

### https://www.theregister.com/2026/04/28/tenstorrent_galaxy_blackhole_ai_servers_ga/
- Galaxy: 32 BH chips per box. **DeepSeek-R1-0528 671B at 350+ tok/s/user "Blitz mode."**
- 23 PFLOPS Block FP8, 6.2 GB SRAM @ 2.9 PB/s, 1 TB DRAM @ 16 TB/s, 56 × 800G Ethernet.
- $110K/system; $440K starter cluster of 4.

### https://www.eetimes.com/tenstorrent-unveils-next-gen-servers-for-fast-tokens-no-disaggregation-needed/
- EE Times **measured 255 tok/s/user** pre-launch (cooler than vendor 350 number — same 30-50% reality discount we saw in P's QuietBox findings).

### https://wccftech.com/tenstorrent-vows-to-crush-everyone-galaxy-blackhole-hits-350-tokens-on-deepseek-r1-undercut-nvidia-gb300-ai-tco/
- "Crush everyone" pricing claim: 5× lower AI TCO vs NVIDIA GB300.

### https://tenstorrent.com/newsroom/tenstorrent-enables-ai-at-scale-with-industry-leading-performance
- **"90% pass rate for running models directly from Hugging Face"** — Tenstorrent's claim about TT-Forge model coverage. Indicates the easy-models bring-up is largely automated; the hard part (custom optimization like ours) isn't.

### https://www.theregister.com/2025/11/27/tenstorrent_quietbox_review/ — re-read for the missed quote
- Direct read-across to our (1,4): **"When scaling the Llama-3.1 70B model from QuietBox to Galaxy, tokens per second throughput scales near-linear at 3.8× as model replicas are tiled across the mesh."** This is REPLICA-tiling (data parallel), not TP — confirms our 4× TP target is fundamentally harder than the 3.8× DP scaling Tenstorrent demos.

---

## P150 firmware downgrade (140 → 120 cores, Jan 2026)

### https://www.tomshardware.com/tech-industry/semiconductors/jim-kellers-tenstorrent-is-downgrading-blackhole-p150-cards-from-140-to-120-tensor-cores-via-firmware-update-will-ship-cards-with-140-tensor-cores-going-forward-company-claims-existing-users-should-expect-1-2-percent-performance-drop
- **Firmware v19.5.0+ disables 20 cores: 140 → 120 Tensix.** All Jan-2026+ shipping cards have 120; existing cards get downgraded on firmware update.
- "Claim 1-2% performance drop on typical workloads." But raw compute: 774 → 664 TFLOPS, a -14% paper drop.
- Reason given: "present a unified interface for software compatibility." Communicated via email + GitHub firmware page.
- Backlash documented. URL: https://en.gamegpu.com/news/zhelezo/tenstorrent-otklyuchit-20-yader-na-uzhe-prodannykh-uskoritelyakh-blackhole-p150
- VideoCardz: https://videocardz.com/newz/tenstorrent-downgrades-blackhole-p150-pcie-cards-specs-from-140-to-120-cores
- **Action item**: check `tt-smi` / firmware version on qb1, qb2. If FW ≥ 19.5.0, our core count is 120 not 140. El Reg's 1.78× ceiling was reported pre-downgrade.

---

## HuggingFace Qwen3.6 community context

### https://huggingface.co/Qwen/Qwen3.6-27B — official model card
- **Architecture quote (exact)**: "64 layers with embedding dim 5120, FFN dim 17408, 16 repetitions of (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)), DeltaNet heads 48 V / 16 QK head_dim 128, Gated Attention 24 Q / 4 KV head_dim 256, 262144-token context."
- "Production workloads: SGLang, KTransformers, or vLLM are strongly recommended" — Qwen team's pick. We are essentially building a 4th option (TT-native).

### https://github.com/QwenLM/Qwen3.6/discussions/139 — "Real-world Qwen3-Next-80B on 128GB Apple Silicon — hybrid + SWA fits in 92 GB wired, 40× cache speedup"
- 1M tokens of total context costs only ~25 GB of KV — **about 4× less than naive pure-transformer estimate**. Direct confirmation that hybrid arch's *memory profile* is what makes long context tractable. Our DeltaNet 898 MiB fixed recurrent state IS the win.

### https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF, https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF, https://huggingface.co/havenoammo/Qwen3.6-27B-MTP-UD-GGUF
- Three community GGUF distributions of MTP-augmented Qwen3.6-27B already exist. The MTP weights are shipping in HF format publicly — we're not pioneering MTP availability, only Tenstorrent serving.

### https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html (vLLM Qwen3.5/3.6 recipe)
- Canonical TP setup. Per LLMKube: "vLLM wins throughput by 3-4× at high concurrency thanks to NVFP4 and PagedAttention; llama.cpp + TurboQuant wins context (one 43K-token prompt where vLLM's 16K cap was exceeded)."

### Community single-GPU benchmarks (calibration points)
- **devnen/qwen3.6-windows-server**: 158 tok/s on RTX 5090, 72 tok/s on RTX 3090 (single-GPU, no TP). URL: https://github.com/devnen/qwen3.6-windows-server
- **Medium "Overnight Stack" — RTX 3090 24GB**: 85 TPS sustained / 106 TPS peak with vLLM at 125K context. URL: https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914
- **LLMKube — 2× RTX 5060 Ti 16GB**: ~$800 hardware, 3-4× vLLM-vs-llama.cpp at concurrency, $0.13/M tokens amortized. URL: https://llmkube.com/blog/qwen3-6-27b-bakeoff
- **AEON-7 NVFP4 on DGX Spark**: 32 tok/s median / 56 tok/s peak. URL: https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4

---

## vLLM Qwen3.5/3.6 critical bugs (cross-validation of architectural friction)

### https://github.com/vllm-project/vllm/issues/35924 — "GatedDeltaNet in_proj_ba fails Marlin MIN_THREAD_N=64 at TP>=4"
- **At TP4 (matches our config!) the in_proj_ba projection of size num_v_heads (48 for Qwen3.6-27B) is too small for Marlin kernel's MIN_THREAD_N=64 constraint.** Not a TT-specific issue — even mature vLLM Marlin breaks at TP4 on hybrid arch. Confirms: TP4 is uniquely uncomfortable for Qwen3.5/3.6 even on NVIDIA.

### https://github.com/vllm-project/vllm/issues/38643 — "FLA linear attention tensor format mismatch causes gibberish output"
- Wrong tensor format silently produces gibberish, no crash. Same drift-class symptom we hit.

### https://github.com/vllm-project/vllm/issues/35238 — "dtype mismatch in DeltaNet layers during torch.compile (float != c10::Half)"
- DeltaNet operates at mixed precision and torch.compile catches it. Mirror of our bf16-vs-fp32 drift work.

### https://github.com/vllm-project/vllm/issues/34948 — "Qwen3.5 CUDA Illegal Memory Access in GDN Kernel"
- Outright IMA in the GDN kernel under specific shapes. Hybrid arch is a new kernel surface even on CUDA.

### https://github.com/vllm-project/vllm/issues/37995 — "RFC: Prefill Context Parallel for Qwen3.5 Hybrid Attention"
- Active RFC for CP support on Qwen3.5 hybrid. **Note: combines full + linear layers under one CP scheme.** When this lands, it's a free reference for how to think about long-context multi-chip on hybrid arch.

### https://github.com/vllm-project/vllm/issues/38041 — "V2 model runner crashes on Qwen3.5 mixed attention"
- vLLM's V2 model runner specifically fails the linear+full hybrid pattern. Architectural-level mismatch.

### MTP empirical findings — central D'3 implication
- **https://github.com/NodeNestor/qwen3.5-27b-mtp-llamacpp** measures **MTP layer at ~47.5% acceptance** but speculative decoding is NET NEGATIVE because the 48 gated-delta-net layers maintain ~150 MiB of recurrent state that must be checkpointed before speculation and restored on rejection.
- **Break-even threshold: ~70% acceptance** OR near-zero checkpoint overhead.
- **Some implementations report better numbers**: D3 per-position acceptance [100%, 97.96%, 93.88%] on recorded prompts — but those are best-case, not aggregate.
- **MTPLX (Apple Silicon)**: 2.24× decode TPS on Qwen3.6-27B at temp=0.6 with native MTP, no external drafter. URL: https://github.com/youssofal/MTPLX. Counter-evidence: when checkpoint cost is amortized differently (Apple MPS unified memory), MTP can still win.
- **HackMD "every llama.cpp speculative-decode mode" benchmark on Qwen3.6-35B-A3B + RTX 3090: NONE faster than baseline.** URL: https://hackmd.io/ODXuOQNzSiyUITz7g9mtBw
- **Practical conclusion for D'3**: our memory note `feedback_speculative_decoding.md` claims "1.5-2× tok/s." Web evidence narrows that to: it works only if (a) acceptance > 70% AND (b) checkpoint cost is engineered to near-zero (in-place ttnn.copy on slot 0 / slot 1 is exactly the right pattern — DeepSeek V3 reference). Otherwise it's a wash or net negative. **MTP probe DSO 57.9% match rate is below the 70% break-even threshold.** Re-evaluate before shipping D'3 to perf wall.

---

## Academic / arxiv (2026 papers P missed)

### https://arxiv.org/abs/2603.05931 — "A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA" (March 2026)
- **Most relevant 2026 paper for our architecture.**
- "Hybrid LLMs like Qwen3-Next use 75% GDN layers... at batch-1, GDN decode is memory-bound on GPUs since the full recurrent state must be round-tripped through HBM every token."
- FPGA solution: hold **full 2 MB recurrent state persistently in on-chip BRAM**, converts memory-bound → compute-bound.
- "Five-phase tiled datapath: only one read pass and one write pass per state matrix per token."
- "Exploits Grouped Value Attention structure to share query/key datapaths across pairs of values, scaling head-level parallelism without increasing pipeline interval."
- **Direct read-across to Tenstorrent**: Blackhole P150 has 210 MB on-chip SRAM. Our DeltaNet recurrent state is ~898 MiB (per Qwen team), or per-layer 150 MiB ÷ 48 layers ≈ 3 MiB per layer. **A single Blackhole chip's L1 could fit ALL 48 DeltaNet recurrent states (3 MiB × 48 = 144 MiB < 210 MiB).** This is the exact "persistent state" win the FPGA paper exploits.
- Our current `gated_attn_step` reloads recurrent state from DRAM every step (per L1 budget assumptions). If we can pin recurrent state in L1 → big perf win, potentially comparable to the FPGA paper's claim of "converts memory-bound to compute-bound."

### https://arxiv.org/abs/2603.23343 — "Numerical Kernels on a Spatial Accelerator: A Study of Tenstorrent Wormhole" (March 2026)
- New 2026 Tenstorrent-on-arxiv (Wormhole-focused, not BH).
- **"At maximum problem size, BF16 and FP32 implementations are 7× and 16× slower than H100 GPU respectively."** Sobering calibration — Tenstorrent has a long way to go on numerical-bound workloads, but we're not numerical (we're decode-LLM-bound).
- Author: Maya Taylor (independent academic), March 24, 2026.

### https://arxiv.org/pdf/2510.26692 — "Kimi Linear: An Expressive, Efficient Attention Architecture"
- Linear attention competitor to Gated DeltaNet. Relevant if Kimi Linear lands as a candidate model (per memory `reference_kimi_glm_bringup_menu.md`).

### https://arxiv.org/abs/2604.21100 — "Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences" (April 2026)
- Algorithmic improvements to DeltaNet — adds preconditioning, channel-wise vector βt. If the Qwen team adopts these in a future Qwen3.7, we'd need to re-port.

### https://arxiv.org/abs/2604.19021 — "FG²-GDN: Enhancing Long-Context Gated Delta Networks with Doubly Fine-Grained Control" (April 2026)
- Same theme: channel-wise vector βt + decouple key/value scaling. Background on where GDN is heading.

### https://arxiv.org/abs/2605.13473 — "OSDN: Improving Delta Rule with Provable Online Preconditioning" (May 2026)
- Most recent. Mostly theoretical; not actionable yet.

### https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/ParallelKittens.pdf (re-cite — P had this)
- Stanford Hazy Research on multi-GPU NVIDIA kernel patterns. Fused GEMM + reduce-scatter is the higher-leverage pattern. Direct mapping to Agent O candidate #3.

---

## Vendor analysis blogs / market

### https://newsletter.semianalysis.com/p/tenstorrent-blackhole-grendel-and — SemiAnalysis Blackhole / Grendel / Buda
- **Pre-launch (2022) Dylan Patel writeup.** Useful for the scale-out architecture philosophy. Doesn't have 2026 benchmark numbers but contextualizes Tenstorrent's deliberate scale-out bet vs NVIDIA's scale-up.
- "Tenstorrent's in-house core is a 64-bit OoO design with two 256-bit vector units, performing about the same level as Apple's old Cyclone core."
- Note: Grendel succeeds Blackhole. Not on roadmap for our 2026 window.

### Atlas / MindStudio / BuildFastWithAI blogs (2026 Qwen3.6 reviews)
- Coding-benchmark heavy, less perf-engineering. Marginal value for our work.

### IEEE Spectrum: https://spectrum.ieee.org/ai-workstation-looks-like-pcs (QuietBox 2)
- Marketing piece. Not engineering.

---

## Optimization ideas NEW to Y (not in Agent P's menu or O's menu)

1. **L1-pinned recurrent state for DeltaNet (FPGA-paper pattern).** Per https://arxiv.org/abs/2603.05931, hybrid LLM decode is memory-bound on GPUs precisely because recurrent state round-trips HBM. P150 has 210 MB L1 — total DeltaNet state ~150 MiB fits. If we can keep recurrent state L1-resident across decode steps, we shift the bottleneck. Currently `gated_attn_step` and `deltanet_step` reload state every step. Effort: medium (kernel + scheduler tweak). Potential win: 30-50% latency reduction on the DeltaNet path per the FPGA paper. NEW high-priority candidate.

2. **Two-CQ trace pipeline (I/O overlap during decode).** Per AdvancedPerformanceOptimizationsForModels tech report, current 1-CQ trace path serializes input upload + execute + output read. Two CQs let one write inputs while the other executes. Need to verify MeshDevice supports 2 CQs on Blackhole. Estimated win: 1-3 ms/tok at our current 192 ms baseline (1-2%). Effort: low-medium. URL: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md

3. **Validate firmware version & core-count BEFORE benchmarking.** `tt-smi` on qb1/qb2 to confirm we're on 140 or 120 cores. If 120, our perf headroom relative to El Reg numbers needs a 1.0875× upward adjustment (more theoretically possible than we thought). URL: https://github.com/tenstorrent/tt-system-firmware/blob/main/doc/release/release-notes-19.5.md

4. **Use tt-npe to PREDICT CCL latency before measuring.** Per https://github.com/tenstorrent/tt-npe, there's a NoC estimator. We could model expected (1,4) all_reduce latency and validate against measurement; if measured > predicted by >2×, the gap is software-stack overhead and Agent O candidate #2 (async CCL) is high-yield. Useful baseline before sinking dev time.

5. **Replica-tile DP fallback on (1,4).** El Reg said replica tiling QB→Galaxy scales 3.8×, while TP4 only hits 1.78×. **Within one QuietBox**, we could shard memory differently — instead of 4-way TP, do (DP=2, TP=2) hybrid: 2 model replicas each on 2 chips. Halves the TP all_reduce traffic AND doubles concurrent throughput. Specifically useful if (a) batch > 1 ever, OR (b) prefill/decode disaggregation matters for daily-driver UX. NEW candidate.

6. **Direct head-shard-count optimization.** Qwen3.6-27B Gated Attention has 24 Q heads / 4 KV heads. We currently shard as (6 Q + 1 KV) per chip. If we instead pad to 32 Q (8 per chip) and ignore the extra heads via mask, we get cleaner shape for sharded matmul / SDPA but pay 33% extra compute. Likely net negative for compute-bound workloads but maybe positive for memory-bound. Quick simulate-only check before any code.

7. **MTP D'3 reconsider gate.** Web evidence puts ~70% acceptance as net-breakeven. Our HF-CPU probe measured 57.9%. **Don't ship D'3 to production performance flow yet.** Either (a) re-probe on-device (server contention SIGBUS'd previous attempt — `tt-smi -r` then retry), or (b) prototype the checkpoint mechanism with timing and validate net win on a realistic prompt set BEFORE wiring B=2 verify. URL: https://github.com/NodeNestor/qwen3.5-27b-mtp-llamacpp

8. **Set ABSOLUTE ceiling at Galaxy Blackhole tok/s/user.** Tenstorrent's Galaxy (32 chips) at 350 tok/s/user on DeepSeek 671B. Our QuietBox is 1/8th the chips on a smaller model. Order-of-magnitude estimate: 350 ÷ 8 × (27B/671B reciprocal) ≈ ~700 tok/s/user is what Tenstorrent thinks 4 chips can do on a small model. Our 7 tok/s is 1% of that. The headroom is enormous but it's all locked in their proprietary Galaxy-class optimizations (subdevices + dram_prefetcher + fused CCL) that haven't trickled down to QuietBox/Qwen3.6.

9. **Sub-device decomposition for DeltaNet recurrence.** Per DeepWiki's MultiDeviceDistributedExecution: "MeshDevice implements kernel compilation broadcasting." But sub-devices INSIDE a chip aren't widely deployed. If we sub-device-partition Blackhole's 120 Tensix into (96 cores DeltaNet recurrence | 24 cores attention QKV setup), pipeline-parallel within the chip via the dedicated RISC-V data-movement engines. Speculative; needs probing.

---

## Numbers worth knowing (concrete benchmarks others published)

| Source | Model | Hardware | Number | Notes |
|--------|-------|----------|--------|-------|
| Tenstorrent Galaxy GA | DeepSeek R1-0528 671B | 32 BH (Galaxy) | 350+ tok/s/user, sub-4s TTFT 100K ctx | Blitz mode, vendor; EE Times measured 255 |
| Tenstorrent newsroom | Llama 3.1 70B BFP8 | 4 BH (QuietBox 2) | 476.5 tok/s aggregate | Vendor; replica-tiled |
| El Reg QuietBox review | Llama 3.1 70B BFP8 | 4 BH→Galaxy | 3.8× replica-tile scaling | Independent meas. (vs 1.78× TP scaling) |
| tt-metal GEMM_FLOPS report | Matmul (isolated) | 1 BH | 580 TFLOPs | 78% of 745 FP8 theoretical |
| tt-metal PERF.md | Llama 3.1/3.3-70B | 8 WH (T3K) | 16.6 tok/s/user, 164 ms TTFT | Top-1 96%, Top-5 100% |
| tt-metal PERF.md | Qwen2.5-72B | 8 WH (T3K) | 15.2 tok/s/user, 225 ms TTFT | Top-1 99%, Top-5 100% |
| tt-metal PERF.md | Qwen3-32B | Wormhole Galaxy | 65 tok/s/user, batch=1 | "Still working on further improvements" |
| tt-metal PERF.md | Qwen2.5-VL | Wormhole Galaxy | 65 tok/s/user, batch=32 | Decode mode |
| tt-metal PERF.md | Llama-3.1-8B | T3K DP=4 | 39.6 tok/s, 58ms TTFT | |
| tt-metal PERF.md | Llama-3.1-8B | T3K DP=8 | 24.9 tok/s, 86ms TTFT | |
| tt-metal PERF.md | Llama-3.1-70B TG | DP=4 | 14.8 tok/s, 189ms TTFT | TG = Tensor Glow |
| tt-metal PERF.md | Qwen2.5-32B | T3K | 22.4 tok/s, 190ms TTFT | |
| tt-metal PERF.md | Llama-3.2-11B | T3K | 62.7 tok/s, 47ms TTFT | |
| devnen GitHub | Qwen3.6-27B | RTX 5090 (single) | 158 tok/s | Native Windows, no WSL |
| devnen GitHub | Qwen3.6-27B | RTX 3090 (single) | 72 tok/s | Same stack |
| Medium "Overnight Stack" | Qwen3.6-27B | RTX 3090 (single) | 85 TPS sustained, 106 peak | 125K ctx, vLLM |
| AEON-7 HF | Qwen3.6-27B NVFP4 | DGX Spark / Blackwell | 32 tok/s median, 56 peak | |
| Google Cloud Medium | Qwen 3.5 27B | 96× B200 | 1.1M tok/s aggregate | High-concurrency vLLM |
| NodeNestor llama.cpp | Qwen3.5-27B MTP | RTX (single) | 47.5% MTP acceptance | NET NEGATIVE for speculative decode |
| MTPLX | Qwen3.6-27B native MTP | Apple Silicon | 2.24× decode TPS @ temp 0.6 | Apple MPS unified memory amortizes checkpoint |
| HackMD speculative-decode | Qwen3.6-35B-A3B | RTX 3090 | None faster than baseline | All 19 spec-decode configs tested |
| Our Qwen3.6-27B | Qwen3.6-27B BF16 | 4 BH (QuietBox) | 7.02 tok/s, 1.35× TP4 | Below all targets |
| Llama 3.3 70B alibaba API | Llama 3.3 70B | reference | 28.4 tok/s | NVIDIA reference |

---

## Dead ends (what Y also couldn't surface)

- **No SemiAnalysis Blackhole 2026 post.** Dylan Patel's Tenstorrent deep dive is from 2022 (Blackhole / Grendel / Buda). No fresh 2026 SemiAnalysis on Blackhole performance characterization.
- **No NextPlatform Blackhole inference benchmark article.** They have a Tenstorrent tag page but no Galaxy-launch benchmark deep-dive.
- **No Stanford CS preprint other than our own work.** No CS440LX / CS340 / CS324 / HazyResearch publications on Tenstorrent specifically. Hazy Research's NVIDIA-focused ParallelKittens is the closest.
- **No YouTube video of Hot Chips 2024 Blackhole talk** (only the PDF P found).
- **No Tenstorrent Discord archive surfaced.** Their Discord exists at https://discord.com/invite/tenstorrent but Google doesn't index it.
- **No public ChatBoTArena or HF leaderboard with Qwen3.6-27B on Tenstorrent.** We remain the only published-context datapoint at this shape on TT.
- **TT-Forge JAX backend docs are thin.** Per QuietBox 2 announcement TT-Forge claims JAX support but the docs are sparse. (Our PJRT research path is independent and likely complementary.)
- **No dram_prefetcher Qwen-style implementation example** — only the Llama3.1-8B template at #12637.
- **No tt-metal PR or issue specifically mentions Qwen3.5 or Qwen3-Next.** The hybrid arch hasn't reached TT-Metal's prioritization queue. We're upstream.

---

## Recommendations layered on top of P's ladder

Add (in priority order):

A. **Verify firmware version on qb1 + qb2 immediately.** If FW ≥ 19.5.0, real core count is 120 not 140; recalibrate all rooflines. *Effort: 30 seconds.*

B. **Run tt-npe to predict (1,4) CCL latency before shipping `all_reduce_async`.** If predicted ≈ measured, software stack isn't the gap. *Effort: 1-2 hours.*

C. **L1-pinned DeltaNet recurrent state probe.** Mirror the FPGA paper's persistent-state pattern. Single-chip experiment first, multi-chip after correctness. *Effort: 2-3 days.* **Potentially biggest single win on Qwen3.6's hybrid arch.**

D. **Don't ship D'3 (MTP speculative decoding) for performance gains yet.** 57.9% probe < 70% break-even. Either improve acceptance via better B=2 verify, or shelve. *Effort: re-evaluate, don't build.*

E. **2-CQ trace pipeline.** Smaller win (~1-2%) but cheap. *Effort: 1 day.*

F. **Add hybrid (DP=2, TP=2) configuration to the Plan B menu** for concurrent serving when daily-driver UX permits >1 in-flight prompt.

Pair these with P's existing recommendations (sweep CCL knobs, distributed RMSNorm + reduce_scatter_minimal_async stack, DRAM-sharded matmul, vocab-sharded lm_head). Together they form a 7-9 item priority queue with concrete ROI bands.
