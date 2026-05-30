# Web research: Tenstorrent multi-chip optimization (2026-05-14)

Agent: P (web research). Companion to Agent O's `multi_chip_optimizations_menu.md` and the in-flight Tracy agent.
Goal: surface public-web evidence about Tenstorrent multi-chip scaling, all_reduce/all_gather efficiency, fabric latency, and any optimization patterns that don't already live in `experiments/.refs/tt-metal/`.

NOTE: `WebFetch` was denied in this sandbox. All extractions below come from `WebSearch` summaries with each URL preserved. Anywhere a number is critical we cite the exact source URL so the user can verify in a browser.

---

## Summary findings (TL;DR)

1. **Independent confirmation that 4-chip Blackhole TP today scales at ~1.6-1.8×, not 4×.** El Reg's QuietBox review measured single-batch TP1→TP2 and TP2→TP4 each shedding ~25% latency, and 4-card online serving reaches **1.78× throughput vs 1 card** on Llama-3.1-70B (BFP8). Our 1.35× at 7.02 tok/s sits roughly in the same regime — slightly worse, consistent with no dram_prefetcher / no fused CCLs yet. URL: https://www.theregister.com/2025/11/27/tenstorrent_quietbox_review/
2. **Tenstorrent's own demos peak around 41% of theoretical on 4×P150 Llama70B**, with "76 of 140 Tensix cores idle" — the El Reg review explicitly blames "kernels written for Wormhole, forward-compatible but not tuned for Blackhole's higher core counts." This validates Agent O's candidates #4 (DRAM-sharded matmul) and #5 (dram_prefetcher) as the highest-ROI levers. URL: https://www.theregister.com/2025/11/27/tenstorrent_quietbox_review/
3. **GitHub tt-metal issue #26252 is the single most useful external document for our problem.** It measures `ttnn.all_gather` effective bandwidth on N300 and T3K at **1-4 GB/s** vs the 25 GB/s theoretical link, and attributes the gap to **algorithmic latency from N-1 serial ring hops + software stack overhead**, not the physical link. Direct read-across to our (1,4) mesh: persistent-buffer `all_reduce_async` (Agent O #2) only fixes the *software-stack* portion; the *ring hop* portion is fundamental and only fixed by `reduce_scatter_minimal_async` (#3) or by shrinking the message via distributed RMSNorm (#1). URL: https://github.com/tenstorrent/tt-metal/issues/26252
4. **Issue #33147 ("Investigate CCL scaling QB→LB") is an active Tenstorrent internal investigation on Llama-3.1-8B / 3.3-70B confirming bad CCL scaling on Blackhole QuietBox and LoudBox.** Tunable knobs they call out: `num_chunks_per_sync`, `num workers per direction = 2`. We are not alone, and Tenstorrent has open work in this exact area. URL: https://github.com/tenstorrent/tt-metal/issues/33147
5. **The Falcon40B 2025-07-02→07-04 perf regression took decode from 10.6 → 6.9 tok/s/u** — informative for scale: a ~35% regression is the order of magnitude of changes that get bisected. Our 1.35× → 4× gap is much bigger than a typical kernel-regression delta; the win has to come from *architectural* changes (sharded weights, fused collectives), not micro-tuning. URL: https://github.com/tenstorrent/tt-metal/issues/27831
6. **No prior public 4-chip Qwen3.6-27B Blackhole numbers exist.** Our work is the first published-context datapoint at this shape.
7. **Galaxy headline numbers (476.5 tok/s Llama70B; 350 tok/s/user DeepSeek across 16 Galaxies) are advertising, not benchmarks.** Spheron Blog and others repeatedly note: "from Tenstorrent's own benchmarking, single-model TT-Metal runs without a production serving layer."

---

## Tenstorrent official sources

### https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf — Hot Chips 2024: Blackhole & TT-Metalium (Jasmina Vasiljevic, Davor Capalija)
- The official scale-out story. Blackhole = 745 TFLOPS FP8 / 372 TFLOPS FP16, 10 × 400 Gbps Ethernet = 1 TB/s aggregate per chip, 32 GB GDDR6, 512 GB/s DRAM BW. 4×8 mesh = "Galaxy" (32 chips).
- Direct ROI for our menu: the Hot Chips slides describe SubDevices + dram_prefetcher as the "scale-out" mechanism Tenstorrent designs around. Validates Agent O candidate #5 as not just a Galaxy thing but Tenstorrent's official recommended decode pattern.

### https://tenstorrent.com/en/solutions/llm-inference — "Fastest Large-Context LLM Inference"
- Marketing page. Mentions long-context inference focus. No actionable per-op data.

### https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/TT-Fabric/TT-Fabric-Architecture.md — TT-Fabric Architecture doc
- "Data plane uses NOC for intra-device routing, point-to-point ethernet for inter-device routing." "Blackhole Ethernet core supports 2 RISC-V processors and fabric router functions are mapped to both processors." Confirms fabric model.

### https://github.com/tenstorrent/tt-inference-server/blob/main/docs/model_support/llm/Llama-3.3-70B-Instruct_galaxy.md — TT-Inference-Server Llama70B Galaxy doc
- Tenstorrent's own production serving doc. Not in our local refs (which are tt-metal). Worth reading directly — it's the canonical production wiring of Galaxy serving with vLLM-fork + continuous batching, which is the missing layer above what we have.

---

## Community / blog sources

### https://www.theregister.com/2025/11/27/tenstorrent_quietbox_review/ — Tobias Mann, "Blackhole QuietBox, Tenstorrent's AI workstation reviewed" (Nov 2025) — **MOST USEFUL EXTERNAL SOURCE**
- Independent reviewer ran TP1/TP2/TP4 sweeps on a 4×P150 QuietBox.
- Concrete numbers (model = Llama-3.1-70B BFP8):
  - **TP1→TP2 latency: ~25% reduction. TP2→TP4 latency: another ~25% reduction.**
  - **Throughput TP1→TP2: +36%. TP2→TP4: +27%.**
  - **Online serving TP4 vs TP1: 1.78× more requests served.**
  - **4-card achieves ~41% of peak theoretical perf** on 70B.
- Diagnosis from the reviewer: "all the models tested appear to be using kernels written for Wormhole, leaving 76 of the chip's 140 Tensix cores sitting idle."
- Read-across to our situation: our 1.35× scale (5.19→7.02 tok/s) is *below* even the published TP4 1.78× ceiling because (a) we haven't shipped DRAM-sharded matmul, (b) we haven't shipped fused CCLs, (c) Qwen3.6-27B + DeltaNet hybrid is unlike anything in tt-metal's tested model set. The 1.78× number is a credible near-term target before exotic optimizations.

### https://clehaxze.tw/gemlog/2025/04-21-programming-tensotrrent-processors.gmi — Martin Chang, "Programming Tenstorrent processors"
- Multi-chip mesh in software: "QuietBox contains 8 Wormhole or 4 Blackhole processors and forms a mesh of processors. The exact same scheme can scale to 32 chips attached to a single host, and to multiple hosts."
- Useful framing of programming-model continuity but no perf numbers.

### https://clehaxze.tw/gemlog/2025/03-17-memory-on-tenstorrent.gmi — Martin Chang, "Memory on Tenstorrent"
- Background reading on L1/DRAM hierarchy; reinforces that L1-resident decode is the goal (motivates Agent O #10 sharded residual).

### https://www.corsix.org/content/tt-wh-part4 — Corsix Wormhole Series Part 4: "A touch of Ethernet"
- Wormhole-era (not Blackhole) but the topology lessons carry: each E tile manages 100 Gb/s simultaneous TX+RX; "tile-to-tile propagation delay is 9 clock cycles."
- Notable design detail: cross-ASIC traffic in a 2-chip card already requires an ethernet hop (E8 → E0). That implies the *minimum* fabric latency floor for any inter-chip op is fundamentally higher than NoC ops.

### https://www.servethehome.com/tenstorrent-blackhole-and-metalium-for-standalone-ai-processing/ — Patrick Kennedy, ServeTheHome
- Reports the Hot Chips 2024 takeaways for Blackhole. Confirms 10 × 400 GbE per chip = 1 TB/s aggregate.
- Quotes the unified scale model: "32 chips per host → multiple hosts → tt-Fabric routes." No new perf numbers.

### https://www.spheron.network/blog/tenstorrent-vs-nvidia-open-source-ai-hardware/ — Spheron Blog, "Tenstorrent vs NVIDIA (2026)"
- Useful for marketing-vs-reality framing.
- Direct quote (paraphrased from search summary): "Llama 70B throughput figures for Galaxy come from Tenstorrent's own benchmarking, typically from controlled single-model TT-Metal runs without a production serving layer."
- Estimates 40-60% TFLOPS realization on Blackhole vs 60-80% on mature NVIDIA stacks.

### https://futurumgroup.com/insights/tenstorrents-galaxy-blackhole-can-risc-v-processors-expand-fast-inference-globally/ — Futurum Group
- Trade press. Repeats the "350 tok/s/user DeepSeek across 16 Galaxies at batch 32" claim; "roadmap target 500 tok/s/user at $6/M tokens." Order-of-magnitude only.

---

## GitHub issues / discussions

### https://github.com/tenstorrent/tt-metal/issues/26252 — "Understanding `ttnn.all_gather` Effective Bandwidth vs. Theoretical Bandwidth in LLM Inference" — **HIGHEST DIRECT RELEVANCE**
- Measured: T3K 1×8 ring, 0.52 MB aggregated gather → 0.18 ms → **2.9 GB/s effective vs 25 GB/s theoretical link**.
- N300 1×2 linear / T3K 1×8 ring: "consistently in 1-4 GB/s range."
- Root cause stated: "primary bottleneck is NOT the physical link speed — it is algorithmic latency (N-1 serial hops in ring all-gather) + fixed overhead from software stack, protocol handshakes (flow control), and on-chip NoC data movement."
- Direct implication for our work:
  - Agent O #2 (`all_reduce_async`) addresses the software-stack portion.
  - Agent O #1 + #3 (distributed RMSNorm + reduce_scatter_minimal_async) addresses the message-size + ring-hop portion.
  - **#1 + #3 together is likely the single biggest win available** because they cut both the data on the wire AND eliminate one of the two ring traversals.
- New idea (not in O's menu): consider raising `num workers per direction` — Tenstorrent uses 2 in #33147's parametrization; tt-fabric supports more. Per-link parallelism may amortize the hop latency.

### https://github.com/tenstorrent/tt-metal/issues/33147 — "[Blackhole] Investigate CCL scaling issues from QB to LB"
- Tenstorrent-internal test plan; Llama-3.1-8B and Llama-3.3-70B on QuietBox and LoudBox showing "bad scaling factor on CCLs (All Gather Async and Reduce Scatter async ops)."
- Test matrix: seq_len ∈ {128, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k}.
- Tunable knobs documented: `num_chunks_per_sync`, `num workers per direction = 2`.
- **Actionable**: when we ship `all_reduce_async`, expose those knobs and sweep them.

### https://github.com/tenstorrent/tt-metal/issues/27831 — "Falcon40B performance regression in CI occurred between 2025-07-02 and 2025-07-04"
- Decode dropped 10.6 → 6.9 tok/s/u in 48 hours. Useful as calibration for the *kind* of perf delta single-kernel changes produce. Our 4× target is multiple compounded changes, not one.

### https://github.com/tenstorrent/tt-metal/issues/30030 — "All to All Combine and Dispatch not functional on BH Loudbox and BH Galaxy"
- Blackhole-specific MoE collective bring-up gap. Not relevant to Qwen3.6-27B (no MoE in our path), but confirms general theme: Blackhole CCL surface is younger than Wormhole.

### https://github.com/tenstorrent/tt-metal/issues/30681 — "Fabric Testing Inconsistency on Blackhole Machines"
- Fabric correctness/test framework issue on BH. Background risk to be aware of when chasing CCL perf.

### https://github.com/tenstorrent/tt-metal/issues/36430 — "[Fabric] Bandwidth test is not working on Blackhole Galaxy because traffic does not appear to be sent by the fabric firmware"
- Blackhole Galaxy fabric firmware bug, late 2025/early 2026. Reinforces that BH fabric is still maturing.

### https://github.com/tenstorrent/tt-metal/issues/30134 — "Enable 2d all-gather benchmark on Blackhole P300 nightly"
- Confirms the 2D AG kernel exists and is being CI-enabled on Blackhole. Not yet (1,4), but the kernel infra is there.

### https://github.com/tenstorrent/tt-metal/issues/28272 — "Need multi-galaxy CCL specifications for DeepSeek"
- Multi-galaxy spec work. Galaxy/DeepSeek priorities show Tenstorrent's roadmap focus is on huge MoE scale-out, not 4-chip Qwen-style single-box decode. Implies we likely won't get a free upstream win for our shape soon — we need to do the wiring ourselves.

---

## Academic / talks

### https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf — "Dissecting the Tenstorrent Blackhole Architecture via Microbenchmarking" (ASPLOS 2025, submitted/published Sep 2025)
- Independent academic paper. Most useful concrete measurements from search summary:
  - **Single Tensix DRAM read sustained: ~55-60 GB/s.** Aggregate scales roughly linearly to ~8 cores hitting different regions, then approaches the 512 GB/s ceiling.
  - **Pipeline microbench: ~70% overlap of compute with communication achievable** on BH.
- Read-across: ~70% overlap is the empirical ceiling — Agent O #5 (`dram_prefetcher`) targeting weight-load overlap is asking for ~70% of theoretical, not 100%.
- This paper is the cleanest external citation for "what's actually achievable on a single Blackhole chip" and frames the per-chip ceilings before the fabric tax. Worth reading the PDF directly.

### http://riscv.epcc.ed.ac.uk/assets/files/hpcasia25/Tenstorrent.pdf — "Introduction to Tenstorrent" (HPC Asia 2025 tutorial slides, EPCC)
- University tutorial deck. Programming-model overview. No new perf data.

### https://arxiv.org/html/2509.19294v1 — "Accelerating Gravitational N-Body Simulations Using Tenstorrent Wormhole" (Sept 2025)
- Non-AI workload but reports concrete Wormhole performance characterizations. Not directly applicable.

### https://arxiv.org/html/2506.15437v1 — "Exploring Fast Fourier Transforms on the Tenstorrent Wormhole"
- Another scientific-computing characterization paper. Useful background on Wormhole NoC behavior; less so for our LLM decode shape.

### https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/ParallelKittens.pdf — Stanford Hazy Research, "Systematic and Practical Simplification of Multi-GPU AI Kernels" (Nov 2025)
- NVIDIA-focused but the *patterns* are transferable. Key takeaway: fusing GEMM with subsequent reduce-scatter is one of the higher-leverage patterns they identify. Maps directly onto Agent O #3 (`reduce_scatter_minimal_async` on out_proj/w2) and #8 (`all_gather_minimal_matmul_async`). Useful as a third-party validation that the fused-CCL pattern is the right shape.

---

## New optimization ideas (not in Agent O's menu)

1. **Sweep `num_chunks_per_sync` and `num workers per direction` for our (1,4) CCL calls** (per #33147). Pure tuning knob, no new kernels. Could be the difference between 2 GB/s and 4 GB/s effective AG bandwidth (#26252 sees a 2× range across tuning). Effort: trivial. Win estimate: 1-3 ms/tok if our CCLs are currently mid-range on those knobs.
2. **Audit how many ring-hops our (1,4) topology actually traverses per all_reduce.** #26252 explicitly says ring-hop count is the dominant latency. For (1,4) a ring is 3 hops out + 3 hops in for all_reduce. If the kernel uses a tree algorithm on 4 chips (only 2 levels), the wall-clock should be lower. Worth checking with Tracy whether the chosen algorithm is ring or tree — and forcing tree if not.
3. **Consider doubling the inter-chip link count by routing traffic over 2 ethernet links per pair** (Wormhole has 2 active links per neighbor — Blackhole presumably more). Some tt-metal config knobs control "num_links" on CCL ops. Worth probing what's exposed in Python.
4. **Pipeline-parallel split as a fallback if TP keeps underperforming.** El Reg explicitly cites TP > PP scaling, but at 1.35× our TP is *not* hitting that promise. For decode (small activations, large weights), a 16-layer PP split would put 4 layers on each chip, no cross-chip CCLs except residual handoff once per chip. Worth keeping as a "Plan B" if Tracy says the fabric tax is irreducible.
5. **2D fabric tuning is futureproofing.** #30134 shows Tenstorrent CI gaining 2D all_gather on Blackhole P300. When tt-metal exposes 2D collective configs on (1,4), there may be a free win even without re-meshing.

---

## Numbers worth knowing (concrete benchmarks others published)

| Source | Model | Chips | Number | Notes |
|--------|-------|-------|--------|-------|
| El Reg QuietBox review | Llama-3.1-70B BFP8 | 4 × P150 | 1.78× online throughput vs 1 chip | Independent measurement |
| El Reg QuietBox review | Llama-3.1-70B BFP8 | 4 × P150 | ~41% of peak theoretical | "76 of 140 cores idle"; pre-Blackhole-tuned kernels |
| El Reg QuietBox review | Llama-3.1-70B BFP8 | TP1→TP2, TP2→TP4 | ~25% latency drop each step | Sub-linear |
| El Reg QuietBox review | Llama-3.1-70B BFP8 | TP1→TP2, TP2→TP4 | +36%, +27% throughput | Sub-linear |
| Tenstorrent (marketing) | Llama-3.1-70B | 4 × Blackhole (QuietBox 2) | 476.5 tok/s | Aggregate, not per-user; vendor number |
| tt-metal #26252 | LLM AG microbench | T3K 1×8 ring | 2.9 GB/s @ 0.52 MB / 0.18 ms | vs 25 GB/s theoretical link |
| tt-metal #26252 | LLM AG microbench | N300 1×2 + T3K | 1-4 GB/s range | Across configurations |
| ASPLOS TT-bench | DRAM read | 1 Tensix | 55-60 GB/s sustained | Single core |
| ASPLOS TT-bench | DRAM read | 8 Tensix in parallel | ~near 512 GB/s | Linear scaling to ceiling |
| ASPLOS TT-bench | Pipeline overlap | 1 chip | ~70% compute/comm overlap | Empirical ceiling |
| Hot Chips 2024 | Blackhole spec | 1 chip | 745 TF FP8, 1 TB/s ethernet aggregate | Theoretical |
| Tenstorrent marketing | DeepSeek R1 | 1 Galaxy (32 BH) | 350 tok/s/user @ batch 32 | Roadmap to 500 |
| Spheron analyst | Blackhole | General | 40-60% TFLOPS realization | vs 60-80% NVIDIA |
| tt-metal #27831 (regression) | Falcon-40B | T3K | 10.6 → 6.9 tok/s/u in 48h | Calibration for kernel-change magnitudes |
| Our current state | Qwen3.6-27B | 4 × P150 | 7.02 tok/s, 1.35× | Below 1.78× El Reg ceiling |

---

## Dead ends

- **Tenstorrent's official blog (`tenstorrent.com/blog`, `/newsroom`)** has marketing/announcement posts (Llama-3.1 support, TT-QuietBox 2, TT-Deploy) but no engineering write-ups with multi-chip perf data.
- **HuggingFace blog**: no Tenstorrent posts surfaced in searches.
- **Modal/Together/Lambda/Latitude blogs**: no posts on running models on Tenstorrent hardware.
- **vLLM github** has Tenstorrent fork mentions but no public perf comparisons we can extract; relevant analog patterns live in vLLM PR #8089 (TP=4 ceiling) but Tenstorrent-specific commentary is absent.
- **YouTube**: Hot Chips 2024 Blackhole talk exists as a PDF; the video itself wasn't surfaced through search.
- **`asplos.dev` PDF and `clehaxze.tw` gemlog posts**: both were blocked from WebFetch in this session; content extracted via WebSearch summary only. Recommend the user download the ASPLOS PDF directly for the full microbenchmark tables.

---

## Recommended action ladder, given web evidence

1. **Set the realistic interim target = 1.78× from El Reg (i.e. ~9.2 tok/s).** Anything above that is novel territory for 4-chip Blackhole TP.
2. **Sweep `num_chunks_per_sync` / `num_workers_per_direction` on our current `ttnn.all_reduce` calls** before shipping anything new. Cheapest possible probe.
3. **Stack #1 (distributed RMSNorm) + #3 (reduce_scatter_minimal_async) together.** Web evidence (#26252) says the gain is the *product* of message-size reduction and hop-count reduction.
4. **Anchor recommendation #6 (vocab-sharded lm_head)** still stands as the lowest-risk first move.
5. **Treat #5 (dram_prefetcher) as the bridge to >2× scaling.** El Reg's "76 cores idle" + ASPLOS's "70% overlap empirical ceiling" both say this is where the remaining headroom lives.
