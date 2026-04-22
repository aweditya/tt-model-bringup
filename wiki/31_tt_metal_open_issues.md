# Wiki 31: Open Issues We Can Contribute To

## Q: What open issues exist in tt-xla and tt-metal that we can help with?

**A:** After surveying 861+ open issues across tenstorrent/tt-xla and tenstorrent/tt-metal, here are the 7 most actionable ones given our deep knowledge of TT-NN ops, trace capture, SDPA, KV cache, and device memory management.

## Q: What are the easy wins?

### tt-metal #32546 — `ttnn.mean` allocates 256-1024x more memory than needed
- **Severity:** P1 (Critical)
- **Problem:** `ttnn.mean()` requests 512x the input size in L1. A 0.98MB input requests 500MB allocation.
- **Failing shapes:** `(1000, 256, 1, 2)` → 500MB requested. `(20000, 256, 1, 2)` → 10GB (exceeds DRAM).
- **Root cause:** Likely a stride/padding calculation bug in the reduce kernel's buffer sizing.
- **Our edge:** We use `reduce_sum` as one of our 26 mapped Jaxpr primitives. We can isolate the exact allocation behavior on our device.
- **Difficulty:** Easy-Medium
- **Link:** https://github.com/tenstorrent/tt-metal/issues/32546

### tt-metal #25503 — Linear defaults to small core grid on Blackhole P150
- **Problem:** `ttnn.linear` uses 22-24 cores instead of available 88 on P150. 2-3x perf left on the table.
- **Our edge:** Every matmul in our GPT-2 and Qwen pipeline hits this. We can benchmark core grid overrides and quantify speedup.
- **Difficulty:** Easy — pass explicit `core_grid` parameter.
- **Link:** https://github.com/tenstorrent/tt-metal/issues/25503

## Q: What issues leverage our deepest expertise?

### tt-xla #4337 — SDPA composite not matched in llama/gpt-oss
- **Problem:** SDPA falls back to MLIR fusion patterns instead of being fused as a composite op. Affects attention performance.
- **Our edge:** We mapped SDPA and Flash-Decode tensor layouts in detail (Experiment 33b, Wiki 29). We know exactly which ttnn ops compose attention and the precise tensor shapes required.
- **Difficulty:** Medium — pattern matching in tt-mlir's composite pass.
- **Link:** https://github.com/tenstorrent/tt-xla/issues/4337

### tt-metal #36318 — nanoGPT training hangs non-deterministically on Blackhole
- **Problem:** Training hangs after 1-60 steps on Blackhole QB (4x P150c). Requires hardware reset. Still open.
- **Symptoms:** Debug loop repeats `get_closest_mmio_chip` and `Cluster::read_buffer`. Happens ~99% of runs.
- **Our edge:** We have deep experience with device hangs, trace capture stability, and the GPT-2 forward pass code paths.
- **Difficulty:** Hard — non-deterministic hangs suggest race conditions in fast dispatch.
- **Link:** https://github.com/tenstorrent/tt-metal/issues/36318

## Q: What about PJRT plugin issues?

### tt-xla #4347 — `BufferInstance::copyFromHost` PJRT semantics violation
- **Problem:** `kImmutableUntilTransferCompletes` is handled as zero-copy alias, violating the PJRT contract. Forces callers to retain host memory longer than necessary.
- **Maintainer response:** Intentional for PyTorch-XLA (lazy transfer between compile and execute), but would need different semantics for a JAX path.
- **Our edge:** We're building a JAX PJRT plugin and understand the host-device transfer lifecycle from our Jaxpr interpreter work (Wiki 23).
- **Difficulty:** Medium
- **Link:** https://github.com/tenstorrent/tt-xla/issues/4347

### tt-xla #1537 — PJRT plugin cannot be shared between torch-xla and JAX
- **Problem:** Running both PyTorch-XLA and JAX in the same process hangs because `DeviceConnector` is a singleton.
- **Our edge:** Our Wiki 23 deep dive on PJRT plugin architecture gives us deep knowledge of the initialization flow.
- **Difficulty:** Hard — singleton device management is architecturally baked in.
- **Link:** https://github.com/tenstorrent/tt-xla/issues/1537

### tt-xla #1650 — JAX model training OOMs (GPT2_XL, LongT5, mT5)
- **Problem:** Single-chip DRAM exhaustion during training. Fragmentation or over-allocation issue.
- **Our edge:** Our device memory management experience (KV caches, weight tensors, multi-trace allocation) lets us profile where DRAM pressure comes from.
- **Difficulty:** Medium
- **Link:** https://github.com/tenstorrent/tt-xla/issues/1650

## Q: What's our plan of attack?

**Phase 1 (Easy wins):**
1. Reproduce `ttnn.mean` memory bug (#32546) on our device — write minimal repro script
2. Benchmark `ttnn.linear` with explicit core grid on our GPT-2/Qwen workloads (#25503)

**Phase 2 (Medium):**
3. Study SDPA composite matching in tt-xla's compiler (#4337)
4. Test PJRT `copyFromHost` semantics with our JAX integration (#4347)

**Phase 3 (Hard):**
5. Attempt to reproduce nanoGPT hang on single P150 (#36318)
6. Profile DRAM allocation patterns during JAX model training (#1650)

---

*Research conducted April 2026 via GitHub issue search across tenstorrent/tt-xla (861 open) and tenstorrent/tt-metal.*
