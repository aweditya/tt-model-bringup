# Tenstorrent Open-Source Ecosystem: Issues & Contribution Opportunities

Research date: 2026-04-22
Context: Stanford CS440LX tt-xla project, running on Blackhole P150

---

## 1. Issues Directly Related to Our Findings

### 1a. Kernel Config State Leak (HiFi4 on SDPA corrupts subsequent matmul)

**No existing issue found.** We searched extensively for issues about `compute_kernel_config` state leaking between ops, and about SDPA math_fidelity settings corrupting subsequent matmul. Nothing matches.

The closest related issue is a release-note fix for "CB config dispatch corruption when the same CB index has different configs on different core groups" — which is a similar class of bug (config state bleed) but at the circular buffer level, not the compute kernel config level.

- **Our action:** File a new bug report in tenstorrent/tt-metal
- **Estimated difficulty:** Medium (reproducer is straightforward — run SDPA with HiFi4, then matmul, observe NaN)
- **Potential impact:** HIGH. This is a silent correctness bug. Anyone running SDPA + matmul in sequence with non-default compute_kernel_config could get wrong results without knowing it. Our workaround (setting ALL ops to the same config) is not documented anywhere.

### 1b. ttnn.argmax Performance in Traced Execution (~90ms, should be <1ms)

**Partial match: [Issue #8662](https://github.com/tenstorrent/tt-metal/issues/8662) — "Implement Argmax OP on RISC-V, as a data movement kernel"** (CLOSED)

This older issue addressed argmax returning bfloat16 (precision loss for indices). It was resolved by moving to INT32 output. However, the *performance* aspect — argmax taking 90ms inside metal trace when a simple 1xN reduction should be microseconds — is NOT covered by any existing issue.

The root cause we observed: argmax appears to fall back to a host-side or inefficient multi-kernel path inside traces, becoming the dominant bottleneck in a decode loop.

- **Our action:** File a new performance bug in tenstorrent/tt-metal
- **Estimated difficulty:** Medium to Hard (requires profiling the argmax kernel path inside trace)
- **Potential impact:** HIGH for any LLM decode pipeline. Argmax is called every token; 90ms/token means it alone caps throughput at ~11 tok/sec.

### 1c. ttnn.split Fails on Blackhole Due to Tile Padding

**No existing issue found.** Searched for ttnn.split, chunk, split+blackhole — nothing specific.

- **Our action:** File a new bug report with reproducer
- **Estimated difficulty:** Easy (clear reproducer: split a non-tile-aligned tensor on Blackhole)
- **Potential impact:** MEDIUM. Workaround exists (manual slice), but split is a basic op that should work.

### 1d. L1_HEIGHT_SHARDED_MEMORY_CONFIG Crashes (No ShardSpec)

**Related: [Issue #28807](https://github.com/tenstorrent/tt-metal/issues/28807) — "ttnn.to_memory_config crashes with nd shard spec"** (OPEN)

This issue reports `RuntimeError: bad optional access` when using NdShardSpec. Our finding is similar but distinct: the convenience constant `L1_HEIGHT_SHARDED_MEMORY_CONFIG` has no ShardSpec attached, so passing it to ops crashes. The underlying problem is that MemoryConfig objects without a ShardSpec are invalid for sharded layouts.

Also related: [Issue #13007](https://github.com/tenstorrent/tt-metal/issues/13007) — "Override of output shard spec in ttnn eltwise binary op" (shard spec silently ignored)

- **Our action:** Comment on #28807 with our specific crash case, or file a separate issue about the convenience constant being unusable
- **Estimated difficulty:** Easy (the fix is either removing the constant or adding proper documentation that you must always construct your own ShardSpec)
- **Potential impact:** LOW-MEDIUM. Affects developer experience — the constant suggests a shortcut that doesn't work.

### 1e. rotary_embedding_llama Requires HEIGHT_SHARDED but Only Supports Interleaved Rotation Format

**Related: [Issue #14540](https://github.com/tenstorrent/tt-metal/issues/14540) — "Pre-Implementation Review Plan for fused_rotary_embedding Op"** (CLOSED)

This was resolved via PR #14860 which added `ttnn.experimental.rotary_embedding_llama_fused_qk`. The fused op requires HEIGHT_SHARDED q and k tensors on specific core grids. Our finding — that the non-fused `rotary_embedding_llama` requires HEIGHT_SHARDED trans_mat but only supports interleaved (not half-format) rotation — is a separate, undocumented limitation.

- **Our action:** File an issue or documentation PR about the interleaved-only limitation
- **Estimated difficulty:** Easy (documentation) to Hard (supporting half-format rotation on-device)
- **Potential impact:** MEDIUM. Affects any model using half-format RoPE (Qwen, LLaMA variants) that tries to use the on-device rotary embedding.

---

## 2. Existing Issues Related to Our Work

### 2a. SDPA Issues on Blackhole

- **[Issue #30362](https://github.com/tenstorrent/tt-metal/issues/30362)** — Paged SDPA decode fails PCC at certain positions (CLOSED). PCC failures at specific sequence positions after sliding window support was added. P0 priority, resolved.
- **[Issue #21534](https://github.com/tenstorrent/tt-metal/issues/21534)** — SDPA decode paged attention Blackhole test failure (CLOSED). Program cache entry count mismatch (3 vs 4 expected). Fix was to disable the test.
- **[Issue #15876](https://github.com/tenstorrent/tt-metal/issues/15876)** — New SDPA op for chunked prefill (feature request)
- **[Issue #14197](https://github.com/tenstorrent/tt-metal/issues/14197)** — SDPA forward and backward for training

**Relevance:** We use SDPA heavily in our transformer interpreter. The PCC failure at certain positions and the program cache mismatch are bugs we could encounter. The kernel config state leak we discovered (1a above) may be related to the PCC failures.

### 2b. Blackhole-Specific Issues

- **[Issue #41827](https://github.com/tenstorrent/tt-metal/issues/41827)** — MoE: BH (Blackhole) support (OPEN, April 2026). Getting MoE running on Blackhole. "Relatively straightforward" for basic support, optimization needed.
- **[Issue #17162](https://github.com/tenstorrent/tt-metal/issues/17162)** — `ttnn::embedding` Blackhole generality (OPEN). Only 28% of sweep tests pass on Blackhole. Infinite values in outputs. Directly relevant if we use embedding ops.
- **[Issue #21319](https://github.com/tenstorrent/tt-metal/issues/21319)** — BH harvested P150a: resnet test failures
- **[Issue #25553](https://github.com/tenstorrent/tt-metal/issues/25553)** — Blackhole P150b: Eth mailbox timeout
- **[Issue #30681](https://github.com/tenstorrent/tt-metal/issues/30681)** — Fabric testing inconsistency on Blackhole (20-30% bandwidth drop)

**Relevance:** The embedding issue (#17162) is directly relevant — 28% pass rate on Blackhole means many shapes produce garbage. We should verify our embedding usage is safe.

### 2c. Memory/Sharding Issues

- **[Issue #28807](https://github.com/tenstorrent/tt-metal/issues/28807)** — `ttnn.to_memory_config` crashes with nd shard spec (OPEN)
- **[Issue #15306](https://github.com/tenstorrent/tt-metal/issues/15306)** — Sharded memory config low PCC when shards exceed core count
- **[Issue #13007](https://github.com/tenstorrent/tt-metal/issues/13007)** — Shard spec silently overridden in eltwise binary ops
- **[Issue #24681](https://github.com/tenstorrent/tt-metal/issues/24681)** — DRAM-sharded matmul hanging when invoked twice

**Relevance:** We work extensively with sharded memory configs. The silent shard spec override (#13007) could cause subtle bugs in our interpreter.

---

## 3. Bounty Opportunities

The tt-metal repo has an active [bounty program](https://github.com/tenstorrent/tt-metal/labels/bounty) with paid issues. Most relevant to our skillset:

| Issue | Bounty | Difficulty | Relevance to Us |
|-------|--------|------------|-----------------|
| [#38114](https://github.com/tenstorrent/tt-metal/issues/38114) — Auto-optimal matmul config | $2,500 | Hard | HIGH — We understand matmul configs deeply from our kernel config leak discovery |
| [#16618](https://github.com/tenstorrent/tt-metal/issues/16618) — Add `ttnn.flip` support | $1,500 | Medium | LOW — Conv-focused, not our domain |
| [#21412](https://github.com/tenstorrent/tt-metal/issues/21412) — FFT and inverse FFT | $3,000 | Hard | LOW — Signal processing |
| Various model bring-ups | $1,500 each | Medium | MEDIUM — We have the hardware and TT-NN experience |

The auto-optimal matmul config bounty (#38114) is especially interesting given our deep understanding of how matmul performance depends on core grid, compute kernel config, and memory layout.

---

## 4. Tenstorrent's Official tt-xla Repository

Tenstorrent has their own [tt-xla repo](https://github.com/tenstorrent/tt-xla) — a PJRT plugin connecting JAX/PyTorch to TT-MLIR. This is distinct from our project (which uses TT-NN directly via a Jaxpr interpreter). Key observations:

- **861 open issues** as of April 2026 — very active development
- Uses TT-MLIR compiler backend (not TT-NN directly)
- Focused on vLLM integration and model bringup at scale
- Recent issues: SDPA composite matching (#4337), performance regressions (#4312), training OOMs (#1650)
- [Issue #1537](https://github.com/tenstorrent/tt-xla/issues/1537) — PJRT plugin cannot coexist with torch in same process

**Relevance:** Their approach (PJRT + TT-MLIR) is the "proper" path we researched in Wiki 23. Our direct TT-NN interpreter approach is complementary — we operate at a lower level and can surface bugs that the MLIR path may mask. We could potentially contribute test cases or bug reports upstream that help both projects.

---

## 5. Community & Developer Experience

- **GitHub Discussions**: [tenstorrent/tt-metal/discussions](https://github.com/tenstorrent/tt-metal/discussions) — active Q&A forum
- **Discord**: Primary real-time community channel
- **Developer Hub**: Tenstorrent's official developer resources
- **Contributing**: [CONTRIBUTING.md](https://github.com/tenstorrent/tt-metal/blob/main/CONTRIBUTING.md) — standard PR workflow
- **Bounty Program**: $500-$50k bounties for various contributions

Developer experience feedback (from The Register review, Nov 2025): Finding demos requires digging through GitHub repos, documentation is sparse, new users shouldn't need to parse code comments to get demos running. Our wiki and experiments could serve as unofficial community documentation.

---

## 6. Recommended Actions (Prioritized)

### Priority 1: File Bug Reports (High Impact, Easy Effort)

1. **Kernel config state leak bug** — File in tenstorrent/tt-metal with full reproducer (SDPA with HiFi4 -> matmul produces NaN, fixed by uniform compute_kernel_config). This is a silent correctness bug affecting anyone running multi-op pipelines.

2. **ttnn.argmax 90ms in trace** — File performance bug with profiling data showing argmax dominates decode latency. Include comparison of traced vs non-traced execution times.

3. **ttnn.split on Blackhole** — File with minimal reproducer showing tile-padding failure.

### Priority 2: Contribute Documentation (Medium Impact, Easy Effort)

4. **Document HEIGHT_SHARDED requirements** for rotary_embedding_llama, SDPA, and other ops that have undocumented sharding constraints. Could be a PR to tt-metal's tech_reports.

5. **Document compute_kernel_config best practices** — Explain the all-or-nothing requirement we discovered (must set config on ALL ops or NONE).

### Priority 3: Engage with Existing Issues (Medium Impact, Medium Effort)

6. **Comment on [#17162](https://github.com/tenstorrent/tt-metal/issues/17162)** (embedding Blackhole generality) with our test results if we encounter embedding failures.

7. **Comment on [#28807](https://github.com/tenstorrent/tt-metal/issues/28807)** (shard spec crash) with our L1_HEIGHT_SHARDED_MEMORY_CONFIG crash case.

### Priority 4: Bounty Work (High Impact, High Effort)

8. **Consider [#38114](https://github.com/tenstorrent/tt-metal/issues/38114)** — Auto-optimal matmul config ($2,500). Our matmul expertise from the kernel config investigation makes us well-positioned. Requires Stage 1 proposal first.

### Priority 5: Contribute Test Cases

9. **Blackhole-specific test cases** — Our experiments have exercised many TT-NN ops on Blackhole P150 in configurations that standard CI doesn't cover (full transformer pipeline, traced execution with paged KV cache, multi-head attention with various head counts). Contributing these as test cases would improve Blackhole CI coverage.

---

## Sources

- [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) — Main repo
- [tenstorrent/tt-xla](https://github.com/tenstorrent/tt-xla) — Official PJRT plugin
- [tt-metal Issues](https://github.com/tenstorrent/tt-metal/issues)
- [tt-metal Bounty Label](https://github.com/tenstorrent/tt-metal/labels/bounty)
- [tt-metal Blackhole Label](https://github.com/tenstorrent/tt-metal/labels/blackhole)
- [tt-metal Discussions](https://github.com/tenstorrent/tt-metal/discussions)
- [CONTRIBUTING.md](https://github.com/tenstorrent/tt-metal/blob/main/CONTRIBUTING.md)
- [Advanced Performance Optimization Guide](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md)
- [Blackhole Bring-Up Programming Guide](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md)
