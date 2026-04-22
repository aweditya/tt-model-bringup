# Wiki 35: Blackhole Hardware Status & Upstream Findings

## Q: Is there a BlackholeComputeKernelConfig?

**A:** No, and there never will be. PR [#41806](https://github.com/tenstorrent/tt-metal/pull/41806) (merged Apr 18, 2026) removed all architecture asserts from compute kernel config. The author's rationale: "Grayskull is gone. This config is probably going to be usable for all future architectures." A new architecture called **Quasar** is being developed and will also use `WormholeComputeKernelConfig`.

## Q: Is our kernel config state leak bug known upstream?

**A:** **No — our finding is novel.** Extensive search of tt-metal issues and PRs found no report of `compute_kernel_config` state leaking between ops. We should file this as an issue.

**Our finding (experiments 46b-d):** On Blackhole, setting `WormholeComputeKernelConfig(HiFi4, fp32_dest_acc_en=True)` on SDPA but not subsequent matmuls causes the matmul to run with corrupted settings (cosine drops from 0.999 to 0.873 at layer 3). The fix is to apply the config uniformly to ALL compute ops.

## Q: Were there recent SDPA fixes for Blackhole?

**A:** Yes! PR [#41790](https://github.com/tenstorrent/tt-metal/pull/41790) (merged Apr 18, 2026) fixed two critical SDPA bugs:
1. **Missing write barriers in KV forwarding** — `cb_push_back` called before flushing multicast writes, causing race conditions
2. **Incorrect multicast destination count** — caused `noc_async_write_barrier()` to hang permanently

These manifested most clearly on Blackhole when NOC sanitization overhead was disabled. We should check if our tt-metal version includes this fix (firmware 19.6.0).

## Q: How many cores does Blackhole P150 have?

**A:** 120 Tensix cores (not 140). Firmware v19.5.0+ changed all Blackhole P150 cards from 140 to 120 cores. Our device reports 110 usable cores (11x10 grid) due to harvesting (2 cores harvested per column).

## Q: What are the reference performance targets?

From tt-metal's model zoo (on N300, which is 2x Wormhole):
| Model | Decode tok/s |
|-------|-------------|
| Llama-3.2-1B | 105.9 |
| Llama-3.2-3B | 68.0 |
| Qwen2.5-7B | 24.6 |

Our Qwen2.5-0.5B on single Blackhole: **29.3 tok/s** (with CPU round-trips for RoPE). The gap to 100+ tok/s is from:
1. CPU round-trips for RoPE (144 transfers per forward)
2. No trace capture (JIT overhead)
3. No memory layout optimization (HEIGHT_SHARDED)

## Q: What KV cache APIs exist beyond what we use?

**Paged KV cache** (for vLLM-style serving):
- `ttnn.experimental.paged_fill_cache(keys, k_slice, page_table, batch_idx=...)`
- `ttnn.experimental.paged_update_cache(keys, k_heads, update_idxs_tensor=..., page_table=...)`
- `ttnn.transformer.paged_scaled_dot_product_attention_decode(q, keys, values, page_table_tensor=..., cur_pos_tensor=..., scale=...)`

Note: There was a Blackhole-specific hang with `paged_update_cache` (issue #16674, resolved Jan 2025).

---

*Research from April 2026. Sources: tt-metal PRs #41806, #41790; issues #41827, #16674; model zoo PERF.md.*
