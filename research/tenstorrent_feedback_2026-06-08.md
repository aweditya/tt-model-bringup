# Tenstorrent feedback 2026-06-08 (Yossi)

Capture of the email from Yossi at Tenstorrent so we don't lose the
context across compaction. The Tenstorrent team is reading our wiki +
research dirs.

## What they appreciated (signal on what's working)

- Our wiki/research approach + Q&A style entries
- The Cuda↔TT programming-model comparison wiki entry:
  `research/kernel_research/08_tensix_vs_cuda_programming_model.md`
- The kernel-optimization ablation methodology + fused-op work
- The implementation-vs-tt-metal comparison draft for Qwen3.6 GDN

## Correction: branches we missed

Our prior `Research: MoE+trace public/upstream precedents` agent
(task #245, commit `486436e`, research note
`research/moe_trace_precedents.md`) reportedly couldn't find a Qwen
3.6 demo branch. Yossi confirmed it DOES exist:

**Qwen 3.6 9B Blackhole demo**:
https://github.com/tenstorrent/tt-metal/tree/1cecd16c43cb73c218310e053fb37eaa1a380033/models/demos/blackhole/qwen3_5_9b
- Pinned commit `1cecd16`
- Path `models/demos/blackhole/qwen3_5_9b`
- Memory: [[reference-tt-metal-qwen36-branch]]

**Gemma 4 12B optimizations branch (`arg/gemma4_optimizations`)**:
https://github.com/tenstorrent/tt-metal/tree/f7d016135f1cca70bcc034b6d0745eac718084a8/models/demos/gemma4
- Pinned commit `f7d0161`
- Path `models/demos/gemma4`
- Active development by Tenstorrent engineer "arg/"
- **Directly relevant** to our active Gemma 4 perf work (#169, #178, #179, Round 11+)
- Memory: [[reference-tt-metal-gemma4-branch]]

## Research direction: tile-based + megakernel

Yossi's hint: combine our ablation methodology with tile-based
optimization + megakernel concepts:

- **TileRT**: https://github.com/tile-ai/TileRT (tile-based runtime)
- **TileOps**: https://github.com/tile-ai/TileOps (tile-based op library)
- **Paper**: https://arxiv.org/html/2512.22168v2 (megakernels / tile-based)

"I believe you'll read the thoughts in my mind when you look at these
links :)" — implicit ask: think about how TileLang/TileRT-style
scheduling could apply to Tensix. The Tensix programming model is
fundamentally tile-based (32×32 BFloat16 tiles via CB→DST→packer
pipeline); these abstractions may map cleanly.

Adjacent to our owned-kernel work: `qwen36_gdn_decode_owned`,
`qwen36_decay_gate_decode_owned`, MM7's Mamba2 SSD kernel, the topk
owned-op work in #241. A megakernel pass over multiple of these may
be the next level of fusion.

Memory: [[reference-tile-ai-megakernel]]

## Action items (NOT NOW — Gemma 4 spec-dec foreground is active)

1. **Read the Gemma 4 branch** (#269, see below) and diff against our
   `server_gemma4_unified_ttnn.py`. Capture which optimizations
   (#178 distributed RMSNorm, #179 paged-SDPA on globals, bfp8,
   DRAM-sharded MLP, others) they ship and which we have. Reconcile
   with our perf log `research/gemma4_perf_qb2_2026-06-05/log.md`.
2. **Read the Qwen 3.6 9B branch** (#270) and harvest patterns that
   could inform our 27B / 35B / Nemotron Qwen-family bringup.
3. **Investigate TileRT/TileOps/megakernels** (#271) — wiki entry +
   design exploration to evaluate the framework for our owned-kernel
   work. Probably belongs after the current spec-dec build.

## What to do NEXT

Continue the foreground Gemma 4 spec-dec build (Phase 2.B.1 — 6/8 steps
complete, #265 probe in flight). Tenstorrent feedback is captured;
follow-ups tracked. Don't context-switch.
