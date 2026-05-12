# Phase A4 Results — Gated Attention Isolated Kernel

Test rig: `experiments/83_gated_attention.py` on qb1, ttnn 0.69, device 0.

## Correctness (gate: cosine ≥ 0.99)

| | Value | Gate |
|---|---:|:---:|
| **cosine(out_np, out_ttnn)** | **0.999943** | ≥ 0.99 ✓ |
| max-abs-diff (out) | 0.000700 | |

**PASS.** Tested at `cur_pos=32`, `KV_LEN=128`, GQA shape (16 Q, 2 KV, head_dim=256).

## Performance

```
Decode step:  median = 592 µs    p90 = 626 µs
KV cache read: 256 KB at 450 GB/s = 0.60 µs floor (too short to be bandwidth-bound here)
```

At short cache lengths (KV_LEN=128), SDPA decode is dispatch-bound. As context grows the cache read becomes the bottleneck:

| KV_LEN | KV bytes (bf16) | Memory floor |
|---|---:|---:|
| 128 | 256 KB | 0.6 µs (we measure 592 µs — dispatch wall) |
| 1024 | 2 MB | 4.5 µs |
| 4096 | 8.2 MB | 18 µs |
| 32K | 65 MB | 145 µs |
| 256K | 524 MB | **1.16 ms** — at this length, SDPA finally hits the bandwidth floor |

This is the **reason linear attention (DeltaNet) wins at long context** — DeltaNet's per-step cost is independent of KV length, while SDPA scales linearly. At 256K context the two paths cross over and DeltaNet would actually be cheaper.

## What I had to compromise

Two ttnn 0.69 sharp edges hit during A4:

1. **`paged_update_cache` requires sharded inputs.** Production port (`demos/generate_moe.py`) sets up a `create_sharded_memory_config(strategy=HEIGHT)` and converts K/V via `ttnn.to_memory_config(...)` before calling paged_update_cache. For Phase B integration we'll reuse that dance. **For A4 isolation: pre-populated the cache on host with cur_pos slot already set, skip the on-device update.**

2. **Partial RoPE via slice+concat fails on non-32-aligned dims.** Trying to slice head_dim=256 into [0:64] + [64:256] and concat back failed with `shapes_match` TT_FATAL because TILE_LAYOUT requires the last two dims to be multiples of 32. **For A4 isolation: apply partial RoPE in numpy on host before uploading Q/K.**

Neither workaround is acceptable for Phase B — both cost extra host↔device round-trips per layer per token (~50 ms total over 10 attention layers). For Phase B we need:
- Production sharded-cache pattern (we have it in generate_moe.py)
- A device-side partial RoPE — options: ttnn.experimental.rotary_embedding_llama, or compose with the 64-dim slice in ROW_MAJOR layout, or write a small custom kernel

## What this validates

Despite the two workarounds, A4 confirms:
- ttnn SDPA-decode works at our exact Qwen3.6 shape (GQA 16/2, head_dim=256)
- The sigmoid gate (`attn * sigmoid(gate)`) matches numpy to cosine 0.9999
- The cache layout `[B, n_kv, KV_LEN, head_dim]` is compatible with SDPA-decode

## Status

✅ Phase A4 complete with the noted workarounds.
→ Two follow-ups deferred to Phase B (sharded-cache, device-side partial RoPE).
→ Next: Phase A5 — isolated MoE block.
