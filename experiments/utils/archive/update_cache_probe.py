#!/usr/bin/env python3
"""
Probe: can ttnn.kv_cache.update_cache_for_token_ replace ttnn.scatter on the
KV cache write path?

Background (research/kernel_research/04_update_cache_reference_op.md):
- ttnn.kv_cache.update_cache_for_token_(cache, input, update_index) is an
  IN-PLACE writer that takes cache [1, B, P, D] + input [1, B, 1, D] and
  writes input into cache[:, :, update_index, :].
- Existing usage in tt-metal/models/experimental/grok/tt/grok_attention.py:225
  uses the [1, B, P, D] layout (no permute, no batch padding).
- Issue #16674 (Blackhole hang) is correlated with SHARDED writer path.
  INTERLEAVED memory config should sidestep it.

If this probe passes, we can drop ttnn.scatter from gated_attn_step entirely:
  - Eliminates 10 ms/tok of pure dispatch overhead (5% of decode time)
  - Eliminates the in-trace ttnn.copy(scatter_out, cache) state-threading
    workaround in C'4 v4 — update_cache mutates in place natively
  - No custom kernel needed

This probe answers:
  1. Does ttnn.kv_cache.update_cache_for_token_ exist + accept our shapes?
  2. Does it mutate cache in place (bit-correct vs numpy scatter reference)?
  3. Does it work with INTERLEAVED memory config on Blackhole (no #16674)?
  4. Latency vs ttnn.scatter at production shape — confirms the perf claim.
  5. Does it work inside a trace (no host write triggers)?

Run on qb2 (device 1 if server is on 0; or just device 0):
    cd ~/tt-xla && TT_DEVICE_ID=0 .venv/bin/python experiments/utils/update_cache_probe.py
"""
import os, sys, time
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

# Production shape for Qwen3.6-27B GQA
N_KV = 4
HEAD_DIM = 256
MAX_POS = 256
N_ITER = 50
N_WARMUP = 5


def _alloc(shape, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, value=0.0,
           random=False, memory_config=None):
    if random:
        arr = np.random.randn(*shape).astype(np.float32)
    else:
        arr = np.full(shape, value, dtype=np.float32)
    kw = dict(dtype=dtype, device=device, layout=layout)
    if memory_config is not None:
        kw["memory_config"] = memory_config
    return ttnn.from_torch(torch.from_numpy(arr), **kw)


def _to_np(t):
    return ttnn.to_torch(t).float().cpu().numpy()


def main():
    device_id = int(os.environ.get("TT_DEVICE_ID", "0"))
    print("=" * 72)
    print(f"Probe: ttnn.kv_cache.update_cache_for_token_  device_id={device_id}")
    print(f"  shape: cache=[1, {N_KV}, {MAX_POS}, {HEAD_DIM}]  src=[1, {N_KV}, 1, {HEAD_DIM}]")
    print("=" * 72)

    device = ttnn.open_device(device_id=device_id)
    try:
        # 1. Verify the op exists + introspect signature
        print("\n[1] Inspect ttnn.kv_cache.update_cache_for_token_:")
        if not hasattr(ttnn, "kv_cache"):
            print("  ✗ ttnn.kv_cache module not exposed")
            return
        if not hasattr(ttnn.kv_cache, "update_cache_for_token_"):
            print("  ✗ ttnn.kv_cache.update_cache_for_token_ not found")
            print(f"  available in ttnn.kv_cache: {[x for x in dir(ttnn.kv_cache) if not x.startswith('_')]}")
            return
        print(f"  ✓ found ttnn.kv_cache.update_cache_for_token_")
        doc = ttnn.kv_cache.update_cache_for_token_.__doc__ or "(no doc)"
        print(f"  doc preview:\n  " + "\n  ".join(doc.strip().split("\n")[:6]))

        # 2. Correctness test: write src to position 17, verify
        print("\n[2] Correctness — INTERLEAVED memory config, write at pos=17:")
        cache_np = np.zeros((1, N_KV, MAX_POS, HEAD_DIM), dtype=np.float32)
        cache_tt = _alloc(cache_np.shape, device, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Force-overwrite with known content for verification
        cache_seed = np.random.randn(*cache_np.shape).astype(np.float32) * 0.1
        cache_init = ttnn.from_torch(torch.from_numpy(cache_seed),
                                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(cache_init, cache_tt)

        src_np = np.random.randn(1, N_KV, 1, HEAD_DIM).astype(np.float32)
        src_tt = ttnn.from_torch(torch.from_numpy(src_np), dtype=ttnn.bfloat16,
                                  device=device, layout=ttnn.TILE_LAYOUT,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)

        try:
            ttnn.kv_cache.update_cache_for_token_(cache_tt, src_tt, 17)
            ttnn.synchronize_device(device)
            print("  ✓ call succeeded, no hang")
        except Exception as e:
            print(f"  ✗ call FAILED: {type(e).__name__}: {str(e)[:200]}")
            return

        cache_back = _to_np(cache_tt)
        # Position 17 should be src
        at_17 = cache_back[0, :, 17, :]   # [N_KV, HEAD_DIM]
        target_17 = src_np[0, :, 0, :]
        diff_17 = np.abs(at_17 - target_17).max()
        print(f"  cache[:, :, 17, :] vs src: max|Δ| = {diff_17:.4e}")
        # Position 16 should still be the seed
        at_16 = cache_back[0, :, 16, :]
        target_16 = cache_seed[0, :, 16, :]
        diff_16 = np.abs(at_16 - target_16).max()
        print(f"  cache[:, :, 16, :] preserved: max|Δ| = {diff_16:.4e}")

        in_place_ok = diff_17 < 0.01 and diff_16 < 0.01
        if in_place_ok:
            print("  ✓ IN-PLACE WRITE CORRECT (target written, neighbours preserved)")
        else:
            print("  ✗ in-place write WRONG — investigate before swapping into model")
            return

        # 3. Sequential writes (mimics real decode pattern)
        print("\n[3] Sequential writes at pos = 0, 32, 64, 100 — accumulation correct?")
        cache_tt2 = _alloc(cache_np.shape, device, dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, value=0.0)
        writes = {}
        for pos in [0, 32, 64, 100]:
            row = np.random.randn(1, N_KV, 1, HEAD_DIM).astype(np.float32)
            src_x = ttnn.from_torch(torch.from_numpy(row), dtype=ttnn.bfloat16,
                                     device=device, layout=ttnn.TILE_LAYOUT,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.kv_cache.update_cache_for_token_(cache_tt2, src_x, pos)
            writes[pos] = row[0, :, 0, :]
        ttnn.synchronize_device(device)
        back = _to_np(cache_tt2)
        all_ok = True
        for pos, expected in writes.items():
            d = np.abs(back[0, :, pos, :] - expected).max()
            status = "✓" if d < 0.01 else "✗"
            print(f"    pos={pos:3d}: max|Δ| = {d:.4e}  {status}")
            if d >= 0.01:
                all_ok = False
        if all_ok:
            print("  ✓ all 4 sequential writes correct")
        else:
            print("  ✗ accumulation broken")
            return

        # 4. Perf vs ttnn.scatter at production shape
        print("\n[4] Perf vs ttnn.scatter at production shape:")
        cache_a = _alloc(cache_np.shape, device, dtype=ttnn.bfloat16,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cache_b = _alloc(cache_np.shape, device, dtype=ttnn.bfloat16,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
        src_a = _alloc((1, N_KV, 1, HEAD_DIM), device, dtype=ttnn.bfloat16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
        idx_for_scatter = ttnn.from_torch(
            torch.from_numpy(np.zeros((1, N_KV, 1, HEAD_DIM), dtype=np.int32)),
            dtype=ttnn.int32, device=device, layout=ttnn.TILE_LAYOUT)

        # update_cache_for_token_ timing
        for _ in range(N_WARMUP):
            ttnn.kv_cache.update_cache_for_token_(cache_a, src_a, 0)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for i in range(N_ITER):
            ttnn.kv_cache.update_cache_for_token_(cache_a, src_a, i % MAX_POS)
        ttnn.synchronize_device(device)
        update_cache_ms = (time.perf_counter() - t0) * 1000.0 / N_ITER

        # ttnn.scatter timing for comparison
        for _ in range(N_WARMUP):
            _ = ttnn.scatter(cache_b, dim=2, index=idx_for_scatter, src=src_a)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for i in range(N_ITER):
            _ = ttnn.scatter(cache_b, dim=2, index=idx_for_scatter, src=src_a)
        ttnn.synchronize_device(device)
        scatter_ms = (time.perf_counter() - t0) * 1000.0 / N_ITER

        print(f"  update_cache_for_token_: {update_cache_ms:.3f} ms")
        print(f"  ttnn.scatter:            {scatter_ms:.3f} ms")
        speedup = scatter_ms / update_cache_ms if update_cache_ms > 0 else float("inf")
        print(f"  speedup:                 {speedup:.2f}x")
        savings = (scatter_ms - update_cache_ms) * 32  # 32 KV writes per decode step
        print(f"  Per-token decode savings (32 writes): {savings:.2f} ms/tok")

        # 5. Inside-trace test
        print("\n[5] Inside a trace — does update_cache_for_token_ work?")
        # Pre-allocate + WARMUP this exact op so JIT binaries are uploaded.
        cache_tr = _alloc(cache_np.shape, device, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        src_tr = _alloc((1, N_KV, 1, HEAD_DIM), device, dtype=ttnn.bfloat16,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.kv_cache.update_cache_for_token_(cache_tr, src_tr, 0)
        ttnn.synchronize_device(device)

        try:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            ttnn.kv_cache.update_cache_for_token_(cache_tr, src_tr, 5)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            print(f"  ✓ trace captured, tid={tid}")
            # Execute once
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            print(f"  ✓ execute_trace replay succeeded")
            ttnn.release_trace(device, tid)
        except Exception as e:
            print(f"  ✗ trace FAILED: {type(e).__name__}: {str(e)[:200]}")
            return

        print("\n" + "=" * 72)
        print("VERDICT")
        print("=" * 72)
        print(f"✓ ttnn.kv_cache.update_cache_for_token_ works on Blackhole")
        print(f"✓ INTERLEAVED memory config — no #16674 hang")
        print(f"✓ In-place mutation correct (single + sequential)")
        print(f"✓ Trace-compatible (no host write blocker)")
        print(f"")
        print(f"PERF: {update_cache_ms:.3f} ms vs scatter {scatter_ms:.3f} ms")
        print(f"      = {speedup:.2f}x faster, saves {savings:.2f} ms/tok in 27B decode")
        print(f"")
        print(f"NEXT: swap ttnn.scatter -> ttnn.kv_cache.update_cache_for_token_ in")
        print(f"      gated_attn_step_ondevice / gated_attn_step_ondevice_traced.")
        print(f"      The in-trace ttnn.copy state-threading in C'4 v4 becomes")
        print(f"      redundant (update_cache is in-place natively).")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
