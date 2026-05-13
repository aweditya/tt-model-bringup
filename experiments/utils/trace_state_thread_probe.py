#!/usr/bin/env python3
"""
Probe: does ttnn.copy(scatter_out, cache) INSIDE a trace correctly
thread state across multiple execute_trace calls?

This is the make-or-break test for C'4 trace capture. C'4 v3 proved
the trace mechanism is bit-correct for one-shot replay but multi-step
fails because C'1's functional scatter doesn't thread the new cache
back as the next replay's input.

The proposed fix: include `ttnn.copy(scatter_out, cache_in)` inside
the trace itself. Each execute_trace ends by committing the new cache
state back to the input address. Next execute_trace then reads from
the updated input.

This probe tests EXACTLY that pattern in isolation, at production-like
shape (mini cache + scatter on dim=2 with src/index pre-allocated).

If two sequential execute_trace calls with different (src, index)
buffer contents accumulate writes correctly in the cache:
  → PASS: C'4 trace path is unblocked
If the second call's writes overwrite or fail to land:
  → FAIL: need a different threading approach

The big risk to watch: ttnn.copy(B, A) inside a trace may hit the same
Blackhole writer-family hang we hit with slice-as-destination. Hard
timeout 60s detects that.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/utils/trace_state_thread_probe.py
"""
import sys
import time
import signal
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def timeout_handler(signum, frame):
    raise TimeoutError("op took > 60 s")


def _cosine(a, b):
    a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    MAX_POS = 8        # tiny cache
    HEAD_DIM = 32
    N_KV = 1

    print("=" * 64)
    print("Probe: in-trace ttnn.copy state-threading")
    print(f"  cache=[1, {N_KV}, {MAX_POS}, {HEAD_DIM}], scatter dim=2")
    print("=" * 64)

    device = ttnn.open_device(device_id=0)
    try:
        # Pre-allocate ALL buffers BEFORE begin_trace_capture
        # (the design doc rule: no allocation during trace)
        cache_np = np.zeros((1, N_KV, MAX_POS, HEAD_DIM), dtype=np.float32)
        cache_tt = ttnn.from_torch(torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
        src_np = np.zeros((1, N_KV, 1, HEAD_DIM), dtype=np.float32)
        src_tt = ttnn.from_torch(torch.from_numpy(src_np), dtype=ttnn.bfloat16,
                                  device=device, layout=ttnn.TILE_LAYOUT)
        idx_np = np.zeros((1, N_KV, 1, HEAD_DIM), dtype=np.int32)
        idx_tt = ttnn.from_torch(torch.from_numpy(idx_np), dtype=ttnn.int32,
                                  device=device, layout=ttnn.TILE_LAYOUT)
        print(f"  cache shape: {tuple(cache_tt.shape)}")
        print(f"  src   shape: {tuple(src_tt.shape)}")
        print(f"  idx   shape: {tuple(idx_tt.shape)}")

        # Smoke: just the copy at scatter output (NO trace yet)
        print("\n[Smoke] Test 1 step EAGERLY first (no trace):")
        row_A = np.arange(HEAD_DIM, dtype=np.float32) * 0.1   # distinctive
        idx_value_A = 2
        src_np_new = row_A.reshape(1, N_KV, 1, HEAD_DIM)
        idx_np_new = np.full((1, N_KV, 1, HEAD_DIM), idx_value_A, dtype=np.int32)
        # Update src and idx via copy_host_to_device_tensor
        src_temp = ttnn.from_torch(torch.from_numpy(src_np_new), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
        idx_temp = ttnn.from_torch(torch.from_numpy(idx_np_new), dtype=ttnn.int32,
                                    device=device, layout=ttnn.TILE_LAYOUT)
        try:
            ttnn.copy(src_temp, src_tt)
            ttnn.copy(idx_temp, idx_tt)
        except Exception as e:
            print(f"  copy_host_to_device_tensor analog (ttnn.copy) failed: {e}")
            # Fall back to direct assignment via from_torch
            src_tt = src_temp
            idx_tt = idx_temp
        scatter_out = ttnn.scatter(cache_tt, dim=2, index=idx_tt, src=src_tt)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        try:
            ttnn.copy(scatter_out, cache_tt)
            ttnn.synchronize_device(device)
            signal.alarm(0)
            print("  ✓ eager scatter+copy completed")
        except TimeoutError:
            signal.alarm(0)
            print("  ✗ HANG: copy(scatter_out, cache_tt) hangs even without trace")
            return
        except Exception as e:
            signal.alarm(0)
            print(f"  ✗ FAILED: {str(e)[:200]}")
            return

        back = ttnn.to_torch(cache_tt).float().cpu().numpy()
        cos = _cosine(back[0, 0, idx_value_A, :], row_A)
        print(f"  cache[{idx_value_A}] vs row_A: cos={cos:.6f}")
        if cos > 0.99:
            print("  ✓ Eager pattern works. Proceeding to trace test.")
        else:
            print("  ✗ Eager pattern doesn't even work. Trace test aborted.")
            return

        # Now reset cache and try the trace pattern
        # Reset cache to zeros
        cache_zero = ttnn.from_torch(torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
                                      device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.copy(cache_zero, cache_tt)
        ttnn.synchronize_device(device)

        print("\n[Trace] Capture + 2 replays with different (src, idx):")
        # Update src/idx with row_A, idx=2 before capture (warmup state)
        ttnn.copy(src_temp, src_tt)
        ttnn.copy(idx_temp, idx_tt)

        # Capture: the program is scatter → copy
        print("  begin_trace_capture...")
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        scatter_out_tr = ttnn.scatter(cache_tt, dim=2, index=idx_tt, src=src_tt)
        ttnn.copy(scatter_out_tr, cache_tt)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        print(f"  trace captured. tid={tid}")

        # Reset cache to zeros again to start clean
        ttnn.copy(cache_zero, cache_tt)
        ttnn.synchronize_device(device)

        # Replay 1: src=row_A, idx=2
        print("\n  Replay 1: src=row_A, idx=2")
        ttnn.copy(src_temp, src_tt)
        ttnn.copy(idx_temp, idx_tt)
        signal.alarm(60)
        try:
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            signal.alarm(0)
            print("    ✓ execute_trace #1 completed")
        except TimeoutError:
            signal.alarm(0)
            print("    ✗ HANG inside execute_trace #1")
            return
        back1 = ttnn.to_torch(cache_tt).float().cpu().numpy()
        cos_at_2 = _cosine(back1[0, 0, 2, :], row_A)
        other_max = float(np.abs(np.delete(back1[0, 0], 2, axis=0)).max())
        print(f"    cache[2] vs row_A: cos={cos_at_2:.6f}")
        print(f"    other positions max|·|: {other_max:.4e}")

        # Replay 2: src=row_B (different), idx=5 (different position)
        print("\n  Replay 2: src=row_B, idx=5")
        row_B = (-np.arange(HEAD_DIM, dtype=np.float32) * 0.1 + 5.0)
        src_np_B = row_B.reshape(1, N_KV, 1, HEAD_DIM)
        idx_np_B = np.full((1, N_KV, 1, HEAD_DIM), 5, dtype=np.int32)
        src_temp_B = ttnn.from_torch(torch.from_numpy(src_np_B), dtype=ttnn.bfloat16,
                                      device=device, layout=ttnn.TILE_LAYOUT)
        idx_temp_B = ttnn.from_torch(torch.from_numpy(idx_np_B), dtype=ttnn.int32,
                                      device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.copy(src_temp_B, src_tt)
        ttnn.copy(idx_temp_B, idx_tt)
        signal.alarm(60)
        try:
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            signal.alarm(0)
            print("    ✓ execute_trace #2 completed")
        except TimeoutError:
            signal.alarm(0)
            print("    ✗ HANG inside execute_trace #2")
            return

        back2 = ttnn.to_torch(cache_tt).float().cpu().numpy()
        cos_at_2_after = _cosine(back2[0, 0, 2, :], row_A)   # should STILL be row_A
        cos_at_5 = _cosine(back2[0, 0, 5, :], row_B)         # should be row_B
        others = [back2[0, 0, p, :] for p in range(MAX_POS) if p not in (2, 5)]
        other_max_after = float(np.abs(np.stack(others)).max())
        print(f"    cache[2] vs row_A (persistence): cos={cos_at_2_after:.6f}")
        print(f"    cache[5] vs row_B (new write):   cos={cos_at_5:.6f}")
        print(f"    others (2 and 5 excluded) max|·|: {other_max_after:.4e}")

        print("\n" + "=" * 64)
        print("VERDICT")
        print("=" * 64)
        passed = (cos_at_2_after > 0.99 and cos_at_5 > 0.99 and other_max_after < 0.01)
        if passed:
            print("✓ IN-TRACE STATE THREADING WORKS.")
            print("  Two execute_trace calls accumulated writes correctly.")
            print("  C'4 multi-step path is UNBLOCKED.")
        else:
            print("✗ In-trace threading produced wrong state.")
            print("  - Did position 2's write persist after replay 2?")
            print("  - Did position 5 get row_B?")
            print("  - Did anything else get corrupted?")
            print("  Need to investigate; may need different approach.")

    finally:
        try:
            ttnn.release_trace(device, tid)
        except Exception:
            pass
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
