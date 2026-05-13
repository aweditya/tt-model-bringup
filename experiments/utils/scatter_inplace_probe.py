#!/usr/bin/env python3
"""
Probe: can we get zero-copy scatter on Blackhole via the output_tensor= kwarg?

Background: ttnn.scatter is a functional op — it returns a NEW tensor that
holds the cache-with-one-row-updated. Our C'4 v4 design uses
`ttnn.copy(scatter_out, cache_in)` after every scatter to commit the new
state back to the input buffer. That copy is the friction.

Many ttnn ops accept an `output_tensor=` kwarg (or similar) that lets the
caller designate the destination buffer, eliminating the implicit allocation.
If ttnn.scatter supports passing the INPUT buffer as the output buffer, we
get scatter-in-place — no copy, no extra allocation.

This probe answers:
  1. Does ttnn.scatter accept an output kwarg at all?
  2. If yes, does passing cache as both input AND output mutate cache in place?
  3. Does this work inside a trace?
  4. Does it persist state correctly across execute_trace replays?

If 4 passes -> C'4 v4 design simplifies: drop all the in-trace ttnn.copy
calls, slash ~7 ms/tok of overhead, and the state-threading is free.

Run on qb2 (device 0 — kernel profiler exits cleanly so should be free):
    cd ~/tt-xla && .venv/bin/python experiments/utils/scatter_inplace_probe.py
"""
import os, sys, time, signal, inspect
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def _timeout(signum, frame):
    raise TimeoutError("op > 60s")


def _cos(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 64)
    print("Probe: scatter output_tensor= kwarg, in-place semantics, trace")
    print("=" * 64)

    print("\n[0] Inspect ttnn.scatter signature:")
    try:
        sig = inspect.signature(ttnn.scatter)
        print(f"  signature: {sig}")
    except (TypeError, ValueError) as e:
        # nanobind ops typically don't expose Python signatures via inspect
        print(f"  inspect.signature failed: {e}")
        print(f"  doc:")
        print("  " + "\n  ".join((ttnn.scatter.__doc__ or "(no doc)").split("\n")[:30]))

    MAX_POS = 8
    HEAD_DIM = 32
    N_KV = 1
    device_id = int(os.environ.get("TT_DEVICE_ID", "0"))
    print(f"\n  device_id = {device_id}")
    device = ttnn.open_device(device_id=device_id)
    try:
        cache_np = np.zeros((1, N_KV, MAX_POS, HEAD_DIM), dtype=np.float32)
        cache_tt = ttnn.from_torch(torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
        src_np = np.zeros((1, N_KV, 1, HEAD_DIM), dtype=np.float32)
        src_tt = ttnn.from_torch(torch.from_numpy(src_np), dtype=ttnn.bfloat16,
                                  device=device, layout=ttnn.TILE_LAYOUT)
        idx_np = np.zeros((1, N_KV, 1, HEAD_DIM), dtype=np.int32)
        idx_tt = ttnn.from_torch(torch.from_numpy(idx_np), dtype=ttnn.int32,
                                  device=device, layout=ttnn.TILE_LAYOUT)

        row_A = np.arange(HEAD_DIM, dtype=np.float32) * 0.1
        src_temp_A = ttnn.from_torch(torch.from_numpy(
            row_A.reshape(1, N_KV, 1, HEAD_DIM)),
            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        idx_temp_A = ttnn.from_torch(torch.from_numpy(
            np.full((1, N_KV, 1, HEAD_DIM), 2, dtype=np.int32)),
            dtype=ttnn.int32, device=device, layout=ttnn.TILE_LAYOUT)

        signal.signal(signal.SIGALRM, _timeout)
        cache_addr_before = id(cache_tt)
        print(f"\n[1] EAGER probe: ttnn.scatter(...) with output_tensor=cache_tt")
        print(f"  cache id before: {cache_addr_before}")

        # Update src + idx
        ttnn.copy(src_temp_A, src_tt)
        ttnn.copy(idx_temp_A, idx_tt)

        # Attempt various kwarg names ttnn might support
        kwarg_tried = []
        attempted = False
        # Try several plausible kwarg names — ttnn ops use varying conventions
        for kwarg in ("output_tensor", "out", "output", "dst", "dst_tensor"):
            try:
                signal.alarm(60)
                result = ttnn.scatter(cache_tt, dim=2, index=idx_tt, src=src_tt,
                                       **{kwarg: cache_tt})
                ttnn.synchronize_device(device)
                signal.alarm(0)
                kwarg_tried.append((kwarg, "ACCEPTED", result))
                attempted = True
                print(f"  ✓ kwarg '{kwarg}=cache_tt' accepted; result id = {id(result)}")
                break
            except TimeoutError:
                signal.alarm(0)
                kwarg_tried.append((kwarg, "HANG"))
                print(f"  ✗ kwarg '{kwarg}=cache_tt' hung the call")
            except TypeError as e:
                signal.alarm(0)
                kwarg_tried.append((kwarg, f"rejected: {str(e)[:80]}"))
            except Exception as e:
                signal.alarm(0)
                kwarg_tried.append((kwarg, f"err: {type(e).__name__}: {str(e)[:80]}"))

        if not attempted:
            print(f"  ✗ NONE of the kwargs were accepted by ttnn.scatter:")
            for k, status in kwarg_tried:
                print(f"      {k:18s} -> {status}")
            print(f"  -> ttnn.scatter likely has no output-buffer kwarg.")
            print(f"  -> Plan B: probe ttnn.experimental.scatter, ttnn.scatter_,")
            print(f"     or test `cache_tt = ttnn.scatter(cache_tt, ...)` does")
            print(f"     in-place — i.e., the input tensor is mutated when reassigned.")
            # Plan B: does scatter mutate input in place even WITHOUT out= kwarg?
            # Per docs: scatter is FUNCTIONAL = no. But let's confirm.
            print(f"\n[1b] Plan B: confirm scatter is functional (input UNTOUCHED)")
            row_B = (np.arange(HEAD_DIM, dtype=np.float32) * 0.1 + 100.0)
            src_temp_B = ttnn.from_torch(torch.from_numpy(
                row_B.reshape(1, N_KV, 1, HEAD_DIM)),
                dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            ttnn.copy(src_temp_B, src_tt)
            result2 = ttnn.scatter(cache_tt, dim=2, index=idx_tt, src=src_tt)
            ttnn.synchronize_device(device)
            cache_back = ttnn.to_torch(cache_tt).float().cpu().numpy()
            result2_back = ttnn.to_torch(result2).float().cpu().numpy()
            print(f"  cache[2] (input)   first 3: {cache_back[0, 0, 2, :3]}")
            print(f"  result[2] (output) first 3: {result2_back[0, 0, 2, :3]}")
            print(f"  expected row_B[:3]:        {row_B[:3]}")
            input_changed = abs(cache_back[0, 0, 2, :].sum()) > 0.1
            if input_changed:
                print(f"  !!! INPUT WAS MUTATED — scatter is NOT purely functional !!!")
                print(f"      If true, the current C'4 v4 design works WITHOUT explicit copies.")
            else:
                print(f"  input unchanged — scatter is functional, copy IS needed.")
            return

        # If we got here, some kwarg was accepted. Verify the cache was mutated.
        ttnn.synchronize_device(device)
        cache_back = ttnn.to_torch(cache_tt).float().cpu().numpy()
        cos = _cos(cache_back[0, 0, 2, :], row_A)
        max_other = float(np.abs(np.delete(cache_back[0, 0], 2, axis=0)).max())
        print(f"\n  After scatter with out=cache:")
        print(f"  cache[2] vs row_A: cos={cos:.6f}")
        print(f"  other positions max|·|: {max_other:.4e}")
        in_place_ok = cos > 0.99 and max_other < 0.01
        if not in_place_ok:
            print(f"  ✗ scatter accepted the kwarg but the cache was NOT updated in place.")
            return
        print(f"  ✓ scatter mutated cache in place via the {kwarg_tried[-1][0]} kwarg")

        # Trace test
        accepted_kwarg = next(k for k, s, *_ in kwarg_tried if s == "ACCEPTED")
        print(f"\n[2] TRACE probe: capture scatter with {accepted_kwarg}=cache, replay 2x")
        # Reset cache to zero
        cache_zero = ttnn.from_torch(torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
                                      device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.copy(cache_zero, cache_tt)
        ttnn.copy(src_temp_A, src_tt)
        ttnn.copy(idx_temp_A, idx_tt)
        ttnn.synchronize_device(device)

        tid = ttnn.begin_trace_capture(device, cq_id=0)
        ttnn.scatter(cache_tt, dim=2, index=idx_tt, src=src_tt,
                     **{accepted_kwarg: cache_tt})
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        print(f"  trace captured tid={tid}")

        # Reset cache to zero (in place — no fresh alloc)
        ttnn.copy(cache_zero, cache_tt)
        ttnn.synchronize_device(device)

        # Replay 1: src=row_A, idx=2
        ttnn.copy(src_temp_A, src_tt)
        ttnn.copy(idx_temp_A, idx_tt)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        back1 = ttnn.to_torch(cache_tt).float().cpu().numpy()
        cos1 = _cos(back1[0, 0, 2, :], row_A)
        print(f"\n  Replay 1: cache[2] vs row_A cos={cos1:.6f}  "
              f"others max|·|={float(np.abs(np.delete(back1[0,0], 2, axis=0)).max()):.4e}")

        # Replay 2: src=row_B at idx=5 — should persist row_A at idx=2 AND add row_B at idx=5
        row_B = (-np.arange(HEAD_DIM, dtype=np.float32) * 0.1 + 5.0)
        src_temp_B = ttnn.from_torch(torch.from_numpy(
            row_B.reshape(1, N_KV, 1, HEAD_DIM)),
            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        idx_temp_B = ttnn.from_torch(torch.from_numpy(
            np.full((1, N_KV, 1, HEAD_DIM), 5, dtype=np.int32)),
            dtype=ttnn.int32, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.copy(src_temp_B, src_tt)
        ttnn.copy(idx_temp_B, idx_tt)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        back2 = ttnn.to_torch(cache_tt).float().cpu().numpy()
        cos2_at_2 = _cos(back2[0, 0, 2, :], row_A)
        cos2_at_5 = _cos(back2[0, 0, 5, :], row_B)
        others = np.stack([back2[0, 0, p, :] for p in range(MAX_POS) if p not in (2, 5)])
        max_other_after = float(np.abs(others).max())
        print(f"\n  Replay 2:")
        print(f"    cache[2] vs row_A (persistence): cos={cos2_at_2:.6f}")
        print(f"    cache[5] vs row_B (new write):   cos={cos2_at_5:.6f}")
        print(f"    others max|·|: {max_other_after:.4e}")

        print("\n" + "=" * 64)
        print("VERDICT")
        print("=" * 64)
        if cos1 > 0.99 and cos2_at_2 > 0.99 and cos2_at_5 > 0.99 and max_other_after < 0.01:
            print(f"✓ scatter-in-place WORKS via {accepted_kwarg}=cache_tt.")
            print(f"  Trace captures correctly; state persists across replays.")
            print(f"  C'4 v4 design can DROP all in-trace ttnn.copy calls.")
            print(f"  Expected saving: ~7 ms/tok (~3% decode).")
        else:
            print(f"✗ Scatter-in-place did NOT produce correct state across replays.")
            print(f"  Stick with the explicit ttnn.copy path.")

        ttnn.release_trace(device, tid)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
