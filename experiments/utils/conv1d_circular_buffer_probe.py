#!/usr/bin/env python3
"""
Conv1d circular buffer probe on qb1.

Goal: shave state-management cost (concat + slice ≈ 0.218 ms/layer per
diagnosis memo) off the DeltaNet conv1d block. Diagnosis projected up to
~10.5 ms/tok savings at 48 layers; the question is whether any in-place
ring-buffer scheme that's *available in ttnn today* actually delivers it.

Important constraints from prior probes (memory: feedback_trace_state_threading_works.md):
  - `ttnn.copy(B, A)` between two FULL same-shape tensors works (eager + traced).
  - `ttnn.copy(src, ttnn.slice(buf, ...))` — slice-as-destination — has been
    reported to HANG on Blackhole (#16674 family). We empirically re-test below.
  - `ttnn.kv_cache.update_cache_for_token_(buf, src, pos)` is a fast in-place
    slot-writer that does NOT hit #16674 (validated in update_cache_probe).

Variants:
  V1  Current path: concat + mul + sum + silu + slice
  V2  Ring via update_cache_for_token_ with K pre-rotated weights.
      - buf shape [1, 1, K, CONV_DIM], slot pos = step % K is the newest col.
      - For step `t` the conv input matrix [CONV_DIM, KERNEL] viewed left→right
        oldest→newest is the buf rotated so that physical slot (pos+1) % K is
        leftmost. Equivalently: multiplying physical-order buf with a rotation
        of w_conv yields the same per-row inner product as the canonical path.
      - We pre-upload K versions of w_conv (rotations along the KERNEL axis)
        and select w_conv_rot[pos] each step (Python int → fine for EAGER,
        breaks for TRACE; the probe targets eager).
  V3  Slice-write shift (the originally-pitched "approach B"):
      - Maintain conv_state_buf shape [CONV_DIM, KERNEL].
      - Step:
          (a) shift left:  buf[:, 0:K-1] = buf[:, 1:K]   (slice-write — likely hangs)
          (b) write new col into slot K-1:
                ttnn.copy(mixed_col, ttnn.slice(buf, [:, K-1:K]))   (slice-write)
          (c) mul + sum + silu on the full buf
      - If this works it's the cleanest in-trace pattern. We expect it to hang.
  V4  Floor: mul + sum + silu on a pre-built input (zero state mgmt).
      Defines the maximum achievable savings vs V1 — a sanity bound.

Correctness: V2 must match V1 bit-noisily-equivalently across N=5 simulated
decode steps (cos > 0.99999).

Run on qb1 (uses device 2; devices 0, 1 are reserved):
    cd ~/tt-xla && .venv/bin/python experiments/utils/conv1d_circular_buffer_probe.py
"""
import os
import sys
import time
import traceback

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# Production Qwen3.6-27B DeltaNet shape:
#   N_K=16, N_V=48, K_DIM=128, V_DIM=128
#   CONV_DIM = 2*N_K*K_DIM + N_V*V_DIM = 4096 + 6144 = 10240
#   KERNEL = 4
CONV_DIM = 10240
KERNEL = 4
DTYPE = ttnn.bfloat16
DEVICE_ID = 2
N_STEPS = 5         # math correctness check over multiple simulated steps
N_LATENCY = 200     # iterations per latency measurement
WARMUP = 20


def upload(arr, device, dtype=DTYPE, mem=None):
    t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    return ttnn.from_torch(
        t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT,
        memory_config=mem if mem is not None else ttnn.DRAM_MEMORY_CONFIG,
    )


def sync_time(device, fn, N=N_LATENCY, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1000.0 / N


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def numpy_step(conv_state_np, mixed_col_np, w_np):
    """Reference numpy conv1d step.
    conv_state_np: [CONV_DIM, KERNEL-1] (oldest at col 0, newest at col K-2)
    mixed_col_np:  [CONV_DIM]            (new token's column)
    w_np:          [CONV_DIM, KERNEL]
    Returns (silu(out), new_conv_state)
      out:        [CONV_DIM]
      new_state:  [CONV_DIM, KERNEL-1]
    """
    full = np.concatenate(
        [conv_state_np, mixed_col_np.reshape(CONV_DIM, 1)], axis=-1)  # [CONV_DIM, K]
    prod = full * w_np
    raw = prod.sum(axis=-1)
    out = raw * (1.0 / (1.0 + np.exp(-raw)))
    new_state = full[:, 1:KERNEL]
    return out, new_state


def main():
    print("=" * 78)
    print(f"Conv1d circular buffer probe on qb1 device {DEVICE_ID}")
    print(f"  shape: CONV_DIM={CONV_DIM}  KERNEL={KERNEL}  dtype={DTYPE}")
    print(f"  steps simulated for correctness: {N_STEPS}")
    print(f"  latency: warmup={WARMUP}, N={N_LATENCY}")
    print("=" * 78)

    print(f"\n[1] Opening device {DEVICE_ID}…")
    device = ttnn.open_device(device_id=DEVICE_ID)
    try:
        # Deterministic data ---------------------------------------------------
        rng = np.random.default_rng(7)
        w_np = rng.standard_normal((CONV_DIM, KERNEL)).astype(np.float32) * 0.3
        init_state_np = np.zeros((CONV_DIM, KERNEL - 1), dtype=np.float32)
        # N_STEPS distinct mixed_qkv inputs
        cols_np = [rng.standard_normal((CONV_DIM,)).astype(np.float32) * 0.3
                   for _ in range(N_STEPS)]

        # Numpy gold sequence --------------------------------------------------
        print("\n[2] Numpy reference rollout…")
        gold_outs = []
        gold_states = []
        state = init_state_np.copy()
        for i, col in enumerate(cols_np):
            out, state = numpy_step(state, col, w_np)
            gold_outs.append(out.copy())
            gold_states.append(state.copy())
        print(f"  ✓ {len(gold_outs)} gold outputs computed (||out_0||={np.linalg.norm(gold_outs[0]):.2f})")

        # Upload static tensors -------------------------------------------------
        print("\n[3] Uploading tensors…")
        w_tt = upload(w_np, device)
        # Pre-build mixed_col tensors as [CONV_DIM, 1] (matches 91f layout).
        col_tts = [upload(c.reshape(CONV_DIM, 1), device) for c in cols_np]

        # =====================================================================
        # V1: Baseline path (concat + mul + sum + silu + slice)
        # =====================================================================
        print("\n[4] V1 correctness (current path)…")
        cs_v1 = upload(init_state_np, device)  # [CONV_DIM, KERNEL-1]
        outs_v1 = []
        for i in range(N_STEPS):
            conv_input = ttnn.concat([cs_v1, col_tts[i]], dim=-1)        # [CONV_DIM, K]
            conv_prod = ttnn.mul(conv_input, w_tt)
            conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
            new_state = ttnn.slice(conv_input, [0, 1], [CONV_DIM, KERNEL])
            outs_v1.append(ttnn.to_torch(conv_out).float().cpu().numpy().flatten()[:CONV_DIM])
            cs_v1 = new_state
        for i in range(N_STEPS):
            c = cosine(outs_v1[i], gold_outs[i])
            print(f"  step {i}: cos(V1, gold) = {c:.6f}")

        # =====================================================================
        # V2: update_cache_for_token_ ring buffer with K rotated weights
        # =====================================================================
        # buf shape: [1, 1, KERNEL, CONV_DIM]; we write at pos = step % K.
        # Logical mapping: oldest col is physical slot (pos+1) % K → ... → newest at pos.
        # Equivalently: at step t, after writing slot pos, the physical buf in row-major
        # is [x_{t-K+1+ ((pos+1)%K - 0)}, ..., x_t]. To dot against w_conv (which is
        # indexed by logical position 0..K-1), we need to read buf in a rotated order.
        # We pre-build K rotations of w_conv (each is w_np shifted along KERNEL axis).
        # The correct rotation for write-slot `pos`:
        #   we want logical slot k to read physical slot (pos + 1 + k) mod K.
        #   equivalently: physical slot p corresponds to logical k = (p - pos - 1) mod K.
        #   so the per-physical-slot weight is w_np[:, (p - pos - 1) mod K].
        # That is: w_rot_pos = w_np rolled along axis=1 by -(pos+1).
        #   w_rot_pos[:, p] = w_np[:, (p + (-(pos+1))) mod K] ??? let me re-derive carefully.
        #
        # Let buf[:, p] hold token x_{t-(((pos - p) mod K))}. With pos = t mod K, slot
        # pos holds the newest x_t. Slot (pos-1) mod K holds x_{t-1}. Slot (pos-(K-1))
        # mod K = slot (pos+1) mod K holds the oldest x_{t-K+1}.
        # Logical convention (col 0 = oldest):
        #   logical_col k holds x_{t-(K-1-k)} → slot (pos - (K-1-k)) mod K = (pos+1+k - K) mod K = (pos+1+k) mod K
        # So buf[:, (pos+1+k) mod K] = logical[:, k].
        # The conv output = sum_k logical[:, k] * w_np[:, k].
        # Substitute p = (pos+1+k) mod K  →  k = (p - pos - 1) mod K.
        # conv output = sum_p buf[:, p] * w_np[:, (p-pos-1) mod K]
        # So define w_rot_pos[:, p] = w_np[:, (p-pos-1) mod K], i.e. w_np rolled
        # along axis=1 by -(pos+1) (so that w_rot_pos[:, p] takes from w_np[:, p-(pos+1)]).
        # numpy: w_rot_pos = np.roll(w_np, shift=-(pos+1), axis=1)? Test:
        #   np.roll(a, -s)[i] = a[(i+s) mod N]. We want w_rot[i] = w_np[(i - s) mod N]
        #   with s = pos+1, i.e. shift = +s, i.e. np.roll(w_np, shift=+(pos+1), axis=1).
        # Quick sanity test in head:
        #   pos=0, p=0 → k = (0-1) mod K = K-1. logical col K-1 = newest = x_t = slot pos=0 ✓
        #   pos=0, p=1 → k = 0. logical col 0 = oldest = x_{t-K+1} = slot (pos+1)%K=1 ✓
        #
        # We pre-upload w_rot_tts[pos] for pos in 0..K-1.

        print("\n[5] V2 setup (update_cache ring + pre-rotated weights)…")

        w_rot_nps = [np.roll(w_np, shift=(pos + 1), axis=1) for pos in range(KERNEL)]
        w_rot_tts = [upload(w, device) for w in w_rot_nps]

        # buf shape: [1, 1, KERNEL, CONV_DIM] (KERNEL is the "position" axis for update_cache)
        buf_v2_np = np.zeros((1, 1, KERNEL, CONV_DIM), dtype=np.float32)
        buf_v2_tt = upload(buf_v2_np, device, mem=ttnn.DRAM_MEMORY_CONFIG)

        # Sanity: confirm update_cache_for_token_ is available
        if not hasattr(ttnn, "kv_cache") or not hasattr(ttnn.kv_cache, "update_cache_for_token_"):
            print("  ✗ ttnn.kv_cache.update_cache_for_token_ unavailable — abort V2")
            v2_ok = False
        else:
            v2_ok = True
            outs_v2 = []
            for i in range(N_STEPS):
                pos = i % KERNEL
                # Reshape col [CONV_DIM, 1] to [1, 1, 1, CONV_DIM] for update_cache.
                # NOTE: 91f's mixed_col is [CONV_DIM, 1]; we need [1, 1, 1, CONV_DIM].
                col_reshaped = ttnn.reshape(col_tts[i], [1, 1, 1, CONV_DIM])
                try:
                    ttnn.kv_cache.update_cache_for_token_(buf_v2_tt, col_reshaped, pos)
                except Exception as e:
                    print(f"  ✗ update_cache_for_token_ failed at step {i}: {e}")
                    v2_ok = False
                    break
                # Now compute. buf_v2_tt is [1, 1, KERNEL, CONV_DIM]. Reshape view to
                # [KERNEL, CONV_DIM] then transpose semantically. Simpler: reshape so
                # that the layout matches w_rot_tts (which is [CONV_DIM, KERNEL]).
                # We'll bring buf into [CONV_DIM, KERNEL] via reshape+transpose ops.
                # That re-introduces ops; alternative is to keep w in [KERNEL, CONV_DIM]
                # layout from the start so we can mul-sum without transpose.
                #
                # For probe simplicity: reshape buf to [KERNEL, CONV_DIM] then
                # transpose to [CONV_DIM, KERNEL] each step. We'll measure the V2 op
                # count honestly including this.
                buf_2d = ttnn.reshape(buf_v2_tt, [KERNEL, CONV_DIM])
                buf_T = ttnn.transpose(buf_2d, 0, 1)  # [CONV_DIM, KERNEL]
                prod = ttnn.mul(buf_T, w_rot_tts[pos])
                out = ttnn.silu(ttnn.sum(prod, dim=-1))
                outs_v2.append(ttnn.to_torch(out).float().cpu().numpy().flatten()[:CONV_DIM])

        if v2_ok:
            print("  V2 correctness:")
            for i in range(N_STEPS):
                c = cosine(outs_v2[i], gold_outs[i])
                print(f"    step {i}: cos(V2, gold) = {c:.6f}")

        # =====================================================================
        # V3: slice-write shift (canonical "approach B").
        # This may hang on Blackhole per #16674 family. Wrap in try/except.
        # =====================================================================
        print("\n[6] V3 slice-write shift (may hang on Blackhole)…")
        v3_ok = False
        try:
            buf_v3_np = np.zeros((CONV_DIM, KERNEL), dtype=np.float32)
            buf_v3_tt = upload(buf_v3_np, device)

            # Test ONE step first (no looping until we know it doesn't hang)
            # Shift: buf[:, 0:K-1] = buf[:, 1:K] (read-slice + write-slice)
            src_view = ttnn.slice(buf_v3_tt, [0, 1], [CONV_DIM, KERNEL])  # [CONV_DIM, K-1]
            dst_view = ttnn.slice(buf_v3_tt, [0, 0], [CONV_DIM, KERNEL - 1])  # [CONV_DIM, K-1]
            t0 = time.perf_counter()
            ttnn.copy(src_view, dst_view)
            ttnn.synchronize_device(device)
            shift_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  shift op (single try): {shift_ms:.3f} ms")

            # Write new col into slot K-1
            new_slot = ttnn.slice(buf_v3_tt, [0, KERNEL - 1], [CONV_DIM, KERNEL])  # [CONV_DIM, 1]
            t0 = time.perf_counter()
            ttnn.copy(col_tts[0], new_slot)
            ttnn.synchronize_device(device)
            wr_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  write op (single try): {wr_ms:.3f} ms")

            # Read back & sanity-check the buffer has new_col at slot K-1
            buf_back = ttnn.to_torch(buf_v3_tt).float().cpu().numpy()
            slot_match = np.allclose(buf_back[:, KERNEL - 1], cols_np[0], atol=0.05)
            print(f"  buffer slot K-1 matches new_col: {slot_match}")
            v3_ok = slot_match
        except Exception as e:
            print(f"  V3 FAILED: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()

        # =====================================================================
        # V4: floor — mul+sum+silu only (no state mgmt)
        # =====================================================================
        print("\n[7] V4 floor (mul+sum+silu only on pre-built input)…")
        prebuilt_np = rng.standard_normal((CONV_DIM, KERNEL)).astype(np.float32) * 0.3
        prebuilt_tt = upload(prebuilt_np, device)

        def v4_call():
            prod = ttnn.mul(prebuilt_tt, w_tt)
            return ttnn.silu(ttnn.sum(prod, dim=-1))

        # =====================================================================
        # Latency measurement
        # =====================================================================
        print("\n[8] Latency (steady-state, after warmup)…")

        # V1 timing: simulate one step (using a fixed conv_state buffer + col).
        # We need to give it a fresh state each call to be faithful; but for
        # latency we just measure the steady-state op chain — math correctness
        # was verified above.
        cs_for_timing = upload(init_state_np, device)
        col_for_timing = col_tts[0]

        def v1_call():
            conv_input = ttnn.concat([cs_for_timing, col_for_timing], dim=-1)
            conv_prod = ttnn.mul(conv_input, w_tt)
            conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
            new_state = ttnn.slice(conv_input, [0, 1], [CONV_DIM, KERNEL])
            return conv_out, new_state

        ms_v1 = sync_time(device, v1_call)

        # V2 timing: write to slot 0 every iteration + transpose + mul + sum + silu.
        # This is honest because the per-step cost is the SAME regardless of pos
        # (we just pick rot weight by index — same op count).
        col_for_v2 = ttnn.reshape(col_tts[0], [1, 1, 1, CONV_DIM])
        buf_v2_tt2 = upload(buf_v2_np, device, mem=ttnn.DRAM_MEMORY_CONFIG)
        w_rot0 = w_rot_tts[0]

        def v2_call():
            ttnn.kv_cache.update_cache_for_token_(buf_v2_tt2, col_for_v2, 0)
            buf_2d = ttnn.reshape(buf_v2_tt2, [KERNEL, CONV_DIM])
            buf_T = ttnn.transpose(buf_2d, 0, 1)
            prod = ttnn.mul(buf_T, w_rot0)
            return ttnn.silu(ttnn.sum(prod, dim=-1))

        ms_v2 = sync_time(device, v2_call) if v2_ok else float('nan')

        # V4: floor
        ms_v4 = sync_time(device, v4_call)

        # V3 latency: only measure if it didn't hang. Skip for safety.
        ms_v3 = float('nan')

        print(f"\n  V1 (concat + mul + sum + silu + slice): {ms_v1:.4f} ms")
        if v2_ok:
            delta_v2 = (1 - ms_v2 / ms_v1) * 100
            print(f"  V2 (update_cache + reshape + transpose + mul + sum + silu): "
                  f"{ms_v2:.4f} ms  ({delta_v2:+.1f}% vs V1)")
        else:
            print("  V2 SKIPPED (correctness failed)")
        print(f"  V3 (slice-write shift): {'PASS' if v3_ok else 'FAIL/SKIP'} "
              f"(latency not measured — single-shot only)")
        print(f"  V4 (mul + sum + silu only, floor): {ms_v4:.4f} ms  "
              f"({(1 - ms_v4 / ms_v1) * 100:+.1f}% vs V1)")

        # Per-token projection (48 DeltaNet layers in Qwen3.6-27B)
        N_LAYERS = 48
        print(f"\n  Per-token at {N_LAYERS} DeltaNet layers:")
        print(f"    V1 baseline:               {ms_v1 * N_LAYERS:.2f} ms")
        if v2_ok:
            saved = (ms_v1 - ms_v2) * N_LAYERS
            print(f"    V2 update_cache ring:      {ms_v2 * N_LAYERS:.2f} ms  "
                  f"(saves {saved:+.2f} ms/tok)")
        floor_saved = (ms_v1 - ms_v4) * N_LAYERS
        print(f"    V4 floor (no state mgmt):  {ms_v4 * N_LAYERS:.2f} ms  "
              f"(max possible save: {floor_saved:.2f} ms/tok)")

        # Verdict
        print("\n[9] Verdict")
        if v2_ok and ms_v2 < ms_v1 * 0.97:
            print(f"  V2 WIN — {(1 - ms_v2 / ms_v1) * 100:.1f}% faster than V1 (>=3% gate).")
            print(f"  Recommend integrating V2 into 91f eager path (NOT traced).")
        elif v2_ok:
            print(f"  V2 NO WIN — only {(1 - ms_v2 / ms_v1) * 100:.1f}% faster, below 3% gate.")
            print(f"  Extra reshape+transpose costs cancel the concat+slice savings.")
        else:
            print("  V2 correctness failed — cannot ship.")

    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
