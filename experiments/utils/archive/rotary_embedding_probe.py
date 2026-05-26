#!/usr/bin/env python3
"""
Probe ttnn.experimental.rotary_embedding for C'3 native RoPE swap.

Current implementation in 91f.gated_attn_step_ondevice (apply_partial_rope)
does the rotation manually:
  - slice rotary half + passthrough half
  - rotate_half via slice + neg + concat
  - mul by cos and sin tables
  - add then concat with passthrough

C'3 hypothesis: ttnn.experimental.rotary_embedding does this in one fused op.
Prior data (feedback_native_rope.md): 2.6× speedup for full-rotary case.

We have PARTIAL rotary (first 64 of 256 dims rotated). The native op might
or might not support partial — this probe finds out.

Questions to answer:
1. Signature of ttnn.experimental.rotary_embedding — what args?
2. Does it accept a tensor shaped [n_heads, head_dim] or require [batch, n_heads, seq, head_dim]?
3. Does it support partial rotary? (some impls take an explicit "rotary_dim" arg)
4. What dtypes are required for input / cos / sin tables?

Run on qb1 (qb2 is busy with perf C'1):
    cd ~/tt-xla && .venv/bin/python experiments/utils/rotary_embedding_probe.py
"""
import sys
import inspect
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

def main():
    print("=" * 64)
    print("Probe: ttnn.experimental.rotary_embedding")
    print("=" * 64)

    fn = getattr(getattr(ttnn, "experimental", None), "rotary_embedding", None)
    if fn is None:
        print("  rotary_embedding NOT FOUND in ttnn.experimental")
        for name in sorted(dir(ttnn.experimental)):
            if "rot" in name.lower() or "rope" in name.lower():
                print(f"  related: ttnn.experimental.{name}")
        return

    print(f"\nttnn.experimental.rotary_embedding")
    try:
        sig = inspect.signature(fn)
        print(f"  signature: {sig}")
    except Exception as e:
        print(f"  (no signature: {e})")
    doc = inspect.getdoc(fn) or ""
    if doc:
        print(f"  doc:")
        for line in doc.split("\n")[:30]:
            print(f"    | {line}")

    # Also check ttnn.rotary_embedding (non-experimental)
    fn2 = getattr(ttnn, "rotary_embedding", None)
    if fn2 is not None and fn2 is not fn:
        print(f"\nttnn.rotary_embedding (non-experimental)")
        try:
            print(f"  signature: {inspect.signature(fn2)}")
        except Exception:
            pass

    # Try a minimal invocation at our Qwen3.6-27B shapes
    print("\n" + "=" * 64)
    print("Try smallest valid call — does it run?")
    print("=" * 64)
    HEAD_DIM = 256
    ROTARY_DIM = 64   # partial rotary: first 64 of 256
    N_Q = 32          # n_q_heads
    SEQ = 1           # decode-step

    device = ttnn.open_device(device_id=0)
    try:
        # Build full-rotary-style input (try shape variants)
        # Variant 1: [1, n_heads, 1, head_dim]
        for label, shape in [
            ("[1, N_Q, 1, HEAD_DIM]", (1, N_Q, 1, HEAD_DIM)),
            ("[N_Q, HEAD_DIM]",      (N_Q, HEAD_DIM)),
            ("[1, 1, N_Q, HEAD_DIM]", (1, 1, N_Q, HEAD_DIM)),
        ]:
            x_np = np.random.randn(*shape).astype(np.float32)
            x_tt = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            cos_np = np.random.randn(1, HEAD_DIM).astype(np.float32)
            sin_np = np.random.randn(1, HEAD_DIM).astype(np.float32)
            cos_tt = ttnn.from_torch(torch.from_numpy(cos_np), dtype=ttnn.bfloat16,
                                      device=device, layout=ttnn.TILE_LAYOUT)
            sin_tt = ttnn.from_torch(torch.from_numpy(sin_np), dtype=ttnn.bfloat16,
                                      device=device, layout=ttnn.TILE_LAYOUT)
            print(f"\n  shape attempt: {label}")
            try:
                out = fn(x_tt, cos_tt, sin_tt)
                print(f"    SUCCESS: out shape={tuple(out.shape)}")
            except Exception as e:
                msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                print(f"    failed: {msg[:120]}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
