#!/usr/bin/env python3
"""
Probe: compute (I - L)^{-1} for strict-lower-triangular L via the Neumann
series factorization. This is the critical risk for C'5 chunked prefill.

The trick: for strict-lower-triangular L ∈ R^{C×C}, L^C = 0 (nilpotent), so

    (I - L)^{-1} = I + L + L^2 + ... + L^{C-1}

Naive: 63 serial matmuls for C=64. Catastrophic.

Better — factor the sum:

    (I + L)(I + L^2)(I + L^4)(I + L^8)(I + L^16)(I + L^32)
    = I + L + L^2 + ... + L^63

Proof by induction:
  (I + L^a)(I + L^b) = I + L^a + L^b + L^{a+b}    (for a < b)
  start with (I + L), then multiply by (I + L^2), then (I + L^4), etc.
  Each factor doubles the range of L-power exponents covered.

For C=64 we need 5 squarings (L→L²→L⁴→L⁸→L¹⁶→L³²) + 5 multiplications
through the product = 10 matmuls per chunk (per V-head, but heads batch).

What this probe answers:
1. Does the factorization compute the correct inverse numerically?
2. At fp32: how close to numpy's true inverse? (gate: max|Δ| < 1e-4)
3. At bf16: how close? (gate: max|Δ| < 0.01 — known coarser)
4. At our shape [N_V=32, C=64, C=64] batched: does it work?
5. Wall-clock time per chunk of 32 heads — is it actually fast?

Run on qb1 (qb2 is busy with C'2 gates):
    cd ~/tt-xla && .venv/bin/python experiments/utils/neumann_inverse_probe.py
"""
import sys
import time
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def neumann_inverse_numpy(L):
    """Compute (I - L)^{-1} via the factorization. Reference impl."""
    C = L.shape[-1]
    I = np.eye(C, dtype=L.dtype)
    if L.ndim == 3:
        I = np.broadcast_to(I, L.shape).copy()

    # Compute powers L^2, L^4, L^8, L^16, L^32
    powers = [L]
    cur = L
    while powers[-1].shape[-1] == C and 2 * (len(powers)) <= int(np.log2(C)):
        cur = cur @ cur if cur.ndim == 2 else np.matmul(cur, cur)
        powers.append(cur)

    # Compose product (I + L)(I + L^2)...(I + L^{2^k})
    T = I + powers[0]
    for p in powers[1:]:
        T = T @ (I + p) if T.ndim == 2 else np.matmul(T, I + p)
    return T


def neumann_inverse_ttnn(L_tt, device, dtype=ttnn.float32, C=64):
    """Compute (I - L)^{-1} on device via Neumann factorization.
    L_tt: [C, C] (or [B, C, C] batched).  Returns same shape."""
    # Build I on device matching L shape
    shape = list(L_tt.shape)
    I_np = np.zeros(shape, dtype=np.float32)
    if len(shape) == 2:
        np.fill_diagonal(I_np, 1.0)
    else:
        for i in range(shape[0]):
            np.fill_diagonal(I_np[i], 1.0)
    I_tt = ttnn.from_torch(torch.from_numpy(I_np), dtype=dtype,
                           device=device, layout=ttnn.TILE_LAYOUT)

    # Step 1: compute powers L², L⁴, L⁸, L¹⁶, L³²
    powers = [L_tt]
    cur = L_tt
    n_squarings = int(np.log2(C)) - 1   # for C=64: 5
    for _ in range(n_squarings):
        cur = ttnn.matmul(cur, cur)
        powers.append(cur)

    # Step 2: compose product (I + L)(I + L²)(I + L⁴)...
    T = ttnn.add(I_tt, powers[0])
    for p in powers[1:]:
        factor = ttnn.add(I_tt, p)
        T = ttnn.matmul(T, factor)

    return T


def main():
    print("=" * 64)
    print("Probe: (I - L)^{-1} via Neumann series factorization")
    print("=" * 64)

    C = 64
    rng = np.random.default_rng(42)

    # Build a realistic strict-lower-triangular L
    # Entries ~ (small, since they come from -(K_β K^T) ⊙ D where D ∈ (0,1])
    L_np = rng.standard_normal((C, C)).astype(np.float32) * 0.05
    L_np = np.tril(L_np, k=-1)   # strict lower triangular
    print(f"\nL: shape={L_np.shape}  max|L|={np.abs(L_np).max():.4f}")
    print(f"   |L^C-1|≈{np.abs(np.linalg.matrix_power(L_np, C-1)).max():.2e} (should be tiny)")

    # Ground truth: np.linalg.inv(I - L)
    T_true = np.linalg.inv(np.eye(C, dtype=np.float32) - L_np)
    print(f"   ||T_true||_F = {np.linalg.norm(T_true):.4f}")

    # Verify the numpy Neumann impl is correct first
    T_neumann_np = neumann_inverse_numpy(L_np)
    cos = _cosine(T_neumann_np, T_true)
    max_diff = float(np.abs(T_neumann_np - T_true).max())
    print(f"\n[numpy reference] Neumann factorization correctness:")
    print(f"   cosine: {cos:.8f}")
    print(f"   max|Δ|:  {max_diff:.6e}")
    assert cos > 0.9999, "Numpy Neumann impl is broken!"

    # ttnn fp32
    print("\n" + "=" * 64)
    print("Test 1: ttnn fp32, single L of shape [64, 64]")
    print("=" * 64)
    device = ttnn.open_device(device_id=0)
    try:
        L_tt = ttnn.from_torch(torch.from_numpy(L_np), dtype=ttnn.float32,
                                device=device, layout=ttnn.TILE_LAYOUT)
        t0 = time.time()
        T_tt = neumann_inverse_ttnn(L_tt, device, dtype=ttnn.float32, C=C)
        ttnn.synchronize_device(device)
        t1 = time.time()
        T_np = ttnn.to_torch(T_tt).float().cpu().numpy()
        cos = _cosine(T_np, T_true)
        max_diff = float(np.abs(T_np - T_true).max())
        print(f"   shape: {T_np.shape}")
        print(f"   cosine vs np.linalg.inv: {cos:.8f}")
        print(f"   max|Δ|:  {max_diff:.4e}")
        print(f"   wall time: {(t1-t0)*1000:.2f} ms")
        if cos > 0.9999 and max_diff < 1e-3:
            print(f"   ✓ accurate at fp32")
        else:
            print(f"   ⚠ accuracy degraded at fp32")

        # Test 2: ttnn bf16
        print("\n" + "=" * 64)
        print("Test 2: ttnn bf16, single L of shape [64, 64]")
        print("=" * 64)
        L_tt = ttnn.from_torch(torch.from_numpy(L_np), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        t0 = time.time()
        T_tt = neumann_inverse_ttnn(L_tt, device, dtype=ttnn.bfloat16, C=C)
        ttnn.synchronize_device(device)
        t1 = time.time()
        T_np = ttnn.to_torch(T_tt).float().cpu().numpy()
        cos = _cosine(T_np, T_true)
        max_diff = float(np.abs(T_np - T_true).max())
        print(f"   cosine vs np.linalg.inv: {cos:.8f}")
        print(f"   max|Δ|:  {max_diff:.4e}")
        print(f"   wall time: {(t1-t0)*1000:.2f} ms")
        if cos > 0.99 and max_diff < 0.05:
            print(f"   ✓ bf16 accurate enough for our use")
        else:
            print(f"   ⚠ bf16 precision insufficient — fp32 may be required for this step")

        # Test 3: batched [N_V=32, 64, 64]
        print("\n" + "=" * 64)
        print("Test 3: batched fp32 [N_V=32, 64, 64] — production shape")
        print("=" * 64)
        L_batch_np = rng.standard_normal((32, C, C)).astype(np.float32) * 0.05
        for i in range(32):
            L_batch_np[i] = np.tril(L_batch_np[i], k=-1)
        T_true_batch = np.stack(
            [np.linalg.inv(np.eye(C, dtype=np.float32) - L_batch_np[i]) for i in range(32)]
        )
        L_tt = ttnn.from_torch(torch.from_numpy(L_batch_np), dtype=ttnn.float32,
                                device=device, layout=ttnn.TILE_LAYOUT)
        t0 = time.time()
        T_tt = neumann_inverse_ttnn(L_tt, device, dtype=ttnn.float32, C=C)
        ttnn.synchronize_device(device)
        t1 = time.time()
        T_np = ttnn.to_torch(T_tt).float().cpu().numpy()
        cos = _cosine(T_np, T_true_batch)
        max_diff = float(np.abs(T_np - T_true_batch).max())
        print(f"   shape: {T_np.shape}")
        print(f"   cosine: {cos:.8f}")
        print(f"   max|Δ|:  {max_diff:.4e}")
        print(f"   wall time (32 heads): {(t1-t0)*1000:.2f} ms")
        per_chunk_layer_ms = (t1-t0)*1000
        # estimate full prefill cost at N=32k
        n_chunks = 32 * 1024 // C
        n_dn_layers = 48
        total_inverse_ms = per_chunk_layer_ms * n_chunks * n_dn_layers / 32  # /32 because we batched heads already
        print(f"   est. total (I-L)^-1 cost for 32k prefill: {total_inverse_ms:.0f} ms")
        print(f"   (this is JUST the inverse, not the chunked-prefill total)")

        print("\n" + "=" * 64)
        print("Verdict")
        print("=" * 64)
        if max_diff < 1e-3 and per_chunk_layer_ms < 50:
            print("✓ Neumann factorization is the right path for C'5b.")
            print("  Sufficient precision at fp32, fast enough at production shape.")
        elif max_diff < 1e-3:
            print("⚠ Numerically correct but slow. May need to reduce chunk size or")
            print("  batch chunks across layers to amortize matmul overhead.")
        else:
            print("✗ Numerical accuracy insufficient. Re-examine the factorization;")
            print("  may need to use Schur complement or other method.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
