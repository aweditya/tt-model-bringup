"""Numpy oracle for the fused MoE FFN decode op.

Reference math for qwen36_moe_ffn_decode_owned. The on-device kernel must
match this within bf16-precision pcc gates. Shared between the G1..G3
isolation tests and any future debugging probe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def silu_np(x: np.ndarray) -> np.ndarray:
    """SiLU = x * sigmoid(x) = x / (1 + exp(-x))."""
    return x / (1.0 + np.exp(-x.astype(np.float64))).astype(x.dtype)


@dataclass
class MoeFfnFixture:
    h: np.ndarray              # [HIDDEN] fp32
    W1: np.ndarray             # [E, HIDDEN, 2*MOE_INTER] fp32
    W2: np.ndarray             # [E, MOE_INTER, HIDDEN] fp32
    routing_weight: np.ndarray # [E] fp32

    @property
    def E(self) -> int: return int(self.W1.shape[0])
    @property
    def HIDDEN(self) -> int: return int(self.h.shape[0])
    @property
    def MOE_INTER(self) -> int: return int(self.W1.shape[2] // 2)


def make_fixture(
    E: int = 64,
    HIDDEN: int = 2048,
    MOE_INTER: int = 512,
    *,
    seed: int = 0,
    scale_h: float = 1.0,
    scale_W: float = 0.05,
    top_k: Optional[int] = None,
) -> MoeFfnFixture:
    """Synthesize a deterministic test fixture matching production shapes.

    `top_k`: if set, only `top_k` of the `E` routing_weight slots are non-zero
    (rest are exactly 0.0, mirroring how Pattern A's on-device mask zeros
    out non-selected experts). Default is None = all experts contribute.
    """
    rng = np.random.default_rng(seed)
    h = (rng.standard_normal(HIDDEN) * scale_h).astype(np.float32)
    W1 = (rng.standard_normal((E, HIDDEN, 2 * MOE_INTER)) * scale_W).astype(np.float32)
    W2 = (rng.standard_normal((E, MOE_INTER, HIDDEN)) * scale_W).astype(np.float32)
    # Routing weights: simulate Pattern A's post-mask normalized weights.
    if top_k is None:
        rw_raw = np.abs(rng.standard_normal(E)).astype(np.float32)
    else:
        chosen = rng.choice(E, size=top_k, replace=False)
        rw_raw = np.zeros(E, dtype=np.float32)
        rw_raw[chosen] = np.abs(rng.standard_normal(top_k)).astype(np.float32)
    rw = rw_raw / rw_raw.sum()  # normalize to sum to 1
    return MoeFfnFixture(h=h, W1=W1, W2=W2, routing_weight=rw)


def moe_ffn_oracle(fixture: MoeFfnFixture, *, bf16: bool = True) -> np.ndarray:
    """Reference forward through the full MoE FFN chain.

    If bf16=True, casts every intermediate through bfloat16 to mirror what
    the on-device path sees. Bit-cast via torch since numpy lacks native bf16.
    """
    import torch

    def bf(x):
        if not bf16:
            return x
        return torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).float().numpy()

    h  = bf(fixture.h)
    W1 = bf(fixture.W1)
    W2 = bf(fixture.W2)
    rw = bf(fixture.routing_weight)

    E, HIDDEN, twoI = W1.shape
    MOE_INTER = twoI // 2

    # gate_up[e, j] = sum_k h[k] * W1[e, k, j]
    gate_up = bf(np.einsum("k,ekj->ej", h, W1))
    gate = gate_up[:, :MOE_INTER]
    up   = gate_up[:, MOE_INTER:]

    mid = bf(silu_np(gate) * up)
    # expert_out[e, j] = sum_k mid[e, k] * W2[e, k, j]
    expert_out = bf(np.einsum("ek,ekj->ej", mid, W2))

    # routed[j] = sum_e rw[e] * expert_out[e, j]
    routed = bf((rw[:, None] * expert_out).sum(axis=0))
    return routed


def pcc(actual: np.ndarray, expected: np.ndarray) -> float:
    """Pearson correlation coefficient between actual and expected, flattened."""
    a = actual.astype(np.float64).reshape(-1)
    b = expected.astype(np.float64).reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def max_abs_diff(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max())


def diff_report(actual: np.ndarray, expected: np.ndarray) -> dict:
    return {
        "pcc": pcc(actual, expected),
        "max_abs_diff": max_abs_diff(actual, expected),
        "actual_norm": float(np.linalg.norm(actual.astype(np.float64))),
        "expected_norm": float(np.linalg.norm(expected.astype(np.float64))),
    }
