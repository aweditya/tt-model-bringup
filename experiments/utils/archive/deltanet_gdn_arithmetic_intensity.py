#!/usr/bin/env python3
"""Arithmetic-intensity estimate for Qwen3.6 DeltaNet GDN decode fusion.

This is a shape-level calculator, not a performance predictor.  It estimates
the recurrent-state update/output body:

    H' = alpha * H + k^T * beta * (v - k @ (alpha * H))
    y  = q @ H'

for one decode token after Q/K/V have already been prepared into per-value-head
slots.  The goal is to bound whether a fused native op is compute-heavy or
memory/dispatch/layout dominated.
"""

from __future__ import annotations

import argparse
import json


def estimate(
    value_heads_per_chip: int,
    key_dim: int,
    value_dim: int,
    state_bytes: int,
    vector_bytes: int,
    scalar_bytes: int,
) -> dict:
    slots = value_heads_per_chip

    state_elems_per_slot = key_dim * value_dim
    vector_elems_per_slot = key_dim * 2 + value_dim
    scalar_elems_per_slot = 2

    # Main work per slot, counted as FLOPs with MAC = 2 FLOPs.
    prediction_flops = 2 * key_dim * value_dim
    delta_flops = value_dim * 3  # v - pred, * beta, small bookkeeping
    outer_update_flops = key_dim * value_dim * 2  # mul + add, conservative
    output_flops = 2 * key_dim * value_dim
    state_decay_flops = key_dim * value_dim
    flops_per_slot = prediction_flops + delta_flops + outer_update_flops + output_flops + state_decay_flops

    # Minimum useful traffic if the fused op streams state once and writes it once.
    state_read_bytes = slots * state_elems_per_slot * state_bytes
    state_write_bytes = slots * state_elems_per_slot * state_bytes
    qkv_read_bytes = slots * vector_elems_per_slot * vector_bytes
    scalar_read_bytes = slots * scalar_elems_per_slot * scalar_bytes
    output_write_bytes = slots * value_dim * state_bytes
    total_bytes = state_read_bytes + state_write_bytes + qkv_read_bytes + scalar_read_bytes + output_write_bytes

    total_flops = slots * flops_per_slot
    return {
        "slots_per_chip": slots,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "flops_per_slot": flops_per_slot,
        "flops_per_chip_token": total_flops,
        "traffic_bytes_per_chip_token": {
            "state_read": state_read_bytes,
            "state_write": state_write_bytes,
            "qkv_read": qkv_read_bytes,
            "alpha_beta_read": scalar_read_bytes,
            "output_write": output_write_bytes,
            "total": total_bytes,
        },
        "arithmetic_intensity_flop_per_byte": total_flops / total_bytes,
        "lower_bound_us_at_512_GBps": total_bytes / 512e9 * 1e6,
        "lower_bound_us_at_1_TFLOPs": total_flops / 1e12 * 1e6,
        "notes": [
            "This excludes QKV projection, conv, softplus/alpha generation, output RMSNorm/gate, output projection, and collectives.",
            "It is a lower-bound traffic model; generic TTNN graphs materialize many more intermediates.",
            "Low arithmetic intensity means the custom op's value is reducing dispatch/layout/intermediate traffic, not making dense compute the bottleneck.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-heads-per-chip", type=int, default=12)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--state-bytes", type=int, default=4, help="FP32 recurrent state bytes")
    parser.add_argument("--vector-bytes", type=int, default=2, help="Q/K/V vector bytes if bf16 inputs")
    parser.add_argument("--scalar-bytes", type=int, default=4, help="alpha/beta scalar bytes")
    args = parser.parse_args()
    print(json.dumps(estimate(
        value_heads_per_chip=args.value_heads_per_chip,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        state_bytes=args.state_bytes,
        vector_bytes=args.vector_bytes,
        scalar_bytes=args.scalar_bytes,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
