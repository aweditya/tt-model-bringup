# qwen36_moe_ffn_decode_owned

Owned TTNN source for the fused Qwen3.6 MoE Pattern A batched FFN decode op.

```text
gate_up = h @ W1                                # batched over experts
gate    = gate_up[:, :MOE_INTER]
up      = gate_up[:, MOE_INTER:]
mid     = silu(gate) * up
eo      = mid @ W2                              # batched over experts
routed  = sum_e (routing_weight[e] * eo[e])     # cross-expert reduction
```

Replaces the three back-to-back ttnn ops in
`server_35b_ttnn.moe_forward_ttnn_pattern_a_batched`. Goal: keep
`gate_up`, `mid`, and `eo` resident in L1 instead of round-tripping
to DRAM between ops, and reduce trace-time kernel count.

Design context: `research/35b_moe_ffn_kernel_scoping.md` (architecture map),
`archive/superseded_research_2026-06-04/35b_moe_ffn_kernel_build_plan.md` (G0..G4 stages, archived 2026-06-04 after kernel landed).

## Contract (G1b)

- `h`: rank-2 `[1, HIDDEN]` bf16, TILE_LAYOUT.
- `W1`: rank-3 `[E_LOCAL, HIDDEN, 2*MOE_INTER]` bf16, TILE_LAYOUT.
- `W2`: rank-3 `[E_LOCAL, MOE_INTER, HIDDEN]` bf16, TILE_LAYOUT.
- `routing_weight`: rank-3 `[E_LOCAL, TILE, TILE]` bf16, TILE_LAYOUT — caller
  pre-broadcasts `rw[e]` into a full TILE×TILE tile (D-G1b-01).
- Returns `Tensor` of shape `[1, HIDDEN]` bf16.

G1b runs the full chain (gate_up → silu*up → eo → rw·eo summed over experts)
on one Tensix core. Verified at `H=64, I=32, E=2` toy shape:
`pcc=0.99998756` vs bf16 numpy oracle.

## Build & integration

```
python experiments/owned_ops/qwen36_moe_ffn_decode_owned/integrate_into_ttmetal.py \
    --tt-metal $HOME/tenstorrent/tt-metal
cd $HOME/tenstorrent/tt-metal && cmake --build build_Release --target ttnn
cmake --install build_Release --component tt_pybinds
```

Anchors on `qwen36_decay_gate_decode_owned` entries in the CMakeLists and
experimental_nanobind.cpp (decay_gate is already installed via its own
installer + the bulk qwen36 sync from the owned-GDN session).

## Stages

| Stage | Status | What lands |
|---|---|---|
| G0    | done        | Scaffold; identity copy (h → out), pipeline plumbing verified |
| G0a   | done        | Numpy oracle + isolation harness |
| G1a   | done        | Single-core full-chain compute, rw ignored (sum_e eo[e]) |
| G1b   | done        | + routing-weight scaling (pcc=0.999988 @ H=64, I=32, E=2) |
| G2    | pending     | Per-expert work split + cross-core sum |
| G3    | pending     | Mcast h + finer partition for utilization |
| G4    | pending     | Wire into server_35b_ttnn (state.moe_owned_ffn) |
