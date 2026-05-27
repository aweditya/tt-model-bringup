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
`research/35b_moe_ffn_kernel_build_plan.md` (G0..G4 stages).

## Contract (G0)

- `h`: rank-2 `[1, HIDDEN]` bf16, TILE_LAYOUT.
- `W1`: rank-3 `[E_LOCAL, HIDDEN, 2*MOE_INTER]` bf16, TILE_LAYOUT.
- `W2`: rank-3 `[E_LOCAL, MOE_INTER, HIDDEN]` bf16, TILE_LAYOUT.
- `routing_weight`: rank-2 `[1, E_LOCAL]` bf16, TILE_LAYOUT.
- Returns `Tensor` of shape `[1, HIDDEN]` bf16.

G0 (this stage) outputs all zeros — the kernel reads h, drains it, and
emits hidden_tiles zero tiles. No math yet. Purpose: verify build,
nanobind binding, program-factory plumbing, and the reader/compute/writer
pipeline before adding real compute.

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
| G0    | in-progress | Scaffold; output zeros |
| G0a   | pending     | Numpy oracle + isolation harness |
| G1    | pending     | Single-core full-chain compute (no cross-core reduce) |
| G2    | pending     | Per-expert work split + cross-core sum |
| G3    | pending     | Mcast h + finer partition for utilization |
| G4    | pending     | Wire into server_35b_ttnn (state.moe_owned_ffn) |
