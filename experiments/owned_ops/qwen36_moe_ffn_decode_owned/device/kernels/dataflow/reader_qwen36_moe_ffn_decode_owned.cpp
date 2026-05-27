// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for qwen36_moe_ffn_decode_owned — G1a.
//
// Phase 1: stream all of h's tiles into CB_H (resident).
// Phase 2: for each expert e in [0, E):
//   for each gate_up output column j in [0, gate_up_tiles):
//     stream W1[e, :, j] (hidden_tiles inner-dim tiles) into CB_W1
//   for each eo output column j in [0, hidden_tiles):
//     stream W2[e, :, j] (mid_tiles inner-dim tiles) into CB_W2
//
// W1 tile-layout: tiles in row-major over [E, hidden_tiles, gate_up_tiles].
// W2 tile-layout: tiles in row-major over [E, mid_tiles, hidden_tiles].

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"

namespace {

template <typename Accessor>
FORCE_INLINE void read_tile_to_cb(
    const Accessor& acc, uint32_t tile_id, uint32_t cb_id, uint32_t tile_bytes) {
    cb_reserve_back(cb_id, 1);
    noc_async_read(acc.get_noc_addr(tile_id), get_write_ptr(cb_id), tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_h  = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w2 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rw = get_compile_time_arg_val(3);

    constexpr auto h_args  = TensorAccessorArgs<4>();
    constexpr auto w1_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    constexpr auto rw_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

    const uint32_t h_addr        = get_arg_val<uint32_t>(0);
    const uint32_t w1_addr       = get_arg_val<uint32_t>(1);
    const uint32_t w2_addr       = get_arg_val<uint32_t>(2);
    const uint32_t rw_addr       = get_arg_val<uint32_t>(3);
    const uint32_t hidden_tiles  = get_arg_val<uint32_t>(4);
    const uint32_t gate_up_tiles = get_arg_val<uint32_t>(5);
    const uint32_t mid_tiles     = get_arg_val<uint32_t>(6);
    const uint32_t experts       = get_arg_val<uint32_t>(7);

    const auto h_acc  = TensorAccessor(h_args, h_addr);
    const auto w1_acc = TensorAccessor(w1_args, w1_addr);
    const auto w2_acc = TensorAccessor(w2_args, w2_addr);
    const auto rw_acc = TensorAccessor(rw_args, rw_addr);

    const uint32_t tile_bytes = get_tile_size(cb_h);

    // Phase 1: h tiles.
    for (uint32_t t = 0; t < hidden_tiles; ++t) {
        read_tile_to_cb(h_acc, t, cb_h, tile_bytes);
    }

    // Phase 2: per-expert W1, W2, and one rw_broadcast tile.
    for (uint32_t e = 0; e < experts; ++e) {
        const uint32_t w1_expert_base = e * hidden_tiles * gate_up_tiles;
        for (uint32_t j = 0; j < gate_up_tiles; ++j) {
            for (uint32_t k = 0; k < hidden_tiles; ++k) {
                read_tile_to_cb(w1_acc, w1_expert_base + k * gate_up_tiles + j, cb_w1, tile_bytes);
            }
        }
        const uint32_t w2_expert_base = e * mid_tiles * hidden_tiles;
        for (uint32_t j = 0; j < hidden_tiles; ++j) {
            for (uint32_t k = 0; k < mid_tiles; ++k) {
                read_tile_to_cb(w2_acc, w2_expert_base + k * hidden_tiles + j, cb_w2, tile_bytes);
            }
        }
        // One rw_broadcast tile per expert: rw_broadcast[e] is a single
        // [TILE, TILE] tile at flat index e.
        read_tile_to_cb(rw_acc, e, cb_rw, tile_bytes);
    }
}
