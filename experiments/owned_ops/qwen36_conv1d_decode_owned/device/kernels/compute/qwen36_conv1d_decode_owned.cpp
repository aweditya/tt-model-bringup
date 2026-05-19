// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Fused 4-tap depthwise conv1d + silu compute kernel.
//
// Per work block tile (each tile is one [32, 32] slab of the D-dimensional
// per-channel state) we compute:
//
//   acc1 = silu( state0*w0 + state1*w1 + state2*w2 + mixed*w3 )
//
// using explicit 4 muls + 3 adds + 1 silu. No ttnn.sum reduce over the K=4
// dimension — that's exactly the small-reduce-tile-tax the prior
// feedback_conv1d_diagnosis.md identified as 65% of the eager body cost.
//
// debug_fill mode replaces the math path with a copy of the input mixed
// tile so we can sanity-check the scaffold without depending on the real
// arithmetic.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t ONE_TILE = 1;

FORCE_INLINE void mul_front(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    mul_tiles_init(cb_a, cb_b, false, __builtin_LINE());
    cb_wait_front(cb_a, ONE_TILE);
    cb_wait_front(cb_b, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_a, ONE_TILE);
    cb_pop_front(cb_b, ONE_TILE);
}

FORCE_INLINE void add_front(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    add_tiles_init(cb_a, cb_b);
    cb_wait_front(cb_a, ONE_TILE);
    cb_wait_front(cb_b, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_a, ONE_TILE);
    cb_pop_front(cb_b, ONE_TILE);
}

FORCE_INLINE void silu_front(uint32_t cb_in, uint32_t cb_out) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    init_sfpu(cb_in, cb_out);
    cb_wait_front(cb_in, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_in, ONE_TILE);
}

FORCE_INLINE void copy_front(uint32_t cb_in, uint32_t cb_out) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    init_sfpu(cb_in, cb_out);
    cb_wait_front(cb_in, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_in, ONE_TILE);
}

FORCE_INLINE void drain_one(uint32_t cb_in) {
    cb_wait_front(cb_in, ONE_TILE);
    cb_pop_front(cb_in, ONE_TILE);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_mixed = get_compile_time_arg_val(0);
    constexpr uint32_t cb_state0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_state1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_state2 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_weight0 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_weight1 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_weight2 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_weight3 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_product = get_compile_time_arg_val(8);
    constexpr uint32_t cb_acc0 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_acc1 = get_compile_time_arg_val(10);
    constexpr uint32_t cb_conv_out = get_compile_time_arg_val(11);

    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    const uint32_t debug_fill = get_arg_val<uint32_t>(1);

    binary_op_init_common(cb_state0, cb_weight0, cb_acc0);

    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        if (debug_fill) {
            // Scaffold sanity: emit a copy of mixed to conv_out without
            // touching the real arithmetic. Drain all the other input CBs
            // so the reader's per-tile produces don't back up.
            copy_front(cb_mixed, cb_conv_out);
            drain_one(cb_state0);
            drain_one(cb_state1);
            drain_one(cb_state2);
            drain_one(cb_weight0);
            drain_one(cb_weight1);
            drain_one(cb_weight2);
            drain_one(cb_weight3);
            continue;
        }

        mul_front(cb_state0, cb_weight0, cb_acc0);  // acc0 = state0 * w0

        mul_front(cb_state1, cb_weight1, cb_product);
        add_front(cb_acc0, cb_product, cb_acc1);     // acc1 = acc0 + state1 * w1

        mul_front(cb_state2, cb_weight2, cb_product);
        add_front(cb_acc1, cb_product, cb_acc0);     // acc0 = acc1 + state2 * w2

        mul_front(cb_mixed, cb_weight3, cb_product);
        add_front(cb_acc0, cb_product, cb_acc1);     // acc1 = acc0 + mixed * w3

        silu_front(cb_acc1, cb_conv_out);            // conv_out = silu(acc1)
    }
}
