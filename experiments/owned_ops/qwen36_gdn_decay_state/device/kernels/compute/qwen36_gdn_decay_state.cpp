// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t ONE_TILE = 1;

FORCE_INLINE void mul_alpha_tile_indexed(uint32_t cb_in, uint32_t cb_alpha, uint32_t tile_index, uint32_t cb_out) {
    reconfig_data_format(cb_in, cb_alpha);
    pack_reconfig_data_format(cb_out);
    mul_tiles_init(cb_in, cb_alpha);
    cb_wait_front(cb_in, tile_index + 1);
    cb_wait_front(cb_alpha, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_in, cb_alpha, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

FORCE_INLINE void copy_indexed(uint32_t cb_in, uint32_t tile_index, uint32_t cb_out) {
    pack_reconfig_data_format(cb_out);
    copy_tile_to_dst_init_short_with_dt(cb_in, cb_out);
    cb_wait_front(cb_in, tile_index + 1);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    copy_tile(cb_in, tile_index, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

FORCE_INLINE void fill_one(uint32_t cb_out) {
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 1.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_state_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_alpha = get_compile_time_arg_val(1);
    constexpr uint32_t cb_state_out = get_compile_time_arg_val(2);

    const uint32_t block_count = get_arg_val<uint32_t>(0);
    const uint32_t key_tiles = get_arg_val<uint32_t>(1);
    const uint32_t debug_mode = get_arg_val<uint32_t>(2);

    binary_op_init_common(cb_state_in, cb_alpha, cb_state_out);

    for (uint32_t block = 0; block < block_count; ++block) {
        for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
            if (debug_mode == 2) {
                fill_one(cb_state_out);
            } else if (debug_mode == 1) {
                copy_indexed(cb_state_in, key_tile, cb_state_out);
            } else {
                mul_alpha_tile_indexed(cb_state_in, cb_alpha, key_tile, cb_state_out);
            }
        }
        cb_pop_front(cb_state_in, key_tiles);
        cb_pop_front(cb_alpha, ONE_TILE);
    }
}
