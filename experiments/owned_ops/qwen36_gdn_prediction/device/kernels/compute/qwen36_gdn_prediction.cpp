// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"

namespace {

constexpr uint32_t ONE_TILE = 1;

FORCE_INLINE void matmul_reduce(uint32_t cb_k, uint32_t cb_state, uint32_t key_tiles, uint32_t cb_out) {
    mm_init(cb_k, cb_state, cb_out);
    cb_wait_front(cb_k, key_tiles);
    cb_wait_front(cb_state, key_tiles);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
        matmul_tiles(cb_k, cb_state, key_tile, key_tile, 0);
    }
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

FORCE_INLINE void transpose_k_indexed(uint32_t cb_k, uint32_t tile_index, uint32_t cb_k_col) {
    unary_op_init_common(cb_k, cb_k_col);
    transpose_wh_init_short(cb_k);
    pack_reconfig_data_format(cb_k_col);
    cb_wait_front(cb_k, tile_index + 1);
    cb_reserve_back(cb_k_col, ONE_TILE);

    tile_regs_acquire();
    transpose_wh_tile(cb_k, tile_index, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_k_col);
    tile_regs_release();

    cb_push_back(cb_k_col, ONE_TILE);
}

FORCE_INLINE void mul_state_k_col_indexed(
    uint32_t cb_state,
    uint32_t cb_k_col,
    uint32_t tile_index,
    uint32_t cb_product) {
    reconfig_data_format(cb_state, cb_k_col);
    pack_reconfig_data_format(cb_product);
    mul_tiles_init(cb_state, cb_k_col);
    cb_wait_front(cb_state, tile_index + 1);
    cb_wait_front(cb_k_col, ONE_TILE);
    cb_reserve_back(cb_product, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_state, cb_k_col, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_product);
    tile_regs_release();

    cb_push_back(cb_product, ONE_TILE);
}

FORCE_INLINE void reduce_col_to_cb(uint32_t cb_product, uint32_t cb_scaler, uint32_t cb_reduced) {
    cb_wait_front(cb_product, ONE_TILE);
    cb_wait_front(cb_scaler, ONE_TILE);

    tile_regs_acquire();
    reduce_init<REDUCE_OP, REDUCE_DIM>(cb_product, cb_scaler, cb_reduced);
    reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_product, cb_scaler, 0, 0, 0);
    reduce_uninit();
    cb_reserve_back(cb_reduced, ONE_TILE);
    pack_tile(0, cb_reduced);
    tile_regs_commit();
    tile_regs_release();

    cb_push_back(cb_reduced, ONE_TILE);
}

FORCE_INLINE void copy_front_to_cb(uint32_t cb_in, uint32_t cb_out) {
    copy_tile_to_dst_init_short_with_dt(cb_in, cb_out);
    pack_reconfig_data_format(cb_out);
    cb_wait_front(cb_in, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

FORCE_INLINE void add_front_to_cb(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
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
}

FORCE_INLINE void broadcast_row_to_cb(uint32_t cb_in, uint32_t cb_out) {
    unary_op_init_common(cb_in, cb_out);
    pack_reconfig_data_format(cb_out);
    cb_wait_front(cb_in, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    unary_bcast<BroadcastType::ROW>(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
}

FORCE_INLINE void strict_reduce_prediction(
    uint32_t cb_k,
    uint32_t cb_state,
    uint32_t cb_k_col,
    uint32_t cb_product,
    uint32_t cb_scaler,
    uint32_t cb_reduced,
    uint32_t cb_accum_a,
    uint32_t cb_accum_b,
    uint32_t key_tiles,
    uint32_t cb_out) {
    fill_one(cb_scaler);

    uint32_t active_accum = cb_accum_a;
    for (uint32_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
        transpose_k_indexed(cb_k, key_tile, cb_k_col);
        mul_state_k_col_indexed(cb_state, cb_k_col, key_tile, cb_product);
        reduce_col_to_cb(cb_product, cb_scaler, cb_reduced);

        cb_pop_front(cb_k_col, ONE_TILE);
        cb_pop_front(cb_product, ONE_TILE);

        if (key_tile == 0) {
            copy_front_to_cb(cb_reduced, active_accum);
        } else {
            const uint32_t next_accum = active_accum == cb_accum_a ? cb_accum_b : cb_accum_a;
            add_front_to_cb(active_accum, cb_reduced, next_accum);
            cb_pop_front(active_accum, ONE_TILE);
            active_accum = next_accum;
        }
        cb_pop_front(cb_reduced, ONE_TILE);
    }

    broadcast_row_to_cb(active_accum, cb_out);
    cb_pop_front(active_accum, ONE_TILE);
    cb_pop_front(cb_scaler, ONE_TILE);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_state_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_k_col = get_compile_time_arg_val(2);
    constexpr uint32_t cb_product = get_compile_time_arg_val(3);
    constexpr uint32_t cb_reduce_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cb_reduced = get_compile_time_arg_val(5);
    constexpr uint32_t cb_accum_a = get_compile_time_arg_val(6);
    constexpr uint32_t cb_accum_b = get_compile_time_arg_val(7);
    constexpr uint32_t cb_pred_out = get_compile_time_arg_val(8);

    const uint32_t block_count = get_arg_val<uint32_t>(0);
    const uint32_t key_tiles = get_arg_val<uint32_t>(1);
    const uint32_t debug_mode = get_arg_val<uint32_t>(2);

    binary_op_init_common(cb_k, cb_state_in, cb_pred_out);
    compute_kernel_hw_startup(cb_product, cb_reduce_scaler, cb_reduced);
    reduce_init<REDUCE_OP, REDUCE_DIM>(cb_product, cb_reduce_scaler, cb_reduced);
    reduce_uninit();

    for (uint32_t block = 0; block < block_count; ++block) {
        if (debug_mode == 1) {
            cb_wait_front(cb_state_in, key_tiles);
            cb_wait_front(cb_k, key_tiles);
            fill_one(cb_pred_out);
        } else if (debug_mode == 2) {
            strict_reduce_prediction(
                cb_k,
                cb_state_in,
                cb_k_col,
                cb_product,
                cb_reduce_scaler,
                cb_reduced,
                cb_accum_a,
                cb_accum_b,
                key_tiles,
                cb_pred_out);
        } else if (debug_mode == 10) {
            transpose_k_indexed(cb_k, 0, cb_pred_out);
        } else if (debug_mode == 11) {
            transpose_k_indexed(cb_k, 0, cb_k_col);
            mul_state_k_col_indexed(cb_state_in, cb_k_col, 0, cb_pred_out);
            cb_pop_front(cb_k_col, ONE_TILE);
        } else if (debug_mode == 12) {
            fill_one(cb_reduce_scaler);
            transpose_k_indexed(cb_k, 0, cb_k_col);
            mul_state_k_col_indexed(cb_state_in, cb_k_col, 0, cb_product);
            reduce_col_to_cb(cb_product, cb_reduce_scaler, cb_reduced);
            broadcast_row_to_cb(cb_reduced, cb_pred_out);
            cb_pop_front(cb_reduce_scaler, ONE_TILE);
            cb_pop_front(cb_k_col, ONE_TILE);
            cb_pop_front(cb_product, ONE_TILE);
            cb_pop_front(cb_reduced, ONE_TILE);
        } else {
            matmul_reduce(cb_k, cb_state_in, key_tiles, cb_pred_out);
        }
        cb_pop_front(cb_state_in, key_tiles);
        cb_pop_front(cb_k, key_tiles);
    }
}
