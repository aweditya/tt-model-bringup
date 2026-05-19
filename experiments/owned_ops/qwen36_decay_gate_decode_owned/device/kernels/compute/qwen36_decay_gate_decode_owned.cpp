// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Fused DeltaNet decay/gate compute kernel — single tile per call.
//
// Computes:
//   softplus_a = softplus(a + dt_bias)
//   g          = -exp(A_log) * softplus_a
//   decay      = exp(g)
//   beta       = sigmoid(b)
//
// Collapses the 10-op production chain (add → softplus → exp → neg → mul →
// exp → sigmoid → reshape × 2) to a single kernel launch. NV_PER_CHIP=12
// elements fit in one 32×32 tile, so we process exactly 1 tile per call.
//
// debug_fill mode skips math: emits a copy of cb_a -> cb_decay_out and
// cb_b -> cb_beta_out (sanity check that the scaffold compiles, dispatches,
// and writes to the right output buffers).

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

// dst[0] = a + dt_bias; SFPU softplus_tile(0); pack -> cb_softplus
FORCE_INLINE void softplus_step(uint32_t cb_a, uint32_t cb_dt_bias, uint32_t cb_softplus) {
    reconfig_data_format(cb_a, cb_dt_bias);
    pack_reconfig_data_format(cb_softplus);
    add_tiles_init(cb_a, cb_dt_bias);
    cb_wait_front(cb_a, ONE_TILE);
    cb_wait_front(cb_dt_bias, ONE_TILE);
    cb_reserve_back(cb_softplus, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_a, cb_dt_bias, 0, 0, 0);
    softplus_tile_init();
    softplus_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_softplus);
    tile_regs_release();

    cb_push_back(cb_softplus, ONE_TILE);
    cb_pop_front(cb_a, ONE_TILE);
    cb_pop_front(cb_dt_bias, ONE_TILE);
}

// dst[0] = -exp(A_log); pack -> cb_neg_exp_A
FORCE_INLINE void neg_exp_step(uint32_t cb_A_log, uint32_t cb_neg_exp_A) {
    reconfig_data_format_srca(cb_A_log);
    pack_reconfig_data_format(cb_neg_exp_A);
    init_sfpu(cb_A_log, cb_neg_exp_A);
    cb_wait_front(cb_A_log, ONE_TILE);
    cb_reserve_back(cb_neg_exp_A, ONE_TILE);

    tile_regs_acquire();
    copy_tile(cb_A_log, 0, 0);
    exp_tile_init(); exp_tile(0);
    negative_tile_init(); negative_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_neg_exp_A);
    tile_regs_release();

    cb_push_back(cb_neg_exp_A, ONE_TILE);
    cb_pop_front(cb_A_log, ONE_TILE);
}

// dst[0] = a * b; pack -> cb_out
FORCE_INLINE void mul_front(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    mul_tiles_init(cb_a, cb_b);
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

// dst[0] = exp(cb_g); pack -> cb_decay_out
FORCE_INLINE void exp_step(uint32_t cb_g, uint32_t cb_decay_out) {
    reconfig_data_format_srca(cb_g);
    pack_reconfig_data_format(cb_decay_out);
    init_sfpu(cb_g, cb_decay_out);
    cb_wait_front(cb_g, ONE_TILE);
    cb_reserve_back(cb_decay_out, ONE_TILE);

    tile_regs_acquire();
    copy_tile(cb_g, 0, 0);
    exp_tile_init(); exp_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_decay_out);
    tile_regs_release();

    cb_push_back(cb_decay_out, ONE_TILE);
    cb_pop_front(cb_g, ONE_TILE);
}

// dst[0] = sigmoid(cb_b); pack -> cb_beta_out
FORCE_INLINE void sigmoid_step(uint32_t cb_b, uint32_t cb_beta_out) {
    reconfig_data_format_srca(cb_b);
    pack_reconfig_data_format(cb_beta_out);
    init_sfpu(cb_b, cb_beta_out);
    cb_wait_front(cb_b, ONE_TILE);
    cb_reserve_back(cb_beta_out, ONE_TILE);

    tile_regs_acquire();
    copy_tile(cb_b, 0, 0);
    sigmoid_tile_init(); sigmoid_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_beta_out);
    tile_regs_release();

    cb_push_back(cb_beta_out, ONE_TILE);
    cb_pop_front(cb_b, ONE_TILE);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dt_bias = get_compile_time_arg_val(2);
    constexpr uint32_t cb_A_log = get_compile_time_arg_val(3);
    constexpr uint32_t cb_softplus = get_compile_time_arg_val(4);
    constexpr uint32_t cb_neg_exp_A = get_compile_time_arg_val(5);
    constexpr uint32_t cb_g = get_compile_time_arg_val(6);
    constexpr uint32_t cb_decay_out = get_compile_time_arg_val(7);
    constexpr uint32_t cb_beta_out = get_compile_time_arg_val(8);

    const uint32_t debug_fill = get_arg_val<uint32_t>(0);

    binary_op_init_common(cb_a, cb_dt_bias, cb_softplus);

    if (debug_fill) {
        // Scaffold sanity: copy a -> decay_out, b -> beta_out, drain others.
        copy_front(cb_a, cb_decay_out);
        copy_front(cb_b, cb_beta_out);
        drain_one(cb_dt_bias);
        drain_one(cb_A_log);
        return;
    }

    // softplus_a = softplus(a + dt_bias)
    softplus_step(cb_a, cb_dt_bias, cb_softplus);

    // neg_exp_A = -exp(A_log)
    neg_exp_step(cb_A_log, cb_neg_exp_A);

    // g = neg_exp_A * softplus_a
    mul_front(cb_neg_exp_A, cb_softplus, cb_g);

    // decay = exp(g)
    exp_step(cb_g, cb_decay_out);

    // beta = sigmoid(b)
    sigmoid_step(cb_b, cb_beta_out);
}
