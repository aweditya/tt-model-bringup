// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0
//
// Mamba2 SSD decode-step owned kernel — TRISC compute side.
//
// Forked from
//   ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_gdn_decode_owned/
//     device/kernels/compute/qwen36_gdn_decode_owned.cpp
// per the Path-B G1 plan (research/mm7_g1_mamba2_kernel_design.md).
//
// Math implemented here (per
// research/nemotron3_nano_architecture_brief.md §4.3 and
// wiki/65_mamba_state_space_models.md §3):
//
//   dt_eff[b, h] = clamp(softplus(dt[b, h] + dt_bias[h]),
//                        time_step_floor, time_step_max)
//   A[h]         = -exp(A_log[h])
//   decay[b, h]  = exp(dt_eff[b, h] * A[h])
//   dt_B[b, h, s] = dt_eff[b, h] * B[b, g, s]   (broadcast over heads of a group)
//
//   ssm_state[b, h, d, s] = decay[b, h] * ssm_state[b, h, d, s]
//                         + dt_B[b, h, s] * x[b, h, d]    (outer over d × s)
//
//   y[b, h, d] = sum_s(C[b, g, s] * ssm_state[b, h, d, s])
//              + D[h] * x[b, h, d]
//
// Where (per Nemotron-3 Nano shapes):
//   num_heads = 64, head_dim = 64 (= 2 tiles), ssm_state = 128 (= 4 tiles),
//   n_groups = 8 (heads_per_group = 8).
//
// All math runs in fp32 inside the dest accumulator (fp32_dest_acc_en=true).
// State CBs are bf16 on the L1 side and fp32 in the dest register file —
// the same trick the GDN owned kernel uses; see decision D4 in
// research/mm7_g1_dataflow_decisions.md.
//
// SPMD: each Tensix handles ONE (batch, head) block per kernel invocation
// (decision D1). Per-block loop covers the head_dim tile dimension
// (head_dim = 64 = 2 tiles).
//
// Build-up via runtime arg `debug_mode` (decision D7):
//   0 = production: full SSD recursion (state correct + y correct)
//   1 = fill_one smoke (no compute; output y filled with 1.0)
//   2 = decay × state only (no input contribution; output y = 1.0 sentinel)
//   3 = decay × state + input contribution (state correct; y = 1.0 sentinel)
//   4 = state correct + y = D·x (output ignores C·state)
//   5 = production equivalent (mode 0)
//
// Each subsequent commit lands one more debug_mode step. THIS commit
// ships mode=1 (fill_one smoke) only — the rest are TODO blocks with
// explicit math comments. This lets us validate the scaffolding
// (build → register → dispatch → output non-NaN) before any math
// touches the TRISC tile engine.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"

namespace {

constexpr uint32_t ONE_TILE = 1;

// ─────────────────────────────────────────────────────────────────────────────
// FORCE_INLINE helpers — Mamba2 SSD–specific tile ops.
//
// We keep each helper small and named after the math step (decision D7's
// debug_mode pattern: each mode wires up the next helper). The GDN kernel's
// helpers (mul_alpha_*, matmul_reduce, mul_beta_*, etc.) live in the fork-base
// file; this file uses Mamba2-specific names instead so a future reader
// doesn't have to mentally remap.

// fill_one(cb_out): write a single tile of 1.0 to cb_out. Used by debug_mode=1
// to validate the scaffolding (program build, kernel dispatch, output channel
// non-NaN) BEFORE any Mamba2 math runs.
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

// TODO(G1 day-3): compute_decay(cb_dt, cb_dt_bias, cb_A_log, cb_decay)
//
// Implements:
//   dt_eff = clamp(softplus(dt + dt_bias), floor, max)
//   A      = -exp(A_log)
//   decay  = exp(dt_eff * A)
//
// All three are scalar-per-head (1 tile each, with the value broadcast across
// all 32×32 positions). This runs ONCE at the start of the block; the result
// reused for both head_dim tile iterations (decision D5).
//
// LLK calls expected:
//   add_tiles(cb_dt, cb_dt_bias, 0, 0, ...)            // dt + dt_bias
//   softplus_tile(...) OR exp_tile + log1p_tile        // softplus
//                                                        (decision D8: confirm at
//                                                         G1 day-3 via LLK API survey)
//   max_tile(..., time_step_floor)                     // clamp lower
//   min_tile(..., time_step_max)                       // clamp upper
//   exp_tile(cb_A_log, ...)                            // exp(A_log)
//   neg_tile(...)                                       // -exp(A_log) = A
//   mul_tiles(dt_eff, A, ...)                          // dt_eff * A
//   exp_tile(..., cb_decay)                            // exp(dt_eff * A)

// TODO(G1 day-3): compute_dt_B(cb_dt_eff, cb_B, cb_dt_B)
//
// Implements:
//   dt_B[s] = dt_eff * B[s]   (broadcast scalar over [ssm_state] vector tile)
//
// LLK calls expected:
//   bcast_mul_tile_scalar(cb_dt_eff, cb_B, 0, 0, cb_dt_B)

// TODO(G1 day-4): mul_decay_state_to(
//     cb_state, cb_decay, head_dim_tile, cb_state_scaled)
//
// Implements (for the d-th head_dim tile):
//   state_scaled[d, s] = decay * state[d, s]
//
// decay is scalar; state is [head_dim, ssm_state] = 2×4 tiles per head.
// head_dim_tile selects which of the 2 head_dim tiles to multiply.
//
// LLK calls expected:
//   bcast_mul_tile_scalar(cb_state, cb_decay, head_dim_tile, 0, ...)

// TODO(G1 day-4): add_outer_input(
//     cb_state_scaled, cb_x, cb_dt_B, head_dim_tile, cb_state_out)
//
// Implements:
//   state_out[d, s] = state_scaled[d, s] + dt_B[s] * x[d]
//
// dt_B[s] * x[d] is the outer-product portion of the SSD update. Per
// head_dim tile, we have [d=tile, s=4 tiles] × [s=4 tiles] = 4 muladds.
//
// LLK calls expected:
//   outer_product_add(cb_x, cb_dt_B, head_dim_tile, ssm_state_tile, ...)
//   OR (decomposed): bcast_mul + add_tiles for each (d, s) pair.

// TODO(G1 day-4): C_state_reduce(cb_C, cb_state, head_dim_tile, cb_y_partial)
//
// Implements (the output reduce):
//   y_partial[d] = sum_s(C[s] * state[d, s])
//
// Matmul-style: [head_dim=1 tile] = [head_dim=1, ssm_state=4] @ [ssm_state=4, 1]
// Per head: 2 head_dim tiles → 2 calls per (batch, head).
//
// LLK calls expected:
//   matmul_tile(cb_state, cb_C, head_dim_tile, 0, cb_y_partial)
//   (fork from GDN's `matmul_reduce` at line 215 of the GDN compute kernel)

// TODO(G1 day-4): add_skip(cb_y_partial, cb_x, cb_D, head_dim_tile, cb_y)
//
// Implements:
//   y[d] = y_partial[d] + D * x[d]
//
// D is scalar; multiplied by x[d] and added to the reduce result.
//
// LLK calls expected:
//   bcast_mul_tile_scalar(cb_x, cb_D, head_dim_tile, 0, ...)
//   add_tiles(..., cb_y_partial, 0, 0, cb_y)

// ─────────────────────────────────────────────────────────────────────────────
// kernel_main — orchestration.
//
// Compile-time args = CB index assignments (must match program_factory).
// Runtime args = per-block work + debug_mode switch.
//
// The CB index numbering keeps the GDN layout for the producer/consumer
// pipeline pattern (decision D7 — debug_mode flexibility), but the CB roles
// are Mamba2-specific:
//   cb_x          : input x  (head_dim vector)
//   cb_z          : gate (pass-through; not consumed here — decision D10)
//   cb_dt         : scalar dt
//   cb_dt_bias    : scalar dt_bias (weight)
//   cb_A_log      : scalar A_log (weight)
//   cb_D          : scalar D (weight)
//   cb_B          : per-group B (ssm_state vector)
//   cb_C          : per-group C (ssm_state vector)
//   cb_state_in   : ssm_state read-side ([head_dim, ssm_state] = 2×4 tiles)
//   cb_decay      : intermediate scalar
//   cb_dt_B       : intermediate [ssm_state] vector
//   cb_state_scaled : intermediate [head_dim, ssm_state] (decay × state)
//   cb_y_partial  : intermediate [head_dim] (C·state reduce result)
//   cb_state_out  : ssm_state write-side
//   cb_y          : output y ([head_dim])

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x          = get_compile_time_arg_val(0);
    constexpr uint32_t cb_z          = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dt         = get_compile_time_arg_val(2);
    constexpr uint32_t cb_dt_bias    = get_compile_time_arg_val(3);
    constexpr uint32_t cb_A_log      = get_compile_time_arg_val(4);
    constexpr uint32_t cb_D          = get_compile_time_arg_val(5);
    constexpr uint32_t cb_B          = get_compile_time_arg_val(6);
    constexpr uint32_t cb_C          = get_compile_time_arg_val(7);
    constexpr uint32_t cb_state_in   = get_compile_time_arg_val(8);
    constexpr uint32_t cb_decay      = get_compile_time_arg_val(9);
    constexpr uint32_t cb_dt_B       = get_compile_time_arg_val(10);
    constexpr uint32_t cb_state_scaled = get_compile_time_arg_val(11);
    constexpr uint32_t cb_y_partial  = get_compile_time_arg_val(12);
    constexpr uint32_t cb_state_out  = get_compile_time_arg_val(13);
    constexpr uint32_t cb_y          = get_compile_time_arg_val(14);

    // Runtime args:
    //   block_count       : how many (batch, head) blocks this core owns
    //   head_dim_tiles    : 2 for Nemotron-3 (head_dim=64 / TILE_W=32)
    //   ssm_state_tiles   : 4 for Nemotron-3 (ssm_state=128 / TILE_W=32)
    //   debug_mode        : 0 (production) .. 5 (incremental); see file header
    const uint32_t block_count       = get_arg_val<uint32_t>(0);
    const uint32_t head_dim_tiles    = get_arg_val<uint32_t>(1);
    const uint32_t ssm_state_tiles   = get_arg_val<uint32_t>(2);
    const uint32_t debug_mode        = get_arg_val<uint32_t>(3);
    (void)cb_z;  // not consumed in the kernel; pass-through to caller (decision D10)
    (void)ssm_state_tiles;  // unused until day-3+ math lands

    binary_op_init_common(cb_state_in, cb_decay, cb_state_scaled);

    for (uint32_t block = 0; block < block_count; ++block) {
        if (debug_mode == 1) {
            // ── Mode 1: scaffolding smoke ───────────────────────────────────
            // No compute. Drain every input CB so the reader doesn't stall,
            // then write `1.0` tiles to the output CBs (cb_y and cb_state_out).
            //
            // Purpose: validate that the program builds, the kernel dispatches,
            // the CB plumbing is correct (reader/compute/writer pipelining
            // works), and the output channel is connected to the writer.
            //
            // Gate (G0a harness): `--kernel-callable …` returns output of
            // shape `[B, num_heads=64, head_dim=64]` filled with 1.0 — easy
            // to assert. Compare cosine of all-ones vs oracle expected to be
            // not-NaN (cos will be near-zero because oracle is non-trivial).
            // Pass criterion: output exists, no NaN, shape matches.
            cb_wait_front(cb_dt, ONE_TILE);
            cb_wait_front(cb_dt_bias, ONE_TILE);
            cb_wait_front(cb_A_log, ONE_TILE);
            cb_wait_front(cb_D, ONE_TILE);
            cb_wait_front(cb_x, head_dim_tiles);
            cb_wait_front(cb_B, ssm_state_tiles);
            cb_wait_front(cb_C, ssm_state_tiles);
            cb_wait_front(cb_state_in, head_dim_tiles * ssm_state_tiles);

            // Fill output state with 1.0 (one tile per [head_dim_tile, s_tile]).
            for (uint32_t i = 0; i < head_dim_tiles * ssm_state_tiles; ++i) {
                fill_one(cb_state_out);
            }
            // Fill output y with 1.0 (one tile per head_dim_tile).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                fill_one(cb_y);
            }

            cb_pop_front(cb_dt, ONE_TILE);
            cb_pop_front(cb_dt_bias, ONE_TILE);
            cb_pop_front(cb_A_log, ONE_TILE);
            cb_pop_front(cb_D, ONE_TILE);
            cb_pop_front(cb_x, head_dim_tiles);
            cb_pop_front(cb_B, ssm_state_tiles);
            cb_pop_front(cb_C, ssm_state_tiles);
            cb_pop_front(cb_state_in, head_dim_tiles * ssm_state_tiles);
        }
        // TODO(G1 day-3): debug_mode == 2
        //
        // Wire compute_decay(cb_dt, cb_dt_bias, cb_A_log, cb_decay) +
        //      mul_decay_state_to(cb_state_in, cb_decay, _, cb_state_out)
        // No input contribution; just decay × state. y stays at 1.0 sentinel.
        // Gate: state_out should be exactly `decay × state_in` (no +).
        //
        // else if (debug_mode == 2) { ... }
        //
        // TODO(G1 day-3.5): debug_mode == 3
        //
        // Add compute_dt_B + add_outer_input(cb_state_scaled, cb_x, cb_dt_B,
        //                                    cb_state_out)
        // State now correct. y still sentinel.
        // Gate: state_out matches oracle's post-update state; y == 1.0.
        //
        // TODO(G1 day-4): debug_mode == 4
        //
        // Wire add_skip(cb_y_partial, cb_x, cb_D, cb_y) with cb_y_partial == 0.
        // i.e. y = D * x only (skip C·state reduce).
        // Gate: y == D·x bit-close.
        //
        // TODO(G1 day-4.5): debug_mode == 0 / 5 (production)
        //
        // Wire C_state_reduce(cb_C, cb_state_out, _, cb_y_partial) ahead of
        // add_skip. Full math live.
        // Gate: G0a harness PASS at cos ≥ 0.999 vs numpy oracle, both 1-step
        //       AND 8-step multi-step replay.
        //
        // (For now, debug_mode != 1 paths are NOT IMPLEMENTED — kernel will
        // produce no output and the reader/writer pipeline will stall.
        // This is intentional: the day's gate is mode=1 only.)
    }
}
