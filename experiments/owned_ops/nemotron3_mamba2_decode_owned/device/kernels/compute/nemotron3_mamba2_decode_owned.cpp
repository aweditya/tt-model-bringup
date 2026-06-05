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
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

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

// compute_decay(cb_A_log, cb_decay)
//
// Stage A of the decay computation. Produces A = -exp(A_log) into cb_decay.
// Caller follows up with finalize_decay_with_dt_eff() which multiplies by
// the recomputed dt_eff and exps the product to get the final decay.
//
// This helper consumes ONLY cb_A_log. cb_dt and cb_dt_bias are consumed
// by finalize_decay (decision D11 v2: keep the two stages separate, one
// tile_regs cycle each, simpler to debug + matches GDN's single-purpose
// helper convention).
//
// Decision D8 RESOLVED (LLK API survey at G1 day-3): `exp_tile` and
// `negative_tile` are first-class SFPU primitives at
// /home/aditya/tenstorrent/tt-metal/tt_metal/hw/inc/api/compute/
// eltwise_unary/{exp,negative}.h. No decomposition.
FORCE_INLINE void compute_decay(uint32_t cb_A_log, uint32_t cb_decay) {
    cb_wait_front(cb_A_log, ONE_TILE);

    pack_reconfig_data_format(cb_decay);
    reconfig_data_format(cb_A_log, cb_A_log);
    cb_reserve_back(cb_decay, ONE_TILE);

    tile_regs_acquire();
    // A_log → exp → negate → A
    copy_tile_init(cb_A_log);
    copy_tile(cb_A_log, 0, 0);
    exp_tile_init();
    exp_tile(0);
    negative_tile_init();
    negative_tile(0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_decay);
    tile_regs_release();

    cb_push_back(cb_decay, ONE_TILE);
    cb_pop_front(cb_A_log, ONE_TILE);
}

// finalize_decay_with_dt_eff(cb_decay_inout, cb_dt, cb_dt_bias,
//                            cb_dt_eff_scratch, softplus/clamp constants)
//
// Stage B of the decay computation. Reads A from cb_decay_inout (placed
// there by compute_decay), computes dt_eff = clamp(softplus(dt+dt_bias),
// floor, max) into cb_dt_eff_scratch, multiplies A * dt_eff and exps the
// result, overwrites cb_decay_inout with the final decay value.
//
// The caller passes `cb_dt_B` as `cb_dt_eff_scratch`. cb_dt_B is intended
// for the production `dt_eff * B` outer product (used at debug_mode=3+);
// reusing it here as a scalar dt_eff scratch is decision-D11-style
// double-duty. The scratch is empty by the time debug_mode=3 wants it
// for B-multiply (this helper pops it before exit), so no conflict.
//
// Three-stage pipeline:
//   Stage 1: dt_eff = clamp(softplus(dt + dt_bias), floor, max)  → cb_dt_eff_scratch
//   Stage 2: A_dt   = A * dt_eff                                  (binary mul_tiles,
//                                                                  reads both CBs)
//   Stage 3: decay  = exp(A_dt)                                   → cb_decay_inout
//                                                                  (overwrites A)
FORCE_INLINE void finalize_decay_with_dt_eff(
    uint32_t cb_decay_inout,
    uint32_t cb_dt,
    uint32_t cb_dt_bias,
    uint32_t cb_dt_eff_scratch,
    uint32_t softplus_beta_bits,
    uint32_t softplus_beta_recip_bits,
    uint32_t softplus_threshold_bits,
    uint32_t time_step_floor_bits,
    uint32_t time_step_max_bits) {

    // ── Stage 1: dt_eff = clamp(softplus(dt + dt_bias), floor, max)
    cb_wait_front(cb_dt, ONE_TILE);
    cb_wait_front(cb_dt_bias, ONE_TILE);

    pack_reconfig_data_format(cb_dt_eff_scratch);
    reconfig_data_format(cb_dt, cb_dt_bias);
    add_tiles_init(cb_dt, cb_dt_bias);
    cb_reserve_back(cb_dt_eff_scratch, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_dt, cb_dt_bias, 0, 0, 0);
    softplus_tile_init();
    softplus_tile(0, softplus_beta_bits, softplus_beta_recip_bits,
                  softplus_threshold_bits);
    clamp_tile_init();
    clamp_tile(0, time_step_floor_bits, time_step_max_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_dt_eff_scratch);
    tile_regs_release();
    cb_push_back(cb_dt_eff_scratch, ONE_TILE);

    cb_pop_front(cb_dt, ONE_TILE);
    cb_pop_front(cb_dt_bias, ONE_TILE);

    // ── Stage 2 + 3: decay = exp(A * dt_eff) → overwrite cb_decay_inout
    // CB queue semantics: cb_decay_inout currently has [A]. We compute
    // [A * dt_eff → exp] in dest, push as a new tile, then pop the OLD
    // A from the front, leaving [decay] for the downstream mul_decay_state_to.
    cb_wait_front(cb_dt_eff_scratch, ONE_TILE);
    cb_wait_front(cb_decay_inout, ONE_TILE);

    pack_reconfig_data_format(cb_decay_inout);
    reconfig_data_format(cb_decay_inout, cb_dt_eff_scratch);
    mul_tiles_init(cb_decay_inout, cb_dt_eff_scratch);
    cb_reserve_back(cb_decay_inout, ONE_TILE);

    tile_regs_acquire();
    mul_tiles(cb_decay_inout, cb_dt_eff_scratch, 0, 0, 0);  // A * dt_eff
    exp_tile_init();
    exp_tile(0);                                              // exp(...)
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_decay_inout);                             // push new decay
    tile_regs_release();
    cb_push_back(cb_decay_inout, ONE_TILE);
    // Queue is now [OLD_A, decay]. Pop the OLD_A so the front exposes decay.
    cb_pop_front(cb_decay_inout, ONE_TILE);

    cb_pop_front(cb_dt_eff_scratch, ONE_TILE);
}

// mul_decay_state_to(cb_state, cb_decay, head_dim_tile, ssm_state_tile,
//                    cb_state_scaled)
//
// Implements (for one tile of state):
//   state_scaled[head_dim_tile, ssm_state_tile, :, :] =
//       state[head_dim_tile, ssm_state_tile, :, :] * decay   (broadcast scalar)
//
// state is laid out as [head_dim_tiles=2 * ssm_state_tiles=4 = 8 tiles per
// (batch, head)]. The tile_index is `head_dim_tile * ssm_state_tiles +
// ssm_state_tile`. Caller's loop iterates over both dims.
//
// REUSE: this is a direct fork of
//   qwen36_gdn_decode_owned/device/kernels/compute/qwen36_gdn_decode_owned.cpp
//   line 57: `mul_alpha_scalar_tile_indexed`
// renamed for Mamba2 semantics. Same LLK call (`mul_tiles_bcast_scalar`),
// same tile_regs pattern.
FORCE_INLINE void mul_decay_state_to(
    uint32_t cb_state,
    uint32_t cb_decay,
    uint32_t tile_index,
    uint32_t cb_state_scaled) {
    reconfig_data_format(cb_state, cb_decay);
    pack_reconfig_data_format(cb_state_scaled);
    mul_tiles_bcast_scalar_init_short(cb_state, cb_decay);
    cb_wait_front(cb_state, tile_index + 1);
    cb_wait_front(cb_decay, ONE_TILE);
    cb_reserve_back(cb_state_scaled, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_state, cb_decay, tile_index, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_state_scaled);
    tile_regs_release();

    cb_push_back(cb_state_scaled, ONE_TILE);
}

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
    // Softplus + clamp config (passed as float32 bits per LLK calling convention).
    // Defaults match Nemotron-3 config.json: beta=1.0, threshold=20.0,
    // time_step_floor=1e-4, time_step_max=0.1. The host program_factory will
    // emit these as runtime args once the new factory lands; for day-3 we
    // hardcode the bit patterns.
    constexpr uint32_t SOFTPLUS_BETA_BITS        = 0x3f800000u;  // 1.0f
    constexpr uint32_t SOFTPLUS_BETA_RECIP_BITS  = 0x3f800000u;  // 1.0f
    constexpr uint32_t SOFTPLUS_THRESHOLD_BITS   = 0x41a00000u;  // 20.0f
    constexpr uint32_t TIME_STEP_FLOOR_BITS      = 0x38d1b717u;  // 1e-4f
    constexpr uint32_t TIME_STEP_MAX_BITS        = 0x3dcccccdu;  // 0.1f
    (void)cb_z;  // not consumed in the kernel; pass-through to caller (decision D10)

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
        else if (debug_mode == 2) {
            // ── Mode 2: state_out = decay * state_in, no input contribution ───
            // Pipeline:
            //   compute_decay(cb_A_log → cb_decay)               // packs A
            //   finalize_decay(cb_decay, cb_dt, cb_dt_bias, cb_dt_B)
            //                                                    // overwrites
            //                                                    // cb_decay with
            //                                                    // exp(dt_eff*A)
            //   loop: mul_decay_state_to over (head_dim, ssm_state) tiles
            //
            // Day-3.5 status: compute_decay + finalize_decay both functional.
            // Output of mode 2: state_out = decay * state_in (matches oracle's
            // decay-state-update without the input contribution term).
            //
            // cb_dt_B is reused as dt_eff scratch (decision D11 double-duty —
            // safe because debug_mode=2 does NOT consume cb_dt_B for its
            // intended dt_eff*B production purpose; that fires at mode 3+).
            compute_decay(cb_A_log, cb_decay);
            finalize_decay_with_dt_eff(
                cb_decay, cb_dt, cb_dt_bias, cb_dt_B,
                SOFTPLUS_BETA_BITS, SOFTPLUS_BETA_RECIP_BITS,
                SOFTPLUS_THRESHOLD_BITS,
                TIME_STEP_FLOOR_BITS, TIME_STEP_MAX_BITS);

            // state has shape [head_dim_tiles, ssm_state_tiles] tiles.
            // Tile index = head_dim_tile * ssm_state_tiles + ssm_state_tile.
            // Loop order: outer over head_dim (decision D5 — decay reused).
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                for (uint32_t s = 0; s < ssm_state_tiles; ++s) {
                    const uint32_t tile_idx = d * ssm_state_tiles + s;
                    mul_decay_state_to(cb_state_in, cb_decay, tile_idx, cb_state_out);
                }
            }

            // No y math in mode 2; sentinel fill so the writer drains its CB.
            for (uint32_t d = 0; d < head_dim_tiles; ++d) {
                fill_one(cb_y);
            }

            cb_pop_front(cb_state_in, head_dim_tiles * ssm_state_tiles);
            cb_pop_front(cb_decay, ONE_TILE);
            // cb_dt, cb_dt_bias, cb_A_log all popped by the helpers above.
            // cb_dt_B used as scratch and popped by finalize_decay.
        }
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
