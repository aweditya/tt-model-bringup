// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for qwen36_decay_gate_decode_owned.
//
// Loads exactly 1 tile each from a, b, dt_bias, A_log into their compute CBs.
// No shift logic (decay/gate is stateless).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"

namespace {

template <typename Accessor>
FORCE_INLINE void read_tile_to_cb(const Accessor& accessor, uint32_t tile_id, uint32_t cb_id, uint32_t tile_bytes) {
    cb_reserve_back(cb_id, 1);
    noc_async_read(accessor.get_noc_addr(tile_id), get_write_ptr(cb_id), tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dt_bias = get_compile_time_arg_val(2);
    constexpr uint32_t cb_A_log = get_compile_time_arg_val(3);

    constexpr auto a_args = TensorAccessorArgs<4>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto dt_bias_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto A_log_args = TensorAccessorArgs<dt_bias_args.next_compile_time_args_offset()>();

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t dt_bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t A_log_addr = get_arg_val<uint32_t>(3);

    const auto a_accessor = TensorAccessor(a_args, a_addr);
    const auto b_accessor = TensorAccessor(b_args, b_addr);
    const auto dt_bias_accessor = TensorAccessor(dt_bias_args, dt_bias_addr);
    const auto A_log_accessor = TensorAccessor(A_log_args, A_log_addr);

    const uint32_t tile_bytes = get_tile_size(cb_a);

    // Single tile per input.
    read_tile_to_cb(a_accessor, 0, cb_a, tile_bytes);
    read_tile_to_cb(b_accessor, 0, cb_b, tile_bytes);
    read_tile_to_cb(dt_bias_accessor, 0, cb_dt_bias, tile_bytes);
    read_tile_to_cb(A_log_accessor, 0, cb_A_log, tile_bytes);
}
