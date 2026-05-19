// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for qwen36_decay_gate_decode_owned.
// Writes exactly 1 tile of cb_decay_out -> decay_addr and 1 tile of
// cb_beta_out -> beta_addr. No state shift.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"

namespace {

template <typename Accessor>
FORCE_INLINE void write_tile_from_cb(const Accessor& accessor, uint32_t tile_id, uint32_t cb_id, uint32_t tile_bytes) {
    noc_async_write(get_read_ptr(cb_id), accessor.get_noc_addr(tile_id), tile_bytes);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_decay_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_beta_out = get_compile_time_arg_val(1);

    constexpr auto decay_args = TensorAccessorArgs<2>();
    constexpr auto beta_args = TensorAccessorArgs<decay_args.next_compile_time_args_offset()>();

    const uint32_t decay_addr = get_arg_val<uint32_t>(0);
    const uint32_t beta_addr = get_arg_val<uint32_t>(1);

    const auto decay_accessor = TensorAccessor(decay_args, decay_addr);
    const auto beta_accessor = TensorAccessor(beta_args, beta_addr);

    const uint32_t tile_bytes = get_tile_size(cb_decay_out);

    cb_wait_front(cb_decay_out, 1);
    cb_wait_front(cb_beta_out, 1);

    write_tile_from_cb(decay_accessor, 0, cb_decay_out, tile_bytes);
    write_tile_from_cb(beta_accessor, 0, cb_beta_out, tile_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb_decay_out, 1);
    cb_pop_front(cb_beta_out, 1);
}
