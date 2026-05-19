// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <vector>

#include "qwen36_conv1d_decode_owned_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct Qwen36Conv1dDecodeOwnedSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores;
    uint32_t num_cores{};
    uint32_t g1_numcores{};
    uint32_t g2_numcores{};
    uint32_t num_tiles_per_core_group_1{};
    uint32_t num_tiles_per_core_group_2{};
};

struct Qwen36Conv1dDecodeOwnedProgramFactory {
    using shared_variables_t = Qwen36Conv1dDecodeOwnedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const Qwen36Conv1dDecodeOwnedParams& operation_attributes,
        const Qwen36Conv1dDecodeOwnedInputs& tensor_args,
        std::tuple<Tensor, Tensor, Tensor, Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const Qwen36Conv1dDecodeOwnedParams& operation_attributes,
        const Qwen36Conv1dDecodeOwnedInputs& tensor_args,
        std::tuple<Tensor, Tensor, Tensor, Tensor>& output_tensors);
};

}  // namespace ttnn::experimental::prim
