// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "qwen36_moe_ffn_decode_owned_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct Qwen36MoeFfnDecodeOwnedSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores;
    uint32_t num_cores{};
    uint32_t hidden_tiles{};
};

struct Qwen36MoeFfnDecodeOwnedProgramFactory {
    using shared_variables_t = Qwen36MoeFfnDecodeOwnedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const Qwen36MoeFfnDecodeOwnedParams& operation_attributes,
        const Qwen36MoeFfnDecodeOwnedInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const Qwen36MoeFfnDecodeOwnedParams& operation_attributes,
        const Qwen36MoeFfnDecodeOwnedInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::experimental::prim
