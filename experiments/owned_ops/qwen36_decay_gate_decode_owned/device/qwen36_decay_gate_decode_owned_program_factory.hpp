// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <vector>

#include "qwen36_decay_gate_decode_owned_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct Qwen36DecayGateDecodeOwnedSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores;
    uint32_t num_cores{};
};

struct Qwen36DecayGateDecodeOwnedProgramFactory {
    using shared_variables_t = Qwen36DecayGateDecodeOwnedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const Qwen36DecayGateDecodeOwnedParams& operation_attributes,
        const Qwen36DecayGateDecodeOwnedInputs& tensor_args,
        std::tuple<Tensor, Tensor>& output_tensors);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const Qwen36DecayGateDecodeOwnedParams& operation_attributes,
        const Qwen36DecayGateDecodeOwnedInputs& tensor_args,
        std::tuple<Tensor, Tensor>& output_tensors);
};

}  // namespace ttnn::experimental::prim
