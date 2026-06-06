#!/usr/bin/env python3
"""Generate qwen36_topk_owned source files from a cached copy of ttnn::topk.

This is a SCAFFOLDING helper used during initial bring-up. It is NOT the
integration script (see integrate_into_ttmetal.py for the
copy-into-tt-metal step).

What this does:
  1. Reads from a cached snapshot of
     `ttnn/cpp/ttnn/operations/reduction/topk/` (rsync'd from qb1 into
     `.cache/qb1_topk_src/`).
  2. Writes renamed files into this directory.
  3. Rewrites identifiers, namespaces, header paths, and kernel-path
     strings from `topk` → `qwen36_topk_owned`.
  4. Hardcodes `stable_sort=true` at every LLK call site in the compute
     kernels (the whole point of this fork).

Idempotent: running twice overwrites the same outputs.

Run locally (no device required):
    python3 experiments/owned_ops/qwen36_topk_owned/_fork_from_upstream.py \
        --upstream .cache/qb1_topk_src

Why this is a helper instead of hand-edited files:
  - 26 source files, mostly verbatim with rename
  - we want diff against upstream to be trivially auditable
  - the LLK stable_sort flag flips are isolated to one helper file
    (topk_common_funcs.hpp → qwen36_topk_owned_common_funcs.hpp) plus the
    single-core compute kernel; everything else is rename-only.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# Map source filename → destination filename. None means "skip".
FILE_RENAMES: dict[str, str | None] = {
    "topk.hpp":                                            "qwen36_topk_owned.hpp",
    "topk.cpp":                                            "qwen36_topk_owned.cpp",
    # nanobind files are HAND-WRITTEN (not forked) because the upstream nanobind
    # binds `&ttnn::topk` into the top-level `ttnn.` module, while we bind
    # `&ttnn::qwen36_topk_owned` into `ttnn.experimental.` (matches the
    # nemotron3_mamba2_decode_owned pattern). Skipped here.
    "topk_nanobind.hpp":                                   None,
    "topk_nanobind.cpp":                                   None,
    "device/topk_constants.hpp":                           "device/qwen36_topk_owned_constants.hpp",
    "device/topk_device_operation_types.hpp":              "device/qwen36_topk_owned_device_operation_types.hpp",
    "device/topk_device_operation.hpp":                    "device/qwen36_topk_owned_device_operation.hpp",
    "device/topk_device_operation.cpp":                    "device/qwen36_topk_owned_device_operation.cpp",
    "device/topk_single_core_program_factory.hpp":         "device/qwen36_topk_owned_single_core_program_factory.hpp",
    "device/topk_single_core_program_factory.cpp":         "device/qwen36_topk_owned_single_core_program_factory.cpp",
    "device/topk_multi_core_program_factory.hpp":          "device/qwen36_topk_owned_multi_core_program_factory.hpp",
    "device/topk_multi_core_program_factory.cpp":          "device/qwen36_topk_owned_multi_core_program_factory.cpp",
    "device/topk_utils.hpp":                               "device/qwen36_topk_owned_utils.hpp",
    "device/topk_utils.cpp":                               "device/qwen36_topk_owned_utils.cpp",
    "device/kernels/compute/topk.cpp":                     "device/kernels/compute/qwen36_topk_owned.cpp",
    "device/kernels/compute/topk_local.cpp":               "device/kernels/compute/qwen36_topk_owned_local.cpp",
    "device/kernels/compute/topk_final.cpp":               "device/kernels/compute/qwen36_topk_owned_final.cpp",
    "device/kernels/compute/topk_common_funcs.hpp":        "device/kernels/compute/qwen36_topk_owned_common_funcs.hpp",
    "device/kernels/dataflow/topk_dataflow_common.hpp":    "device/kernels/dataflow/qwen36_topk_owned_dataflow_common.hpp",
    # Dataflow kernels: keep filenames the same since they are byte-identical to upstream.
    # But the PFs reference them by path; we rename them too for consistency.
    "device/kernels/dataflow/reader_create_index_tensor.cpp": "device/kernels/dataflow/qwen36_topk_owned_reader_create_index_tensor.cpp",
    "device/kernels/dataflow/reader_create_index_local_topk.cpp": "device/kernels/dataflow/qwen36_topk_owned_reader_create_index_local_topk.cpp",
    "device/kernels/dataflow/reader_final_topk.cpp":        "device/kernels/dataflow/qwen36_topk_owned_reader_final_topk.cpp",
    "device/kernels/dataflow/writer_binary_interleaved.cpp":"device/kernels/dataflow/qwen36_topk_owned_writer_binary_interleaved.cpp",
    "device/kernels/dataflow/writer_local_topk.cpp":        "device/kernels/dataflow/qwen36_topk_owned_writer_local_topk.cpp",
    "device/kernels/dataflow/writer_final_topk.cpp":        "device/kernels/dataflow/qwen36_topk_owned_writer_final_topk.cpp",
    "docs/TopK.md":                                         None,  # skip
}


# Text substitutions applied to every file (order-sensitive: longer patterns first).
TEXT_SUBS: list[tuple[str, str]] = [
    # Header-path references (e.g. #include lines) — install dir.
    ("ttnn/operations/reduction/topk/device/topk_device_operation_types.hpp",
     "ttnn/operations/experimental/transformer/qwen36_topk_owned/device/qwen36_topk_owned_device_operation_types.hpp"),
    ("ttnn/operations/reduction/topk/device/topk_single_core_program_factory.hpp",
     "ttnn/operations/experimental/transformer/qwen36_topk_owned/device/qwen36_topk_owned_single_core_program_factory.hpp"),
    ("ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.hpp",
     "ttnn/operations/experimental/transformer/qwen36_topk_owned/device/qwen36_topk_owned_multi_core_program_factory.hpp"),
    ("ttnn/operations/reduction/topk/device/topk_device_operation.hpp",
     "ttnn/operations/experimental/transformer/qwen36_topk_owned/device/qwen36_topk_owned_device_operation.hpp"),
    ("ttnn/operations/reduction/topk/device/topk_constants.hpp",
     "ttnn/operations/experimental/transformer/qwen36_topk_owned/device/qwen36_topk_owned_constants.hpp"),
    ("ttnn/operations/reduction/topk/device/topk_utils.hpp",
     "ttnn/operations/experimental/transformer/qwen36_topk_owned/device/qwen36_topk_owned_utils.hpp"),
    ("ttnn/operations/reduction/topk/topk.hpp",
     "ttnn/operations/experimental/transformer/qwen36_topk_owned/qwen36_topk_owned.hpp"),
    # Kernel-path strings used by CreateKernel calls in the PFs.
    ("ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp",
     "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/device/kernels/compute/qwen36_topk_owned.cpp"),
    ("ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_local.cpp",
     "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/device/kernels/compute/qwen36_topk_owned_local.cpp"),
    ("ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_final.cpp",
     "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/device/kernels/compute/qwen36_topk_owned_final.cpp"),
    ("ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_common_funcs.hpp",
     "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/device/kernels/compute/qwen36_topk_owned_common_funcs.hpp"),
    ("ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_tensor.cpp",
     "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/device/kernels/dataflow/qwen36_topk_owned_reader_create_index_tensor.cpp"),
    ("ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_local_topk.cpp",
     "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/device/kernels/dataflow/qwen36_topk_owned_reader_create_index_local_topk.cpp"),
    ("ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp",
     "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/device/kernels/dataflow/qwen36_topk_owned_reader_final_topk.cpp"),
    ("ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_binary_interleaved.cpp",
     "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/device/kernels/dataflow/qwen36_topk_owned_writer_binary_interleaved.cpp"),
    ("ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_local_topk.cpp",
     "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/device/kernels/dataflow/qwen36_topk_owned_writer_local_topk.cpp"),
    ("ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_final_topk.cpp",
     "ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/device/kernels/dataflow/qwen36_topk_owned_writer_final_topk.cpp"),
    # Local same-directory includes (e.g. "topk_device_operation_types.hpp" → renamed).
    ('"topk_device_operation_types.hpp"', '"qwen36_topk_owned_device_operation_types.hpp"'),
    ('"topk_single_core_program_factory.hpp"', '"qwen36_topk_owned_single_core_program_factory.hpp"'),
    ('"topk_multi_core_program_factory.hpp"', '"qwen36_topk_owned_multi_core_program_factory.hpp"'),
    ('"topk_device_operation.hpp"', '"qwen36_topk_owned_device_operation.hpp"'),
    ('"topk_constants.hpp"', '"qwen36_topk_owned_constants.hpp"'),
    ('"topk_utils.hpp"', '"qwen36_topk_owned_utils.hpp"'),
    ('"topk_dataflow_common.hpp"', '"qwen36_topk_owned_dataflow_common.hpp"'),
    ('"topk_common_funcs.hpp"', '"qwen36_topk_owned_common_funcs.hpp"'),
    ('"topk_nanobind.hpp"', '"qwen36_topk_owned_nanobind.hpp"'),
    ('"topk.hpp"', '"qwen36_topk_owned.hpp"'),
    # Namespace renames.
    ("namespace ttnn::operations::reduction::topk",
     "namespace ttnn::operations::experimental::qwen36_topk_owned"),
    ("ttnn::operations::reduction::topk::",
     "ttnn::operations::experimental::qwen36_topk_owned::"),
    # Relative form inside `namespace ttnn { ... }` (no leading `ttnn::`).
    ("operations::reduction::topk::",
     "operations::experimental::qwen36_topk_owned::"),
    ("namespace ttnn::operations::reduction::detail",
     "namespace ttnn::operations::experimental::qwen36_topk_owned::detail"),
    # Top-level free-function rename `ttnn::topk` → `ttnn::experimental::qwen36_topk_owned`.
    # We need the declaration AND the definition AND the prim:: alias. Done below in code.
    # Struct rename inside reduction::topk namespace.
    ("struct ExecuteTopK", "struct ExecuteQwen36TopkOwned"),
    # `bind_reduction_topk_operation` symbol — match nemotron pattern `bind_<op>`.
    ("bind_reduction_topk_operation", "bind_qwen36_topk_owned"),
    # Prim-level helpers.
    ("struct TopKDeviceOperation", "struct Qwen36TopkOwnedDeviceOperation"),
    ("struct TopKSingleCoreSharedVariables", "struct Qwen36TopkOwnedSingleCoreSharedVariables"),
    ("struct TopKSingleCoreProgramFactory", "struct Qwen36TopkOwnedSingleCoreProgramFactory"),
    ("struct TopKMultiCoreSharedVariables", "struct Qwen36TopkOwnedMultiCoreSharedVariables"),
    ("struct TopKMultiCoreProgramFactory", "struct Qwen36TopkOwnedMultiCoreProgramFactory"),
    ("TopKSingleCoreProgramFactory", "Qwen36TopkOwnedSingleCoreProgramFactory"),
    ("TopKMultiCoreProgramFactory", "Qwen36TopkOwnedMultiCoreProgramFactory"),
    ("TopKSingleCoreSharedVariables", "Qwen36TopkOwnedSingleCoreSharedVariables"),
    ("TopKMultiCoreSharedVariables", "Qwen36TopkOwnedMultiCoreSharedVariables"),
    ("TopKDeviceOperation", "Qwen36TopkOwnedDeviceOperation"),
    ("TopKCoreConfig", "Qwen36TopkOwnedCoreConfig"),
    ("find_topk_core_config", "find_qwen36_topk_owned_core_config"),
    ("verify_multi_core_cost", "qwen36_topk_owned_verify_multi_core_cost"),
    ("verify_single_core_cost", "qwen36_topk_owned_verify_single_core_cost"),
    ("largest_power_of_two", "qwen36_topk_owned_largest_power_of_two"),
    ("TopkParams", "Qwen36TopkOwnedParams"),
    ("TopkInputs", "Qwen36TopkOwnedInputs"),
    # ttnn::prim::topk → ttnn::prim::qwen36_topk_owned (free function in prim:: namespace).
    # We keep the prim function in `ttnn::prim` (matches upstream layout) — no need to
    # move to `ttnn::experimental::prim` since the struct name `Qwen36TopkOwnedDeviceOperation`
    # already disambiguates from upstream's `TopKDeviceOperation`.
    ("ttnn::prim::topk", "ttnn::prim::qwen36_topk_owned"),
    # The prim:: function declaration itself.
    # The file declares `std::tuple<...> topk(...)` inside `namespace ttnn::prim { ... }`.
    # We rename both signature occurrences via a unique anchor.
    # Documentation strings / docstring fragments containing "topk" are preserved.
]


# LLK stable_sort flag flips. Applied only to specific files.
LLK_STABLE_SORT_FLIPS: dict[str, list[tuple[str, str]]] = {
    "device/kernels/compute/qwen36_topk_owned.cpp": [
        # Single-core kernel: one direct topk_local_sort call.
        ("ckernel::topk_local_sort(0, (int)!largest, end_phase);",
         "ckernel::topk_local_sort</*stable_sort=*/true>(0, (int)!largest, end_phase);"),
    ],
    "device/kernels/compute/qwen36_topk_owned_common_funcs.hpp": [
        ("ckernel::topk_local_sort(0, (int)ascending, end_phase);",
         "ckernel::topk_local_sort</*stable_sort=*/true>(0, (int)ascending, end_phase);"),
        ("ckernel::topk_rebuild(0, (uint32_t)ascending, m_iter, K, logk, target_tiles_is_one);",
         "ckernel::topk_rebuild</*stable_sort=*/true>(0, (uint32_t)ascending, m_iter, K, logk, target_tiles_is_one);"),
        ("ckernel::topk_merge<false>(0, m_iter, K);",
         "ckernel::topk_merge</*idir=*/false, /*stable_sort=*/true>(0, m_iter, K);"),
        ("ckernel::topk_merge<true>(0, m_iter, K);",
         "ckernel::topk_merge</*idir=*/true, /*stable_sort=*/true>(0, m_iter, K);"),
    ],
}


def rewrite(text: str) -> str:
    for needle, sub in TEXT_SUBS:
        text = text.replace(needle, sub)
    return text


def fork_one(src: Path, dst: Path, rel: str) -> None:
    if src.suffix in {".hpp", ".cpp", ".h"} or src.name.endswith(".md"):
        text = src.read_text(encoding="utf-8")
        text = rewrite(text)

        # ttnn::prim::topk(...) declaration in the device-op .hpp lives inside
        # `namespace ttnn::prim { ... }`. We need the bare token `topk` →
        # `qwen36_topk_owned` for the prototype only, but NOT for occurrences
        # inside `ttnn::operations::reduction::topk` (already renamed) or
        # documentation comments. To stay safe, target the specific declaration
        # via its unique signature prefix.
        text = text.replace(
            "std::tuple<ttnn::Tensor, ttnn::Tensor> topk(",
            "std::tuple<ttnn::Tensor, ttnn::Tensor> qwen36_topk_owned(",
        )
        text = text.replace(
            "std::tuple<ttnn::Tensor, ttnn::Tensor> topk\n",
            "std::tuple<ttnn::Tensor, ttnn::Tensor> qwen36_topk_owned\n",
        )
        text = text.replace(
            "std::tuple<Tensor, Tensor> topk(",
            "std::tuple<Tensor, Tensor> qwen36_topk_owned(",
        )
        # User-facing free function `ttnn::topk`.
        text = text.replace(
            "std::vector<Tensor> topk(",
            "std::vector<Tensor> qwen36_topk_owned(",
        )
        # Apply LLK stable_sort flips for compute kernels.
        flips = LLK_STABLE_SORT_FLIPS.get(rel, [])
        for needle, sub in flips:
            if needle in text:
                text = text.replace(needle, sub, 1)
            else:
                print(f"  WARN: stable_sort flip needle not found in {rel}: {needle[:60]!r}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
    else:
        # Binary or other — just copy.
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path,
                        default=Path(__file__).resolve().parents[3] / ".cache" / "qb1_topk_src")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    upstream = args.upstream.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()

    if not (upstream / "topk.hpp").is_file():
        raise SystemExit(f"upstream cache not found at {upstream} (run rsync first)")

    for src_rel, dst_rel in FILE_RENAMES.items():
        if dst_rel is None:
            continue
        src = upstream / src_rel
        dst = out_dir / dst_rel
        if not src.exists():
            print(f"  skip (missing): {src_rel}")
            continue
        fork_one(src, dst, dst_rel)
        print(f"  fork  {src_rel} -> {dst_rel}")

    print(f"wrote files under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
