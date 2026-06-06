#!/usr/bin/env python3
"""Install the owned qwen36_topk_owned op into a TT-Metal checkout.

Forked from nemotron3_mamba2_decode_owned/integrate_into_ttmetal.py.

This is a source/build integration helper. It does not open devices and it
does not run TTNN kernels — it copies the owned-op source tree into the
TT-Metal checkout and patches the relevant CMakeLists.txt + nanobind
registration files so the next `cmake --build` picks up the new op.

Idempotent: running twice is a no-op (the `insert_before_once` helper
detects already-applied patches).

Usage on qb1:
    cd ~/tt-xla
    python3 experiments/owned_ops/qwen36_topk_owned/integrate_into_ttmetal.py \
        --tt-metal ~/tenstorrent/tt-metal
    cmake --build ~/tenstorrent/tt-metal/build_Release --target ttnn -j8
    cp ~/tenstorrent/tt-metal/build_Release/ttnn/_ttnn.so    ~/tenstorrent/tt-metal/ttnn/ttnn/_ttnn.so
    cp ~/tenstorrent/tt-metal/build_Release/ttnn/_ttnncpp.so ~/tenstorrent/tt-metal/ttnn/ttnn/_ttnncpp.so

The forked op lives under
`ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_topk_owned/`
(same parent as the GDN/Mamba2 owned ops). Patches:
  1. experimental/transformer/CMakeLists.txt
  2. ttnn/CMakeLists.txt (nanobind source)
  3. experimental/experimental_nanobind.cpp (include + bind call)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OP_NAME = "qwen36_topk_owned"
REL_OP_DIR = Path("ttnn/cpp/ttnn/operations/experimental/transformer") / OP_NAME


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.write_text(text, encoding="utf-8")


def insert_before_once(text: str, needle: str, insertion: str, label: str) -> str:
    if insertion.strip() in text:
        return text
    if needle not in text:
        raise RuntimeError(f"Could not find insertion point for {label}: {needle!r}")
    return text.replace(needle, insertion + needle, 1)


def copy_op_source(source_dir: Path, tt_metal: Path, dry_run: bool) -> Path:
    destination = tt_metal / REL_OP_DIR
    if dry_run:
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc",
        # Local-only scaffolding helpers that aren't part of the ttnn build.
        "_fork_from_upstream.py",
        "integrate_into_ttmetal.py",
        "INTEGRATION.md", "README.md",
        "test_*.py", "benchmark_*.py",
        "sources.cmake",
    )
    shutil.copytree(source_dir, destination, ignore=ignore)
    return destination


def patch_transformer_cmake(tt_metal: Path, dry_run: bool) -> None:
    """Patch ttnn/cpp/ttnn/operations/experimental/transformer/CMakeLists.txt
    to (1) glob the new kernel sources, (2) install the public api header,
    (3) compile the device_op/program_factory/wrapper/utils sources,
    (4) exclude the device-op + program-factory + utils from unity-build.

    Anchor sites are all on existing GDN-owned insert points (stable, present
    in qb1 + qb2 tt-metal trees).
    """
    path = tt_metal / "ttnn/cpp/ttnn/operations/experimental/transformer/CMakeLists.txt"
    text = read_text(path)
    text = insert_before_once(
        text,
        "    qwen36_gdn_decode_owned/device/kernels/*\n",
        "    qwen36_topk_owned/device/kernels/*\n",
        "transformer kernel glob",
    )
    text = insert_before_once(
        text,
        "            qwen36_gdn_decode_owned/qwen36_gdn_decode_owned.hpp\n",
        "            qwen36_topk_owned/qwen36_topk_owned.hpp\n",
        "transformer api header",
    )
    text = insert_before_once(
        text,
        "        qwen36_gdn_decode_owned/device/qwen36_gdn_decode_owned_device_operation.cpp\n",
        "        qwen36_topk_owned/device/qwen36_topk_owned_device_operation.cpp\n"
        "        qwen36_topk_owned/device/qwen36_topk_owned_single_core_program_factory.cpp\n"
        "        qwen36_topk_owned/device/qwen36_topk_owned_multi_core_program_factory.cpp\n"
        "        qwen36_topk_owned/device/qwen36_topk_owned_utils.cpp\n"
        "        qwen36_topk_owned/qwen36_topk_owned.cpp\n",
        "transformer private sources",
    )
    unity_skip = """\
set_source_files_properties(
    qwen36_topk_owned/device/qwen36_topk_owned_device_operation.cpp
    qwen36_topk_owned/device/qwen36_topk_owned_single_core_program_factory.cpp
    qwen36_topk_owned/device/qwen36_topk_owned_multi_core_program_factory.cpp
    qwen36_topk_owned/device/qwen36_topk_owned_utils.cpp
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            ON
)

"""
    text = insert_before_once(
        text,
        "target_include_directories(ttnn_op_experimental_transformer PRIVATE ${FixmeOpIncDirs})\n",
        unity_skip,
        "qwen36 topk unity-build exclusion",
    )
    write_text(path, text, dry_run)


def patch_ttnn_cmake(tt_metal: Path, dry_run: bool) -> None:
    path = tt_metal / "ttnn/CMakeLists.txt"
    text = read_text(path)
    text = insert_before_once(
        text,
        "    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/"
        "qwen36_gdn_decode_owned/qwen36_gdn_decode_owned_nanobind.cpp\n",
        "    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/"
        "qwen36_topk_owned/qwen36_topk_owned_nanobind.cpp\n",
        "ttnn nanobind source",
    )
    write_text(path, text, dry_run)


def patch_experimental_nanobind(tt_metal: Path, dry_run: bool) -> None:
    path = tt_metal / "ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp"
    text = read_text(path)
    text = insert_before_once(
        text,
        '#include "ttnn/operations/experimental/transformer/'
        'qwen36_gdn_decode_owned/qwen36_gdn_decode_owned_nanobind.hpp"\n',
        '#include "ttnn/operations/experimental/transformer/'
        'qwen36_topk_owned/qwen36_topk_owned_nanobind.hpp"\n',
        "experimental nanobind include",
    )
    text = insert_before_once(
        text,
        "    qwen36_gdn_decode_owned::detail::bind_qwen36_gdn_decode_owned(mod);\n",
        "    qwen36_topk_owned::detail::bind_qwen36_topk_owned(mod);\n",
        "experimental nanobind registration",
    )
    write_text(path, text, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tt-metal", type=Path, default=Path("~/tenstorrent/tt-metal"))
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tt_metal = args.tt_metal.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    if not (tt_metal / "ttnn/CMakeLists.txt").is_file():
        raise RuntimeError(f"TT-Metal checkout not found at {tt_metal}")
    if not (source_dir / f"{OP_NAME}.cpp").is_file():
        raise RuntimeError(f"owned op source directory not found at {source_dir}")

    destination = copy_op_source(source_dir, tt_metal, args.dry_run)
    patch_transformer_cmake(tt_metal, args.dry_run)
    patch_ttnn_cmake(tt_metal, args.dry_run)
    patch_experimental_nanobind(tt_metal, args.dry_run)

    print(f"{'would install' if args.dry_run else 'installed'} {OP_NAME} to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
