#!/usr/bin/env python3
"""Install qwen36_decay_gate_decode_owned into a TT-Metal checkout.

Mirrors experiments/owned_ops/qwen36_conv1d_decode_owned/integrate_into_ttmetal.py.
Anchors on the qwen36_conv1d_decode_owned entries that the conv1d installer
already added.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OP_NAME = "qwen36_decay_gate_decode_owned"
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
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(source_dir, destination, ignore=ignore)
    return destination


def patch_transformer_cmake(tt_metal: Path, dry_run: bool) -> None:
    path = tt_metal / "ttnn/cpp/ttnn/operations/experimental/transformer/CMakeLists.txt"
    text = read_text(path)
    text = insert_before_once(
        text,
        "    qwen36_conv1d_decode_owned/device/kernels/*\n",
        "    qwen36_decay_gate_decode_owned/device/kernels/*\n",
        "transformer kernel glob (decay_gate)",
    )
    text = insert_before_once(
        text,
        "            qwen36_conv1d_decode_owned/qwen36_conv1d_decode_owned.hpp\n",
        "            qwen36_decay_gate_decode_owned/qwen36_decay_gate_decode_owned.hpp\n",
        "transformer api header (decay_gate)",
    )
    text = insert_before_once(
        text,
        "        qwen36_conv1d_decode_owned/device/qwen36_conv1d_decode_owned_device_operation.cpp\n",
        "        qwen36_decay_gate_decode_owned/device/qwen36_decay_gate_decode_owned_device_operation.cpp\n"
        "        qwen36_decay_gate_decode_owned/device/qwen36_decay_gate_decode_owned_program_factory.cpp\n"
        "        qwen36_decay_gate_decode_owned/qwen36_decay_gate_decode_owned.cpp\n",
        "transformer private sources (decay_gate)",
    )
    unity_skip = """\
set_source_files_properties(
    qwen36_decay_gate_decode_owned/device/qwen36_decay_gate_decode_owned_device_operation.cpp
    qwen36_decay_gate_decode_owned/device/qwen36_decay_gate_decode_owned_program_factory.cpp
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            ON
)

"""
    text = insert_before_once(
        text,
        "target_include_directories(ttnn_op_experimental_transformer PRIVATE ${FixmeOpIncDirs})\n",
        unity_skip,
        "qwen decay_gate unity-build exclusion",
    )
    write_text(path, text, dry_run)


def patch_ttnn_cmake(tt_metal: Path, dry_run: bool) -> None:
    path = tt_metal / "ttnn/CMakeLists.txt"
    text = read_text(path)
    text = insert_before_once(
        text,
        "    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/"
        "qwen36_conv1d_decode_owned/qwen36_conv1d_decode_owned_nanobind.cpp\n",
        "    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/experimental/transformer/"
        "qwen36_decay_gate_decode_owned/qwen36_decay_gate_decode_owned_nanobind.cpp\n",
        "ttnn nanobind source (decay_gate)",
    )
    write_text(path, text, dry_run)


def patch_experimental_nanobind(tt_metal: Path, dry_run: bool) -> None:
    path = tt_metal / "ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp"
    text = read_text(path)
    text = insert_before_once(
        text,
        '#include "ttnn/operations/experimental/transformer/qwen36_conv1d_decode_owned/qwen36_conv1d_decode_owned_nanobind.hpp"\n',
        '#include "ttnn/operations/experimental/transformer/qwen36_decay_gate_decode_owned/qwen36_decay_gate_decode_owned_nanobind.hpp"\n',
        "experimental nanobind include (decay_gate)",
    )
    text = insert_before_once(
        text,
        "    qwen36_conv1d_decode_owned::detail::bind_qwen36_conv1d_decode_owned(mod);\n",
        "    qwen36_decay_gate_decode_owned::detail::bind_qwen36_decay_gate_decode_owned(mod);\n",
        "experimental nanobind registration (decay_gate)",
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
    if not (source_dir / "qwen36_decay_gate_decode_owned.cpp").is_file():
        raise RuntimeError(f"owned op source directory not found at {source_dir}")

    destination = copy_op_source(source_dir, tt_metal, args.dry_run)
    patch_transformer_cmake(tt_metal, args.dry_run)
    patch_ttnn_cmake(tt_metal, args.dry_run)
    patch_experimental_nanobind(tt_metal, args.dry_run)

    print(f"{'would install' if args.dry_run else 'installed'} {OP_NAME} to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
