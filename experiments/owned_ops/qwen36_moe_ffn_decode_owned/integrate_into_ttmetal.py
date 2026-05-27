#!/usr/bin/env python3
"""Install qwen36_moe_ffn_decode_owned into a TT-Metal checkout.

Anchors on the qwen36_decay_gate_decode_owned entries (already installed
on qb1/qb2 via the decay_gate installer + the bulk qwen36 sync). Mirrors
experiments/owned_ops/qwen36_decay_gate_decode_owned/integrate_into_ttmetal.py.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OP_NAME = "qwen36_moe_ffn_decode_owned"
REL_OP_DIR = Path("ttnn/cpp/ttnn/operations/experimental/transformer") / OP_NAME

ANCHOR_OP = "qwen36_decay_gate_decode_owned"


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
        f"    {ANCHOR_OP}/device/kernels/*\n",
        f"    {OP_NAME}/device/kernels/*\n",
        f"transformer kernel glob ({OP_NAME})",
    )
    text = insert_before_once(
        text,
        f"            {ANCHOR_OP}/{ANCHOR_OP}.hpp\n",
        f"            {OP_NAME}/{OP_NAME}.hpp\n",
        f"transformer api header ({OP_NAME})",
    )
    text = insert_before_once(
        text,
        f"        {ANCHOR_OP}/device/{ANCHOR_OP}_device_operation.cpp\n",
        f"        {OP_NAME}/device/{OP_NAME}_device_operation.cpp\n"
        f"        {OP_NAME}/device/{OP_NAME}_program_factory.cpp\n"
        f"        {OP_NAME}/{OP_NAME}.cpp\n",
        f"transformer private sources ({OP_NAME})",
    )
    unity_skip = (
        f"set_source_files_properties(\n"
        f"    {OP_NAME}/device/{OP_NAME}_device_operation.cpp\n"
        f"    {OP_NAME}/device/{OP_NAME}_program_factory.cpp\n"
        f"    PROPERTIES\n"
        f"        SKIP_UNITY_BUILD_INCLUSION\n"
        f"            ON\n"
        f")\n\n"
    )
    text = insert_before_once(
        text,
        "target_include_directories(ttnn_op_experimental_transformer PRIVATE ${FixmeOpIncDirs})\n",
        unity_skip,
        f"unity-build exclusion ({OP_NAME})",
    )
    write_text(path, text, dry_run)


def patch_ttnn_cmake(tt_metal: Path, dry_run: bool) -> None:
    path = tt_metal / "ttnn/CMakeLists.txt"
    text = read_text(path)
    text = insert_before_once(
        text,
        f"    ${{CMAKE_CURRENT_SOURCE_DIR}}/cpp/ttnn/operations/experimental/transformer/{ANCHOR_OP}/{ANCHOR_OP}_nanobind.cpp\n",
        f"    ${{CMAKE_CURRENT_SOURCE_DIR}}/cpp/ttnn/operations/experimental/transformer/"
        f"{OP_NAME}/{OP_NAME}_nanobind.cpp\n",
        f"ttnn nanobind source ({OP_NAME})",
    )
    write_text(path, text, dry_run)


def patch_experimental_nanobind(tt_metal: Path, dry_run: bool) -> None:
    path = tt_metal / "ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp"
    text = read_text(path)
    text = insert_before_once(
        text,
        f'#include "ttnn/operations/experimental/transformer/{ANCHOR_OP}/{ANCHOR_OP}_nanobind.hpp"\n',
        f'#include "ttnn/operations/experimental/transformer/{OP_NAME}/{OP_NAME}_nanobind.hpp"\n',
        f"experimental nanobind include ({OP_NAME})",
    )
    text = insert_before_once(
        text,
        f"    {ANCHOR_OP}::detail::bind_{ANCHOR_OP}(mod);\n",
        f"    {OP_NAME}::detail::bind_{OP_NAME}(mod);\n",
        f"experimental nanobind registration ({OP_NAME})",
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
