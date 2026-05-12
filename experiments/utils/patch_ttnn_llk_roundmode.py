#!/usr/bin/env python3
"""
Permanent utility — patch ttnn's LLK SFPU headers for the
'int → sfpi::RoundMode' template-body bug.

# What the bug is

ttnn ships kernel source as C++ headers. Recently `sfpi_lib.h` tightened
the round-mode parameter of conversion functions from plain `int` to a
strongly-typed `enum class RoundMode`. Call sites in the LLK SFPU
headers (`ckernel_sfpu_*.h`) still pass literal `0` where `RoundMode` is
now required:

    sfpi::float_to_int16(pow, 0)
    sfpi::int32_to_float(exp, 0)

The compiler refuses the implicit `int → RoundMode` conversion. Many
ttnn kernels include these headers (via SFPU umbrella include), so any
kernel build that hits a fresh template instantiation crashes:

    trisc1 build failed.
    cannot convert 'int' to 'sfpi::RoundMode' [-Wtemplate-body]

This blocks: 91r (per-layer cosine diff), 91q (substep dump),
91p --weight-dtype {bf16,fp32} (bf8 weight ablation).

# What we patch

In Blackhole's LLK directories under the ttnn wheel, we substitute
`, 0)` → `, sfpi::RoundMode::NearestEven)` at call sites of the
SFPI conversion functions (int32_to_float, float_to_int16, etc.).

The enum value `RoundMode::NearestEven == 0` per `sfpi_lib.h`, so the
patch preserves whatever runtime semantics the original `int 0` had.

# Reversibility

Before patching, we tar up all affected files to a timestamped backup
under ~/tt-xla/.cache/ttnn_llk_backup/<timestamp>/. Restore with
--restore <timestamp>.

# Usage

    # See what would change (dry-run, no writes)
    python experiments/utils/patch_ttnn_llk_roundmode.py --dry-run

    # Apply patch + back up first
    python experiments/utils/patch_ttnn_llk_roundmode.py --apply

    # List backups
    python experiments/utils/patch_ttnn_llk_roundmode.py --list-backups

    # Restore from a specific backup
    python experiments/utils/patch_ttnn_llk_roundmode.py --restore <timestamp>

Run on qb2 (modifies the local ttnn install):
    cd ~/tt-xla && .venv/bin/python experiments/utils/patch_ttnn_llk_roundmode.py --apply
"""
import os, re, sys, time, shutil, glob, argparse
from pathlib import Path

LLK_ROOTS = [
    "/home/aditya/tt-xla/.venv/lib/python3.10/site-packages/ttnn/tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu",
    "/home/aditya/tt-xla/.venv/lib/python3.10/site-packages/ttnn/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu",
]
BACKUP_ROOT = os.path.expanduser("~/tt-xla/.cache/ttnn_llk_backup")

# Pattern: match SFPI conversion functions with literal 0 as second arg.
# Limited to single-identifier first-arg form (all current call sites match).
CONV_FNS = "(?:int32_to_float|float_to_int16|float_to_int32|float_to_uint8|float_to_uint16|float_to_fp16a|float_to_fp16b)"
PATTERN = re.compile(
    r"((?:sfpi::)?" + CONV_FNS + r"\(\s*[A-Za-z_][A-Za-z0-9_]*\s*),\s*0\s*\)"
)
REPLACEMENT = r"\1, sfpi::RoundMode::NearestEven)"


def find_target_files():
    """All .h files under LLK_ROOTS that contain at least one buggy call."""
    targets = []
    for root in LLK_ROOTS:
        if not os.path.isdir(root):
            print(f"WARN: root does not exist: {root}")
            continue
        for path in glob.glob(os.path.join(root, "*.h")):
            with open(path) as f:
                content = f.read()
            if PATTERN.search(content):
                targets.append(path)
    return sorted(targets)


def show_changes(path):
    """Print each line that would change (for dry-run)."""
    changes = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            new_line = PATTERN.sub(REPLACEMENT, line)
            if new_line != line:
                changes.append((lineno, line.rstrip("\n"), new_line.rstrip("\n")))
    return changes


def backup_files(targets, ts):
    """Copy each target into a timestamped backup tree."""
    backup_dir = os.path.join(BACKUP_ROOT, ts)
    os.makedirs(backup_dir, exist_ok=True)
    for src in targets:
        # Preserve directory structure under backup_dir relative to ttnn root
        rel = src.replace("/home/aditya/tt-xla/.venv/lib/python3.10/site-packages/ttnn/", "")
        dst = os.path.join(backup_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    print(f"  backed up {len(targets)} files → {backup_dir}")
    return backup_dir


def apply_patch(targets):
    """Apply PATTERN → REPLACEMENT in place."""
    total_changes = 0
    for path in targets:
        with open(path) as f:
            content = f.read()
        new_content, n = PATTERN.subn(REPLACEMENT, content)
        if n > 0:
            with open(path, "w") as f:
                f.write(new_content)
            total_changes += n
            print(f"  {path}: {n} change(s)")
    return total_changes


def restore_backup(timestamp):
    """Copy files from the given timestamped backup back to their originals."""
    backup_dir = os.path.join(BACKUP_ROOT, timestamp)
    if not os.path.isdir(backup_dir):
        print(f"backup not found: {backup_dir}")
        return 0
    restored = 0
    for root, _, files in os.walk(backup_dir):
        for fname in files:
            if not fname.endswith(".h"):
                continue
            backup_path = os.path.join(root, fname)
            rel = os.path.relpath(backup_path, backup_dir)
            orig = os.path.join(
                "/home/aditya/tt-xla/.venv/lib/python3.10/site-packages/ttnn",
                rel
            )
            shutil.copy2(backup_path, orig)
            restored += 1
    print(f"restored {restored} files from {backup_dir}")
    return restored


def list_backups():
    if not os.path.isdir(BACKUP_ROOT):
        print(f"no backups (dir does not exist: {BACKUP_ROOT})")
        return
    for ts in sorted(os.listdir(BACKUP_ROOT)):
        ts_dir = os.path.join(BACKUP_ROOT, ts)
        n = sum(1 for _ in glob.glob(os.path.join(ts_dir, "**/*.h"), recursive=True))
        print(f"  {ts}  ({n} files)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Find target files and print proposed changes; do NOT modify")
    p.add_argument("--apply", action="store_true",
                   help="Back up affected files then apply patch")
    p.add_argument("--restore", type=str, default=None, metavar="TIMESTAMP",
                   help="Restore from a specific backup timestamp")
    p.add_argument("--list-backups", action="store_true")
    args = p.parse_args()

    if args.list_backups:
        list_backups()
        return

    if args.restore:
        restore_backup(args.restore)
        return

    print("Scanning LLK SFPU headers for buggy `, 0)` call sites…")
    targets = find_target_files()
    print(f"found {len(targets)} affected files\n")

    total = 0
    for t in targets:
        changes = show_changes(t)
        total += len(changes)
        print(f"  {t}: {len(changes)} change(s)")
        for lineno, old, new in changes[:3]:
            print(f"    L{lineno}: {old.strip()}")
            print(f"        →: {new.strip()}")
        if len(changes) > 3:
            print(f"    … ({len(changes)-3} more not shown)")
    print(f"\nTOTAL changes: {total}")

    if args.dry_run:
        print("\n(dry run — no files modified)")
        return

    if not args.apply:
        print("\nPass --apply to back up and patch, or --dry-run to preview")
        return

    ts = time.strftime("%Y%m%d-%H%M%S")
    print(f"\nBacking up to {BACKUP_ROOT}/{ts}/…")
    backup_files(targets, ts)
    print(f"\nApplying patch…")
    n = apply_patch(targets)
    print(f"\n✓ patched {n} call sites across {len(targets)} files")
    print(f"  backup at {BACKUP_ROOT}/{ts}/")
    print(f"  restore: python experiments/utils/patch_ttnn_llk_roundmode.py --restore {ts}")


if __name__ == "__main__":
    main()
