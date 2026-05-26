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

TTNN_ROOT = "/home/aditya/tt-xla/.venv/lib/python3.10/site-packages/ttnn"
# Walk the entire ttnn install — the buggy pattern lives in LLK headers under
# tt_metal/, AND in operation-specific compute kernels under ttnn/cpp/ttnn/operations/.
# Restricting to .h/.hpp/.cpp keeps the scan fast and avoids touching Python.
SCAN_EXTENSIONS = {".h", ".hpp", ".cpp"}
BACKUP_ROOT = os.path.expanduser("~/tt-xla/.cache/ttnn_llk_backup")

# Pattern: match SFPI conversion functions where the LAST argument is a literal 0.
# Some call sites have nested function calls / arithmetic in the first arg, so
# regex alone won't work — we use a paren-balance parser.
CONV_FNS = "(?:int32_to_float|float_to_int16|float_to_int32|float_to_uint8|float_to_uint16|float_to_fp16a|float_to_fp16b)"
CALL_START = re.compile(r"((?:sfpi::)?" + CONV_FNS + r")\(")


def _patch_text(text):
    """Find every SFPI conversion call ending in `, 0)` and rewrite the trailing
    literal to `, sfpi::RoundMode::NearestEven)`. Handles nested parens via a
    depth counter."""
    out = []
    i = 0
    n = len(text)
    n_changes = 0
    while i < n:
        m = CALL_START.search(text, i)
        if not m:
            out.append(text[i:])
            break
        # Emit text before the match unchanged
        out.append(text[i:m.start()])
        fn_name = m.group(1)
        out.append(fn_name + "(")
        # Walk from after the opening '(' until we find the matching close at depth 0
        j = m.end()
        depth = 1
        arg_start = j
        last_comma_at_depth_0 = None
        while j < n and depth > 0:
            c = text[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            elif c == "," and depth == 1:
                last_comma_at_depth_0 = j
            j += 1
        # j now points to the matching ')'
        if j >= n:
            # Unbalanced — emit rest as-is
            out.append(text[m.end():])
            break
        # The call body is text[arg_start:j].
        # Check if the part AFTER the last top-level comma is literal whitespace+'0'+whitespace
        if last_comma_at_depth_0 is not None:
            head = text[arg_start:last_comma_at_depth_0]   # all but last arg
            tail = text[last_comma_at_depth_0+1:j]          # last arg
            if tail.strip() == "0":
                # Replace with NearestEven
                out.append(head + ", sfpi::RoundMode::NearestEven")
                out.append(")")
                n_changes += 1
                i = j + 1
                continue
        # No replacement — emit the call as-is
        out.append(text[m.end():j+1])
        i = j + 1
    return "".join(out), n_changes


def find_changes_text(content):
    """Return (count_of_changes, list_of_(line_no, old_line, new_line))."""
    new_content, n = _patch_text(content)
    if n == 0:
        return 0, []
    # Compute diff by line
    old_lines = content.splitlines(keepends=False)
    new_lines = new_content.splitlines(keepends=False)
    diffs = []
    for i, (o, nn) in enumerate(zip(old_lines, new_lines)):
        if o != nn:
            diffs.append((i + 1, o, nn))
    return n, diffs


def find_target_files():
    """All .h/.hpp/.cpp files under TTNN_ROOT where the patcher would make changes."""
    targets = []
    if not os.path.isdir(TTNN_ROOT):
        print(f"WARN: ttnn root does not exist: {TTNN_ROOT}")
        return targets
    for root, _, files in os.walk(TTNN_ROOT):
        for fname in files:
            ext = os.path.splitext(fname)[1]
            if ext not in SCAN_EXTENSIONS:
                continue
            path = os.path.join(root, fname)
            try:
                with open(path) as f:
                    content = f.read()
            except (UnicodeDecodeError, IOError):
                continue
            n, _ = find_changes_text(content)
            if n > 0:
                targets.append(path)
    return sorted(targets)


def show_changes(path):
    """Return list of (lineno, old_line, new_line) for the patch."""
    with open(path) as f:
        content = f.read()
    _, diffs = find_changes_text(content)
    return diffs


def backup_files(targets, ts):
    """Copy each target into a timestamped backup tree."""
    backup_dir = os.path.join(BACKUP_ROOT, ts)
    os.makedirs(backup_dir, exist_ok=True)
    for src in targets:
        # Preserve directory structure under backup_dir relative to ttnn root
        rel = os.path.relpath(src, TTNN_ROOT)
        dst = os.path.join(backup_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    print(f"  backed up {len(targets)} files → {backup_dir}")
    return backup_dir


def apply_patch(targets):
    """Apply the balanced-parens patcher in place."""
    total_changes = 0
    for path in targets:
        with open(path) as f:
            content = f.read()
        new_content, n = _patch_text(content)
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
            ext = os.path.splitext(fname)[1]
            if ext not in SCAN_EXTENSIONS:
                continue
            backup_path = os.path.join(root, fname)
            rel = os.path.relpath(backup_path, backup_dir)
            orig = os.path.join(TTNN_ROOT, rel)
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
        n = sum(1 for _, _, fs in os.walk(ts_dir) for f in fs
                if os.path.splitext(f)[1] in SCAN_EXTENSIONS)
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
