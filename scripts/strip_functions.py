#!/usr/bin/env python3
"""Delete named top-level functions from a Python file by AST line range.

Removes only the exact `def name(...): ...` line spans (lineno..end_lineno),
leaving every other line verbatim — no reformatting. Refuses unless each name is
found exactly once at module top level, so a typo can't silently no-op or clobber
the wrong thing. Dry-run by default; pass --apply to write.

    python scripts/strip_functions.py <file> --apply <fn1> <fn2> ...

Intended for mechanical dead-code excision (e.g. retired server probe handlers);
review the git diff after applying.
"""
import ast
import sys

args = sys.argv[1:]
apply = "--apply" in args
args = [a for a in args if a != "--apply"]
if len(args) < 2:
    sys.exit("usage: strip_functions.py <file> [--apply] <fn> [<fn> ...]")
path, names = args[0], args[1:]

src = open(path).read()
lines = src.splitlines(keepends=True)
tree = ast.parse(src)

found: dict[str, list[tuple[int, int]]] = {n: [] for n in names}
for node in tree.body:  # top level only — never reach into nested defs
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in found:
        found[node.name].append((node.lineno, node.end_lineno))

problems = [n for n, spans in found.items() if len(spans) != 1]
if problems:
    for n in problems:
        print(f"  {n}: found {len(found[n])} times (need exactly 1)")
    sys.exit("ERROR: refusing — every name must be a unique top-level def")

ranges = sorted((found[n][0] for n in names), reverse=True)
total = sum(end - start + 1 for start, end in ranges)
for start, end in sorted(found[n][0] for n in names):
    print(f"  {start:5d}..{end:<5d}  ({end - start + 1:4d} lines)")
print(f"{'APPLY' if apply else 'DRY-RUN'}: {len(ranges)} functions, {total} lines from {path}")

if apply:
    for start, end in ranges:  # bottom-up so earlier line numbers stay valid
        del lines[start - 1:end]
    open(path, "w").write("".join(lines))
    print(f"  wrote {path} ({len(lines)} lines remain)")
