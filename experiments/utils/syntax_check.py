#!/usr/bin/env python3
"""Tiny helper: ast.parse a file and report OK or fail. Avoids inline `python -c`.

Usage: python experiments/utils/syntax_check.py <path>
"""
import ast
import sys


def main():
    if len(sys.argv) < 2:
        print("usage: syntax_check.py <path>")
        sys.exit(2)
    path = sys.argv[1]
    try:
        ast.parse(open(path).read())
        print(f"OK: {path}")
    except SyntaxError as e:
        print(f"FAIL: {path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
