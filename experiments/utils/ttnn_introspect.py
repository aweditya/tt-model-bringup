#!/usr/bin/env python3
"""Introspect ttnn for ops matching a keyword. Avoids inline `python -c`.

Usage: python experiments/utils/ttnn_introspect.py <keyword> [namespace]
  namespace defaults to 'ttnn'; pass 'ttnn.experimental' / 'ttnn.transformer' etc.
"""
import sys


def main():
    if len(sys.argv) < 2:
        print("usage: ttnn_introspect.py <keyword> [namespace=ttnn]")
        sys.exit(2)
    keyword = sys.argv[1].lower()
    ns = sys.argv[2] if len(sys.argv) > 2 else "ttnn"
    parts = ns.split(".")
    import importlib
    mod = importlib.import_module(parts[0])
    for p in parts[1:]:
        mod = getattr(mod, p)
    matches = [x for x in dir(mod) if keyword in x.lower()]
    print(f"{ns}: {len(matches)} matches for '{keyword}'")
    for m in matches:
        print(f"  {m}")


if __name__ == "__main__":
    main()
