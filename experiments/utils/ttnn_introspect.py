#!/usr/bin/env python3
"""Introspect ttnn for ops matching a keyword. Avoids inline `python -c`.

Usage:
  python experiments/utils/ttnn_introspect.py <keyword> [namespace]
  python experiments/utils/ttnn_introspect.py <keyword> [namespace] --doc

  namespace defaults to 'ttnn'; pass 'ttnn.experimental' / 'ttnn.transformer' etc.
  --doc prints __doc__ + inspect.signature for each match (for kwarg surface checks).
"""
import sys


def main():
    if len(sys.argv) < 2:
        print("usage: ttnn_introspect.py <keyword> [namespace=ttnn] [--doc]")
        sys.exit(2)
    args = [a for a in sys.argv[1:] if a != "--doc"]
    show_doc = "--doc" in sys.argv
    keyword = args[0].lower()
    ns = args[1] if len(args) > 1 else "ttnn"
    import importlib
    import inspect
    # importlib.import_module handles dotted paths AND lazy-imports
    # submodules (e.g. `ttnn.experimental`); plain `getattr` does not.
    try:
        mod = importlib.import_module(ns)
    except ImportError:
        # Fallback for non-importable attribute paths (e.g. a class method).
        parts = ns.split(".")
        mod = importlib.import_module(parts[0])
        for p in parts[1:]:
            mod = getattr(mod, p)
    matches = [x for x in dir(mod) if keyword in x.lower()]
    print(f"{ns}: {len(matches)} matches for '{keyword}'")
    for m in matches:
        print(f"  {m}")
        if show_doc:
            obj = getattr(mod, m)
            try:
                sig = inspect.signature(obj)
                print(f"    signature: {sig}")
            except (TypeError, ValueError):
                pass
            doc = (obj.__doc__ or "").strip()
            if doc:
                for line in doc.splitlines()[:40]:
                    print(f"    | {line}")


if __name__ == "__main__":
    main()
