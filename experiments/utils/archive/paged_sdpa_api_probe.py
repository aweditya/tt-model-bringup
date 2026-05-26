#!/usr/bin/env python3
"""
API discovery probe for ttnn.transformer.paged_scaled_dot_product_attention_decode.

Just dumps the signature, docstring, and any related types/constants. No
actual call attempted — that's the correctness probe (next step).

Run on qb1 (qb2 busy with C'0.6 gates):
    cd ~/tt-xla && .venv/bin/python experiments/utils/paged_sdpa_api_probe.py
"""
import sys
import inspect
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def dump(name, fn):
    if fn is None:
        print(f"  {name} — NOT FOUND")
        return
    print(f"\n{name}")
    print("-" * len(name))
    try:
        sig = inspect.signature(fn)
        print(f"  signature: {sig}")
    except (ValueError, TypeError) as e:
        print(f"  (signature unavailable: {e})")
    doc = inspect.getdoc(fn)
    if doc:
        print("  doc:")
        for line in doc.split("\n"):
            print(f"    | {line}")


def main():
    print("=" * 64)
    print("API discovery: paged + chunked SDPA decode")
    print("=" * 64)

    candidates = [
        ("ttnn.transformer.paged_scaled_dot_product_attention_decode",
         getattr(ttnn.transformer, "paged_scaled_dot_product_attention_decode", None)),
        ("ttnn.transformer.chunked_scaled_dot_product_attention",
         getattr(ttnn.transformer, "chunked_scaled_dot_product_attention", None)),
        ("ttnn.transformer.windowed_scaled_dot_product_attention",
         getattr(ttnn.transformer, "windowed_scaled_dot_product_attention", None)),
        ("ttnn.transformer.scaled_dot_product_attention",
         getattr(ttnn.transformer, "scaled_dot_product_attention", None)),
        ("ttnn.transformer.scaled_dot_product_attention_decode",
         getattr(ttnn.transformer, "scaled_dot_product_attention_decode", None)),
    ]

    for name, fn in candidates:
        dump(name, fn)

    # Companion cache ops
    print("\n" + "=" * 64)
    print("Companion paged-cache ops in ttnn.experimental")
    print("=" * 64)
    for n in sorted(dir(ttnn.experimental)):
        if "paged" in n.lower():
            fn = getattr(ttnn.experimental, n)
            dump(f"ttnn.experimental.{n}", fn)


if __name__ == "__main__":
    main()
