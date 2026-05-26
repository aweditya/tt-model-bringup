#!/usr/bin/env python3
"""
Probe ttnn for KV-cache-update primitives.

Before C'1 (replacing the numpy roundtrip in gated_attn_step), enumerate
what ttnn offers for on-device cache slot writes. Goal: find the minimum-
constraint API that lets us write K/V at a specific position WITHOUT
host roundtrip AND without the tile-alignment trap we hit at n_kv=4.

Candidates we know about:
  - ttnn.experimental.paged_update_cache  (failed at n_kv=4 due to sharded
                                           shard-shape (4, 256) not tile-aligned)
  - ttnn.experimental.update_cache  (non-paged variant if it exists)
  - ttnn.scatter / ttnn.scatter_update / ttnn.indexed_select  (general scatter)
  - on-device mask + multiply + add  (always works, may be slow)

Output: print signatures of all candidates so we can pick.
"""
import inspect
import ttnn


def main():
    print("=" * 64)
    print("KV cache update API discovery")
    print("=" * 64)

    candidates = [
        ("ttnn.experimental.paged_update_cache",
         getattr(getattr(ttnn, "experimental", None), "paged_update_cache", None)),
        ("ttnn.experimental.update_cache",
         getattr(getattr(ttnn, "experimental", None), "update_cache", None)),
        ("ttnn.experimental.paged_fill_cache",
         getattr(getattr(ttnn, "experimental", None), "paged_fill_cache", None)),
        ("ttnn.experimental.fill_cache",
         getattr(getattr(ttnn, "experimental", None), "fill_cache", None)),
        ("ttnn.scatter",      getattr(ttnn, "scatter", None)),
        ("ttnn.scatter_add",  getattr(ttnn, "scatter_add", None)),
        ("ttnn.index_put",    getattr(ttnn, "index_put", None)),
        ("ttnn.slice_set",    getattr(ttnn, "slice_set", None)),
        ("ttnn.assign",       getattr(ttnn, "assign", None)),
    ]

    for name, fn in candidates:
        if fn is None:
            continue
        print(f"\n• {name}")
        try:
            sig = inspect.signature(fn)
            print(f"    signature: {sig}")
        except (ValueError, TypeError) as e:
            print(f"    (signature unavailable: {e})")
        doc = inspect.getdoc(fn) or ""
        if doc:
            for line in doc.split("\n")[:15]:
                print(f"    | {line}")
            if len(doc.split("\n")) > 15:
                print(f"    | …({len(doc.split(chr(10))) - 15} more lines)")

    # Also enumerate all entries in ttnn.experimental that touch "cache"
    print("\n" + "=" * 64)
    print("All 'cache'-named entries in ttnn.experimental")
    print("=" * 64)
    if hasattr(ttnn, "experimental"):
        for name in sorted(dir(ttnn.experimental)):
            if "cache" in name.lower():
                fn = getattr(ttnn.experimental, name)
                kind = type(fn).__name__
                print(f"  ttnn.experimental.{name}  ({kind})")


if __name__ == "__main__":
    main()
