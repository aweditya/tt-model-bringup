#!/usr/bin/env python3
"""
P150 device-introspection helper. Discovers what `device` / mesh objects
expose in this ttnn build so the main bandwidth probe can be written safely.

Permanent (not a one-shot probe). Lives in experiments/utils/ per CLAUDE.md.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python -u experiments/utils/p150_device_introspect.py
"""
import sys

sys.stdout.reconfigure(line_buffering=True)

import ttnn


def dump_obj(name, obj):
    print(f"\n=== {name}: type={type(obj).__name__} ===")
    methods = [m for m in dir(obj) if not m.startswith("_")]
    keys = sorted(
        [m for m in methods if any(k in m.lower() for k in (
            "arch", "core", "grid", "dram", "l1", "size", "mem",
            "device", "buffer", "compute", "physical", "id",
        ))]
    )
    for m in keys:
        try:
            attr = getattr(obj, m)
            if callable(attr):
                try:
                    val = attr()
                except TypeError:
                    val = "<requires args>"
                except Exception as e:
                    val = f"<{type(e).__name__}: {e}>"
            else:
                val = attr
            print(f"  {m:36s} = {val!r:.200}")
        except Exception as e:
            print(f"  {m:36s} = <error {type(e).__name__}: {e}>")


def main():
    print("ttnn introspection on single P150 (device 0)")
    print("=" * 60)
    dev = ttnn.open_device(device_id=0)
    try:
        dump_obj("device", dev)
        # try mem cfg
        print("\n=== DRAM_MEMORY_CONFIG / L1_MEMORY_CONFIG attrs ===")
        for n in ("DRAM_MEMORY_CONFIG", "L1_MEMORY_CONFIG"):
            mc = getattr(ttnn, n, None)
            print(f"  {n}: {mc!r}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
