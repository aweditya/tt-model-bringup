"""Minimal probe — does the harness even reach this file?"""
from __future__ import annotations
import sys
print("[v1_probe] module imported successfully", flush=True)


def main(state=None) -> int:
    print("[v1_probe] main() reached; state attrs:", flush=True)
    print(f"  cb_B = {getattr(state, 'cb_B', '<unset>')}", flush=True)
    import server_35b_cb as cb
    print(f"  hasattr _batched_prelude: {hasattr(cb, '_batched_prelude')}", flush=True)
    print(f"  hasattr update_input_buffers_batched: {hasattr(cb, 'update_input_buffers_batched')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
