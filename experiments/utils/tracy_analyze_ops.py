#!/usr/bin/env python3
"""
Analyze Tracy per-op output from tracy_traced_decode_probe.py.

Joins two CSVs from the tt-metal Tracy harness:
  - tracy_ops_data.csv: JSON-style records giving op TYPE and global_call_count
  - tracy_ops_times.csv: zone-level host timings with zone_text = "id:N"
                           (N == global_call_count) and exec_time_ns

Produces a per-op-category breakdown of total host-dispatch time observed
across the EAGER warmup + capture forwards (trace replays show up as a
single bundled zone and are NOT decomposed here).

Per-step normalisation: tracy_traced_decode_probe.py runs N_WARMUP=2 eager
forwards then 1 eager forward during capture → 3 eager forwards. We
divide the totals by 3 to get per-step host-dispatch numbers, then
scale by (target_layers / chain_layers) to extrapolate to full model.

Run:
    python3 experiments/utils/tracy_analyze_ops.py \\
        --log-dir research/probe_logs/tracy_qb1_traced/.logs \\
        --eager-forwards 3 \\
        --chain-layers 8 \\
        --target-layers 128
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict


def parse_ops_data(path):
    """Yield (op_name, global_call_count) pairs from tracy_ops_data.csv.

    The file uses a custom semicolon-separated header (MessageName;total_ns)
    followed by a stream of backtick-prefixed records:
        `TT_DNN_DEVICE_OP: "OpName", <hash>, <device>, <bool>, <global_call_count> ->
        { ...JSON-ish body... }
    Records continue until the next backtick-line. We only need the header
    line to extract op_name and global_call_count.
    """
    rx = re.compile(
        r"^`TT_DNN_DEVICE_OP:\s+\"([^\"]+)\",\s+\d+,\s+\d+,\s+(?:true|false),\s+(\d+)"
    )
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = rx.match(line)
            if m:
                op_name, gcc = m.group(1), int(m.group(2))
                yield op_name, gcc


def parse_ops_times(path):
    """Yield (global_call_count, host_exec_ns) for TT_DNN_DEVICE_OP zones."""
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_name = header.index("name")
        col_zone_text = header.index("zone_text")
        col_exec_ns = header.index("exec_time_ns")
        for row in reader:
            if len(row) <= col_exec_ns:
                continue
            if row[col_name] != "TT_DNN_DEVICE_OP":
                continue
            zt = row[col_zone_text]
            if not zt.startswith("id:"):
                continue
            try:
                gcc = int(zt[3:])
                ns = int(row[col_exec_ns])
            except ValueError:
                continue
            yield gcc, ns


def categorize(op_name):
    """Map raw device op name to a coarse category for the breakdown."""
    n = op_name.lower()
    if "matmul" in n:
        return "matmul (ttnn.linear)"
    if "layernorm" in n:
        return "rms_norm / layernorm"
    if "binary" in n:
        return "binary (add/mul/sub)"
    if "unary" in n:
        return "unary (silu/sigmoid/exp/log/neg)"
    if "reduce" in n:
        return "reduce (sum)"
    if "reshape" in n:
        return "reshape (view)"
    if "slice" in n:
        return "slice"
    if "concat" in n:
        return "concat"
    if "repeat" in n:
        return "repeat (GQA broadcast)"
    if "scatter" in n:
        return "scatter (KV cache write)"
    if "sdpa" in n:
        return "SDPA decode"
    if "permute" in n:
        return "permute"
    if "copy" in n:
        return "copy (state thread-back)"
    if "tilize" in n and "untilize" in n:
        # shouldn't happen but safety
        return "(un)tilize"
    if "untilizewithunpadding" in n:
        return "untilize_with_unpadding"
    if "tilizewithvalpadding" in n:
        return "tilize_with_padding"
    if "untilize" in n:
        return "untilize"
    if "tilize" in n:
        return "tilize"
    if "typecast" in n:
        return "typecast"
    if "fillpad" in n:
        return "fill_pad"
    return "other: " + op_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--eager-forwards", type=int, default=3,
                    help="Number of eager forward invocations (warmup + capture)")
    ap.add_argument("--chain-layers", type=int, default=8,
                    help="Number of layers in the probe chain")
    ap.add_argument("--target-layers", type=int, default=64,
                    help="Number of layers in production model "
                         "(Qwen3.6-27B: 48 DN + 16 attn = 64 transformer blocks)")
    args = ap.parse_args()

    p_data = os.path.join(args.log_dir, "tracy_ops_data.csv")
    p_times = os.path.join(args.log_dir, "tracy_ops_times.csv")
    if not os.path.exists(p_data):
        print(f"ERROR: missing {p_data}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(p_times):
        print(f"ERROR: missing {p_times}", file=sys.stderr); sys.exit(1)

    # Build gcc -> op_name map
    gcc_to_op = {}
    for op_name, gcc in parse_ops_data(p_data):
        gcc_to_op[gcc] = op_name

    print(f"# parsed {len(gcc_to_op)} op records from tracy_ops_data.csv")

    # Sum host-dispatch ns per category
    cat_total_ns = defaultdict(int)
    cat_count = defaultdict(int)
    matched = 0
    unmatched = 0
    grand_total_ns = 0
    for gcc, ns in parse_ops_times(p_times):
        op = gcc_to_op.get(gcc)
        if op is None:
            unmatched += 1
            continue
        cat = categorize(op)
        cat_total_ns[cat] += ns
        cat_count[cat] += 1
        grand_total_ns += ns
        matched += 1
    print(f"# matched {matched} ops_times rows; {unmatched} unmatched")

    # Per-step values: divide by eager_forwards
    fwd = args.eager_forwards
    if fwd <= 0:
        fwd = 1

    print("\n" + "=" * 92)
    print(f"PER-OP HOST-DISPATCH TIME BREAKDOWN (chain = {args.chain_layers} layers, "
          f"per-step = total / {fwd} eager forwards)")
    print("=" * 92)
    print(f"{'Category':<32} {'# calls/step':>12} {'ms / step':>10} {'% step':>8} "
          f"{'ms × ' + str(args.target_layers) + '/' + str(args.chain_layers):>12}")
    print("-" * 92)

    rows = []
    for cat, ns in cat_total_ns.items():
        n_total = cat_count[cat]
        n_per_step = n_total / fwd
        ms_per_step = ns / fwd / 1e6
        rows.append((cat, n_per_step, ms_per_step))
    rows.sort(key=lambda r: -r[2])

    total_step_ms = grand_total_ns / fwd / 1e6
    target_scale = args.target_layers / args.chain_layers
    for cat, n_per_step, ms_per_step in rows:
        pct = ms_per_step / total_step_ms * 100 if total_step_ms else 0.0
        target_ms = ms_per_step * target_scale
        print(f"{cat:<32} {n_per_step:>12.1f} {ms_per_step:>10.3f} {pct:>7.1f}% "
              f"{target_ms:>11.1f}")
    print("-" * 92)
    print(f"{'TOTAL':<32} {sum(n for _, n, _ in rows):>12.1f} {total_step_ms:>10.3f} {100.0:>7.1f}% "
          f"{total_step_ms*target_scale:>11.1f}")
    print()
    print(f"Per-step host-dispatch sum (chain={args.chain_layers}): {total_step_ms:.2f} ms")
    print(f"Extrapolated to {args.target_layers}-block model:        {total_step_ms*target_scale:.2f} ms")
    print(f"  (linear scaling = same per-layer host overhead × target/chain)")


if __name__ == "__main__":
    main()
