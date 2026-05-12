#!/usr/bin/env python3
"""
Diff two perf_baseline.py JSON snapshots and print a per-region summary.

Usage:
    .venv/bin/python experiments/utils/perf_diff.py <baseline.json> <new.json>

Prints:
    - per-region median and pct delta
    - decode-step derived (= prefill+1decode median − prefill median), the key tok/sec metric
    - compounding estimate: 48*deltanet + 16*gated_attn + 64*mlp + lm_head
"""
import sys, json, statistics


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def fmt_delta(old, new):
    if old == 0 or not (old == old):  # NaN guard
        return ""
    pct = (new - old) / old * 100.0
    sign = "+" if pct >= 0 else ""
    return f"  ({sign}{pct:.1f}%)"


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <baseline.json> <new.json>")
        sys.exit(1)

    a, b = load(sys.argv[1]), load(sys.argv[2])
    a_t, b_t = a.get("raw_times_ms", {}), b.get("raw_times_ms", {})

    regions = sorted(set(a_t.keys()) | set(b_t.keys()))
    print(f"phase: {a.get('phase')} → {b.get('phase')}")
    print(f"model: {a.get('model')}")
    print(f"{'region':<30} {'baseline':>14} {'new':>14}")
    print("-" * 60)
    for r in regions:
        old, new = med(a_t.get(r, [])), med(b_t.get(r, []))
        delta = fmt_delta(old, new)
        print(f"{r:<30} {old:>10.2f} ms {new:>10.2f} ms{delta}")

    # Decode-step derived
    p_old = med(a_t.get("prefill_5_tokens", []))
    pd_old = med(a_t.get("prefill_plus_one_decode", []))
    p_new = med(b_t.get("prefill_5_tokens", []))
    pd_new = med(b_t.get("prefill_plus_one_decode", []))
    if all(x == x for x in [p_old, pd_old, p_new, pd_new]):
        d_old, d_new = pd_old - p_old, pd_new - p_new
        delta = fmt_delta(d_old, d_new)
        print("-" * 60)
        print(f"{'decode_step_derived':<30} {d_old:>10.2f} ms {d_new:>10.2f} ms{delta}")
        print(f"{'tok/s':<30} {1000/d_old:>13.2f} {1000/d_new:>13.2f}")

    # Compounding estimate
    def compound(t):
        return (48 * med(t.get("single_deltanet_step", []))
                + 16 * med(t.get("single_gated_attn_step", []))
                + 64 * med(t.get("single_mlp_step", []))
                + 1 * med(t.get("lm_head", [])))
    c_old, c_new = compound(a_t), compound(b_t)
    print(f"{'compounded_estimate':<30} {c_old:>10.2f} ms {c_new:>10.2f} ms{fmt_delta(c_old, c_new)}")
    print(f"  (= 48×deltanet + 16×gated_attn + 64×mlp + lm_head)")


if __name__ == "__main__":
    main()
