"""PC-P5 smoke test against the running CB API.

Drives the OpenAI-compatible /v1/chat/completions endpoint and verifies that
turn-2 of a conversation hits the prefix cache:
  - turn 1 TTFT measured (cold; prefix cache miss)
  - turn 2 sends the same history + a new short user message
  - turn 2 TTFT measured (expected: hit, MUCH faster)
  - /metrics scraped before+after; expect cb_prefix_cache_hits_total to
    increment by exactly 1, cb_prefix_cache_live_slots to track lifecycle.

Run from project root (will hit qb1's localhost):
  ssh qb1
  cd ~/tt-xla && .venv/bin/python experiments/cb/validate/prefix_cache_smoke.py
or from local with --host qb1 --port 8000.

Exits 0 on success.
"""

from __future__ import annotations
import argparse
import sys
import time
import urllib.request
import urllib.error
import json


def fetch_metrics(host: str, port: int) -> dict[str, float]:
    """Pull /metrics (Prometheus text format) and parse the cb_prefix_cache_* lines."""
    url = f"http://{host}:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            text = r.read().decode("utf-8")
    except urllib.error.URLError as e:
        print(f"  failed to fetch {url}: {e}")
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not line.startswith("cb_prefix_cache") and not line.startswith("cb_engine"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            pass
    return out


def chat(host: str, port: int, messages: list[dict], max_tokens: int = 64) -> tuple[str, float]:
    """Returns (response_text, ttft_seconds). Non-streaming."""
    payload = {
        "model": "Qwen/Qwen3.6-27B",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,  # greedy for reproducibility
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    # Non-streaming, so "TTFT" here is the full response latency. For the
    # signal we want — was prefix prefill skipped — total latency works:
    # turn 2 hit should be drastically faster than turn 1 miss.
    content = resp["choices"][0]["message"]["content"]
    return content, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"[pc-smoke] target = http://{args.host}:{args.port}")
    print("[pc-smoke] fetching baseline metrics...")
    m0 = fetch_metrics(args.host, args.port)
    print(f"  cb_prefix_cache_enabled       = {m0.get('cb_prefix_cache_enabled', '?')}")
    print(f"  cb_prefix_cache_hits_total    = {m0.get('cb_prefix_cache_hits_total', 0)}")
    print(f"  cb_prefix_cache_misses_total  = {m0.get('cb_prefix_cache_misses_total', 0)}")
    print(f"  cb_prefix_cache_live_slots    = {m0.get('cb_prefix_cache_live_slots', 0)}")

    if m0.get("cb_prefix_cache_enabled", 0) != 1.0:
        print("[pc-smoke] FAIL: cb_prefix_cache_enabled != 1. Restart with TT_CB_PREFIX_CACHE=1.")
        return 1

    # Turn 1: cold conversation. ~30-40 prompt tokens (enough to exceed
    # PREFIX_CACHE_MIN_MATCH=16 and get cached).
    system_msg = {"role": "system",
                  "content": "You are a concise, helpful assistant. Reply in 1-2 sentences."}
    user_1 = {"role": "user",
              "content": "What is the capital of France, and what is one famous landmark there?"}
    print("\n[pc-smoke] TURN 1 (cold, miss expected)")
    print(f"  user: {user_1['content']!r}")
    t1_text, t1_latency = chat(args.host, args.port, [system_msg, user_1])
    print(f"  assistant: {t1_text!r}")
    print(f"  latency = {t1_latency:.2f}s")

    print("[pc-smoke] sleeping 2s, then fetching mid-metrics...")
    time.sleep(2.0)
    m_mid = fetch_metrics(args.host, args.port)
    print(f"  cb_prefix_cache_live_slots    = {m_mid.get('cb_prefix_cache_live_slots', 0)}  (expect 1)")
    print(f"  cb_prefix_cache_hits_total    = {m_mid.get('cb_prefix_cache_hits_total', 0)}  (expect 0)")
    print(f"  cb_prefix_cache_misses_total  = {m_mid.get('cb_prefix_cache_misses_total', 0)}")

    # Turn 2: same history + new user message. Expect cache hit.
    assistant_1 = {"role": "assistant", "content": t1_text}
    user_2 = {"role": "user", "content": "And what about Germany?"}
    print("\n[pc-smoke] TURN 2 (warm, hit expected)")
    print(f"  user: {user_2['content']!r}")
    t2_text, t2_latency = chat(args.host, args.port, [system_msg, user_1, assistant_1, user_2])
    print(f"  assistant: {t2_text!r}")
    print(f"  latency = {t2_latency:.2f}s")

    print("\n[pc-smoke] fetching final metrics...")
    m1 = fetch_metrics(args.host, args.port)
    print(f"  cb_prefix_cache_hits_total    = {m1.get('cb_prefix_cache_hits_total', 0)}")
    print(f"  cb_prefix_cache_misses_total  = {m1.get('cb_prefix_cache_misses_total', 0)}")
    print(f"  cb_prefix_cache_live_slots    = {m1.get('cb_prefix_cache_live_slots', 0)}")
    print(f"  cb_prefix_cache_evictions_total = {m1.get('cb_prefix_cache_evictions_total', 0)}")

    hits_delta = m1.get("cb_prefix_cache_hits_total", 0) - m0.get("cb_prefix_cache_hits_total", 0)
    print("\n[pc-smoke] === SUMMARY ===")
    print(f"  turn 1 latency : {t1_latency:.2f}s (cold)")
    print(f"  turn 2 latency : {t2_latency:.2f}s ({'WARM' if hits_delta >= 1 else 'cold?'})")
    print(f"  hits delta     : {hits_delta:.0f}  (expect ≥ 1)")
    print(f"  speedup        : {t1_latency / max(t2_latency, 1e-3):.2f}x  (note: includes decode of turn-2 reply)")

    ok = True
    if hits_delta < 1:
        print("  ✗ FAIL: no prefix cache hit detected")
        ok = False
    else:
        print(f"  ✓ prefix cache hit observed (count={hits_delta:.0f})")
    if t2_latency >= t1_latency:
        print("  ⚠ turn 2 not faster than turn 1 — investigate")
    else:
        print(f"  ✓ turn 2 faster ({t1_latency - t2_latency:.2f}s shaved)")
    print("  qualitative   : turn 2 response should mention Berlin / Germany capital")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
