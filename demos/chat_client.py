#!/usr/bin/env python3
"""
Chat client for the Qwen2.5-0.5B Blackhole chat server.

Usage:
    # Single prompt:
    python3 chat_client.py "What is the capital of France?"

    # Interactive mode:
    python3 chat_client.py --interactive

    # Custom settings:
    python3 chat_client.py --host tenstorrent --port 8080 --temp 0.9 --max-tokens 150 "Tell me a joke"
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def send_request(host, port, prompt, max_tokens=100, temperature=0.7, top_k=50, seed=None):
    """Send a generate request and return the response."""
    url = f"http://{host}:{port}/generate"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
    }
    if seed is not None:
        payload["seed"] = seed

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            body["_client_roundtrip_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return body
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e}"}
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}"}


def health_check(host, port):
    """Check if server is ready."""
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


def print_response(resp):
    """Pretty-print a generate response."""
    if "error" in resp:
        print(f"\nError: {resp['error']}")
        return

    print(f"\n--- Response ---")
    print(resp["text"])
    print(f"\n--- Timing ---")
    t = resp["timing"]
    print(f"  Prompt tokens:   {t['prompt_tokens']}")
    print(f"  Generated:       {t['generated_tokens']} tokens")
    print(f"  Prefill:         {t['prefill_ms']:.0f}ms")
    print(f"  Decode:          {t['decode_total_ms']:.0f}ms ({t['avg_decode_ms_per_token']:.1f}ms/tok)")
    print(f"  Throughput:      {t['tokens_per_sec']:.1f} tok/sec")
    if "_client_roundtrip_ms" in resp:
        print(f"  Client RTT:      {resp['_client_roundtrip_ms']:.0f}ms")


def interactive_mode(host, port, max_tokens, temperature, top_k):
    """Run an interactive chat loop."""
    print(f"Connected to Qwen2.5-0.5B on Blackhole P150 ({host}:{port})")
    print(f"Settings: max_tokens={max_tokens}, temperature={temperature}, top_k={top_k}")
    print(f"Type 'quit' or Ctrl+C to exit. Type '/stats' for last timing.\n")

    last_timing = None
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "/quit"):
            print("Goodbye!")
            break
        if prompt == "/stats":
            if last_timing:
                print(f"  Last: {last_timing['generated_tokens']} tokens, "
                      f"{last_timing['tokens_per_sec']:.1f} tok/sec, "
                      f"{last_timing['avg_decode_ms_per_token']:.1f}ms/tok")
            else:
                print("  No previous request.")
            continue

        resp = send_request(host, port, prompt, max_tokens=max_tokens,
                           temperature=temperature, top_k=top_k)

        if "error" in resp:
            print(f"Error: {resp['error']}")
            continue

        print(f"\nQwen: {resp['text']}")
        last_timing = resp["timing"]
        print(f"  [{last_timing['generated_tokens']} tokens, "
              f"{last_timing['tokens_per_sec']:.1f} tok/sec]\n")


def main():
    parser = argparse.ArgumentParser(description="Chat client for Blackhole Qwen server")
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt to send (omit for interactive mode)")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k for sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--health", action="store_true", help="Just check server health")
    args = parser.parse_args()

    if args.health:
        resp = health_check(args.host, args.port)
        print(json.dumps(resp, indent=2))
        return

    if args.interactive or args.prompt is None:
        # Check health first
        h = health_check(args.host, args.port)
        if "error" in h:
            print(f"Cannot connect to server at {args.host}:{args.port}: {h['error']}")
            sys.exit(1)
        interactive_mode(args.host, args.port, args.max_tokens, args.temp, args.top_k)
    else:
        resp = send_request(args.host, args.port, args.prompt,
                           max_tokens=args.max_tokens, temperature=args.temp,
                           top_k=args.top_k, seed=args.seed)
        print_response(resp)


if __name__ == "__main__":
    main()
