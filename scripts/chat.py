#!/usr/bin/env python3
"""Tiny stdlib chat TUI for the CB OpenAI server. No deps.

Runs locally; talks to http://localhost:8000 over your `ssh -L 8000:...:8000 qb1`
tunnel by default. Multi-turn (keeps the message history), streams tokens
inline, supports slash commands.

  python3 scripts/chat.py
  python3 scripts/chat.py --url http://qb1.local:8000      # if you can reach qb1 directly
  python3 scripts/chat.py --temp 0.7 --seed 42             # start with sampling

In-chat commands (type at the > prompt):
  /new                  reset conversation history
  /sys <text>           set/replace the system prompt
  /temp <float>         set temperature (0 = greedy)
  /top-p <float>        set top_p (sampling only)
  /top-k <int>          set top_k (sampling only; 0 = disabled)
  /seed <int>           set seed (sampling only; '' to clear)
  /max <int>            set max_tokens per turn
  /continue             resume after a finish=length truncation
  /show                 print current params + history length
  /metrics              fetch and dump /metrics
  /help                 show this help
  /exit  /quit          leave
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from urllib.parse import urlparse

# Tiny ANSI helpers (no `rich` dep). Disabled if stdout isn't a tty.
def _ansi(code, s):
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s
BOLD = lambda s: _ansi("1", s)        # noqa: E731
DIM  = lambda s: _ansi("2", s)        # noqa: E731
CYAN = lambda s: _ansi("36", s)       # noqa: E731
GREEN = lambda s: _ansi("32", s)      # noqa: E731
RED  = lambda s: _ansi("31", s)       # noqa: E731


def _conn(url):
    u = urlparse(url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=600)


def _stream_chat(url, messages, params):
    """POST /v1/chat/completions stream=true; yield (kind, payload)."""
    body = {"messages": messages, "max_tokens": params["max_tokens"], "stream": True}
    if params["temperature"] > 0:
        body["temperature"] = params["temperature"]
        body["top_p"] = params["top_p"]
        body["top_k"] = params["top_k"]
        if params["seed"] is not None:
            body["seed"] = params["seed"]
    conn = _conn(url)
    try:
        conn.request("POST", "/v1/chat/completions",
                     json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            yield ("error", f"HTTP {resp.status} {resp.read()[:200].decode('utf-8', 'replace')}")
            return
        while True:
            line = resp.readline()
            if not line:
                return
            line = line.rstrip(b"\r\n")
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:].decode("utf-8")
            if payload == "[DONE]":
                yield ("done", None)
                return
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = ev.get("choices", [{}])[0]
            if ch.get("finish_reason"):
                yield ("finish", ch["finish_reason"])
                continue
            d = ch.get("delta", {}).get("content")
            if d:
                yield ("delta", d)
    finally:
        conn.close()


def _health(url):
    try:
        conn = _conn(url); conn.request("GET", "/health")
        r = conn.getresponse(); raw = r.read().decode("utf-8"); conn.close()
        return r.status, json.loads(raw) if raw else {}
    except Exception as e:
        return None, str(e)


def _metrics(url):
    conn = _conn(url); conn.request("GET", "/metrics")
    r = conn.getresponse(); raw = r.read().decode("utf-8"); conn.close()
    return raw if r.status == 200 else f"<HTTP {r.status}>"


HELP = """\
commands:  /new  /sys <text>  /temp <f>  /top-p <f>  /top-k <i>  /seed <i>
           /max <i>  /show  /metrics  /help  /exit"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000",
                    help="base URL of the cb_api server (default: tunnel)")
    ap.add_argument("--sys", default=None, help="initial system prompt")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="temperature (0 = greedy; >0 = sample)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max", type=int, default=1024, help="max_tokens per turn")
    args = ap.parse_args()

    params = {"temperature": args.temp, "top_p": args.top_p, "top_k": args.top_k,
              "seed": args.seed, "max_tokens": args.max}
    messages: list[dict] = []
    if args.sys:
        messages.append({"role": "system", "content": args.sys})

    # Preflight /health.
    code, body = _health(args.url)
    if code != 200:
        print(RED(f"[chat] cannot reach {args.url}: {body!r}"))
        print(DIM(f"[chat] is the daemon up and your `ssh -L 8000:...:8000 qb1` tunnel open?"))
        sys.exit(1)
    print(CYAN(f"[chat] {args.url} → {body}"))
    print(DIM(HELP))

    CONTINUE_PROMPT = ("Continue your previous response from exactly where you "
                       "left off. Do not restart, do not repeat any text, do not "
                       "re-open the <think> block — just keep generating.")

    while True:
        try:
            user = input(BOLD("\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue

        # Slash commands.
        if user.startswith("/"):
            cmd, _, arg = user[1:].partition(" ")
            arg = arg.strip()
            if cmd in ("exit", "quit"):
                break
            elif cmd == "help":
                print(DIM(HELP))
            elif cmd == "continue":
                # Pretend the user typed the continuation prompt; fall through
                # to the normal turn-send path.
                user = CONTINUE_PROMPT
            elif cmd == "new":
                sys_msg = next((m for m in messages if m["role"] == "system"), None)
                messages = [sys_msg] if sys_msg else []
                print(DIM("[chat] history cleared"))
            elif cmd == "sys":
                messages = [m for m in messages if m["role"] != "system"]
                if arg:
                    messages.insert(0, {"role": "system", "content": arg})
                    print(DIM(f"[chat] system prompt set ({len(arg)} chars)"))
                else:
                    print(DIM("[chat] system prompt cleared"))
            elif cmd == "temp":
                params["temperature"] = float(arg); print(DIM(f"[chat] temperature = {params['temperature']}"))
            elif cmd == "top-p":
                params["top_p"] = float(arg); print(DIM(f"[chat] top_p = {params['top_p']}"))
            elif cmd == "top-k":
                params["top_k"] = int(arg); print(DIM(f"[chat] top_k = {params['top_k']}"))
            elif cmd == "seed":
                params["seed"] = int(arg) if arg else None; print(DIM(f"[chat] seed = {params['seed']}"))
            elif cmd == "max":
                params["max_tokens"] = int(arg); print(DIM(f"[chat] max_tokens = {params['max_tokens']}"))
            elif cmd == "show":
                print(DIM(f"[chat] params={params}  history={len(messages)} msgs"))
            elif cmd == "metrics":
                print(DIM(_metrics(args.url)))
            else:
                print(RED(f"[chat] unknown command /{cmd} — /help"))
            if cmd != "continue":
                continue   # /continue falls through to the turn-send below

        # Send turn.
        messages.append({"role": "user", "content": user})
        print(CYAN("\nassistant: "), end="", flush=True)
        reply = []
        t0 = time.time()
        ttft = None
        n_chunks = 0
        finish = None
        try:
            for kind, payload in _stream_chat(args.url, messages, params):
                if kind == "delta":
                    if ttft is None: ttft = time.time() - t0
                    print(payload, end="", flush=True)
                    reply.append(payload)
                    n_chunks += 1
                elif kind == "finish":
                    finish = payload
                elif kind == "error":
                    print(RED(f"\n[chat] error: {payload}")); reply = None; break
        except KeyboardInterrupt:
            print(DIM("\n[chat] interrupted — server-side request is cancelled on disconnect"))
            reply = None
        elapsed = time.time() - t0

        if reply:
            messages.append({"role": "assistant", "content": "".join(reply)})
            tps = n_chunks / elapsed if elapsed > 0 else 0.0
            print(GREEN(f"\n[{n_chunks} chunks · {elapsed:.1f}s · {tps:.1f} chunk/s · "
                        f"TTFT {ttft*1000:.0f}ms · finish={finish}]"))
            if finish == "length":
                print(DIM(f"[chat] hit max_tokens={params['max_tokens']} — type /continue to "
                          f"resume, or /max <n> to raise the cap"))
        else:
            # Roll back the failed user turn so /new isn't required.
            messages.pop()


if __name__ == "__main__":
    main()
