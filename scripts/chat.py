#!/usr/bin/env python3
"""Claude-Code-style chat TUI for the CB OpenAI server.

Stdlib-only (no `rich` dep). Connects to http://localhost:8000 over an
`ssh -L 8000:...:8000 qb1` tunnel by default. Multi-turn, streams
tokens inline, supports slash commands, OpenAI-style tool calls, and
markdown-flavoured rendering for code blocks.

Quickstart:
  ssh -L 8000:localhost:8000 qb1        # in another terminal first
  python3 scripts/chat.py                # default: greedy, max 1024
  python3 scripts/chat.py --tools        # enable built-in tools (shell, read_file, write_file)
  python3 scripts/chat.py --temp 0.7 --seed 42

In-chat commands (type at the > prompt):
  /new                  reset conversation history
  /sys <text>           set/replace the system prompt
  /temp <float>         set temperature (0 = greedy)
  /top-p <float>        set top_p
  /top-k <int>          set top_k (0 = disabled)
  /seed <int>           set seed
  /max <int>            set max_tokens per turn
  /tools                toggle built-in tool calling
  /continue             resume after a finish=length truncation
  /show                 print current params + history length
  /metrics              fetch and dump /metrics
  /history              dump the conversation transcript
  /save <file>          save transcript to JSON
  /load <file>          load transcript from JSON
  /help                 show this help
  /exit  /quit          leave

Multi-line input: end a line with `\\` to continue. Empty line submits.

Tools: a minimal toolbox of safe shell-style helpers when --tools is set.
Tool calls are detected by parsing the model's output for fenced JSON
blocks tagged with `tool_code` / OpenAI-compatible `tool_calls` deltas.
The result is fed back as a tool message so the model can produce a
final response. Use /tools at runtime to toggle.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
from urllib.parse import urlparse


# ── ANSI helpers ───────────────────────────────────────────────────
TTY = sys.stdout.isatty()
def _c(code, s): return f"\033[{code}m{s}\033[0m" if TTY else s
BOLD = lambda s: _c("1", s)            # noqa: E731
DIM  = lambda s: _c("2", s)            # noqa: E731
ITAL = lambda s: _c("3", s)            # noqa: E731
UND  = lambda s: _c("4", s)            # noqa: E731
RED  = lambda s: _c("31", s)           # noqa: E731
GREEN= lambda s: _c("32", s)           # noqa: E731
YELO = lambda s: _c("33", s)           # noqa: E731
BLUE = lambda s: _c("34", s)           # noqa: E731
MAGE = lambda s: _c("35", s)           # noqa: E731
CYAN = lambda s: _c("36", s)           # noqa: E731
GREY = lambda s: _c("38;5;243", s)     # noqa: E731

# Box-drawing for panels.
H = "─"; V = "│"; TL = "╭"; TR = "╮"; BL = "╰"; BR = "╯"; MID = "├"

def panel(title: str, body: str, color=CYAN, width: int | None = None) -> str:
    """Render a single-line-title panel around body. Body may contain ANSI."""
    if width is None:
        try:
            width = max(40, min(120, os.get_terminal_size().columns - 2))
        except Exception:
            width = 80
    top = color(f"{TL}{H} {title} ").ljust(width + len(color(""))*0, "")
    # Pad with H to width
    plain_title = f"{TL}{H} {title} "
    pad = max(0, width - len(plain_title) - 1)
    top_line = color(f"{TL}{H} {title} " + H*pad + TR)
    bot_line = color(f"{BL}" + H*(width-2) + BR)
    out = [top_line]
    for line in body.splitlines():
        out.append(color(V) + " " + line)
    out.append(bot_line)
    return "\n".join(out)


# ── HTTP helpers ───────────────────────────────────────────────────
def _conn(url):
    u = urlparse(url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=600)


def _stream_chat(url, messages, params, tools=None):
    """POST /v1/chat/completions stream=true; yield (kind, payload)."""
    body = {"messages": messages, "max_tokens": params["max_tokens"], "stream": True}
    if params["temperature"] > 0:
        body["temperature"] = params["temperature"]
        body["top_p"] = params["top_p"]
        body["top_k"] = params["top_k"]
        if params["seed"] is not None:
            body["seed"] = params["seed"]
    if tools:
        body["tools"] = tools
    conn = _conn(url)
    try:
        conn.request("POST", "/v1/chat/completions",
                     json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            yield ("error", f"HTTP {resp.status} {resp.read()[:200].decode('utf-8','replace')}")
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


# ── Markdown-ish rendering of stream chunks ────────────────────────
# Track whether we're inside a fenced code block; flush a styled line.
_FENCE_RE = re.compile(r"^```(\w*)\s*$")

class StreamRenderer:
    """Light streaming markdown renderer — code fences get DIM colour."""
    def __init__(self):
        self.buf = ""
        self.in_code = False

    def feed(self, chunk: str) -> None:
        self.buf += chunk
        while "\n" in self.buf:
            line, _, rest = self.buf.partition("\n")
            self._emit_line(line + "\n")
            self.buf = rest

    def flush(self) -> None:
        if self.buf:
            self._emit_line(self.buf)
            self.buf = ""

    def _emit_line(self, line: str) -> None:
        m = _FENCE_RE.match(line.rstrip("\n"))
        if m:
            self.in_code = not self.in_code
            sys.stdout.write(GREY(line))
        elif self.in_code:
            sys.stdout.write(GREY(line))
        else:
            sys.stdout.write(line)
        sys.stdout.flush()


# ── Multi-line input ──────────────────────────────────────────────
def _read_input(prompt: str) -> str | None:
    """Read possibly-multi-line input. Lines ending with `\\` continue."""
    try:
        first = input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None
    if not first.endswith("\\"):
        return first.strip()
    # Accumulate until a line WITHOUT trailing backslash.
    buf = [first.rstrip("\\")]
    while True:
        try:
            nxt = input(DIM("  … "))
        except (EOFError, KeyboardInterrupt):
            return None
        if not nxt.endswith("\\"):
            buf.append(nxt)
            break
        buf.append(nxt.rstrip("\\"))
    return "\n".join(buf).strip()


# ── Built-in safe toolbox ─────────────────────────────────────────
BUILTIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a SHORT shell command (read-only / safe). 5-second timeout. Returns stdout+stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Command to run, e.g. 'ls -la' or 'date'"}
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file under the user's home directory. Returns up to 4 KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Safely evaluate a Python arithmetic expression. No I/O, no names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "Expression, e.g. '2**16 + 3*7'"}
                },
                "required": ["expr"],
            },
        },
    },
]


def _exec_tool(name: str, args: dict) -> str:
    """Execute a built-in tool. Always returns a string (the model sees it)."""
    try:
        if name == "shell":
            cmd = str(args.get("cmd", ""))
            forbid = ("rm ", "sudo", "shutdown", "reboot", "mkfs", "dd ", ">", ">>", "|", ";", "&&", "&", "$(")
            if any(s in cmd for s in forbid):
                return f"REFUSED: shell command contains forbidden token: {cmd!r}"
            r = subprocess.run(shlex.split(cmd), capture_output=True, timeout=5, text=True)
            return f"exit={r.returncode}\n{r.stdout[:2000]}\n{r.stderr[:1000]}"
        elif name == "read_file":
            p = os.path.expanduser(str(args.get("path", "")))
            if not p.startswith(os.path.expanduser("~")):
                return f"REFUSED: path outside HOME: {p}"
            with open(p, "rb") as f:
                data = f.read(4096)
            return data.decode("utf-8", errors="replace")
        elif name == "calc":
            expr = str(args.get("expr", ""))
            # No names / no calls — only literals + operators.
            if re.search(r"[A-Za-z_]", expr):
                return f"REFUSED: only arithmetic literals allowed: {expr!r}"
            return str(eval(expr, {"__builtins__": {}}, {}))
        return f"unknown tool: {name}"
    except Exception as e:
        return f"tool error: {e!r}"


# Detect tool calls in plain text output (since our cb_api doesn't
# emit proper OpenAI tool_call deltas yet). Both Gemma IT and Qwen3.6
# emit JSON in their assistant turn when given tool definitions:
#   ```tool_code
#   {"name":"calc","arguments":{"expr":"2+2"}}
#   ```
# OR:
#   <tool_call>{"name":...,"arguments":...}</tool_call>
_TOOL_PATTERNS = [
    re.compile(r"```tool_code\s*(\{.*?\})\s*```", re.DOTALL),
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
    re.compile(r"```json\s*(\{[^`]*?\"name\"\s*:[^`]*?\})\s*```", re.DOTALL),
]


def _parse_tool_calls(text: str) -> list[dict]:
    """Return list of {name, arguments} dicts found in the assistant text."""
    out = []
    for pat in _TOOL_PATTERNS:
        for m in pat.finditer(text):
            try:
                obj = json.loads(m.group(1))
                if "name" in obj:
                    out.append(obj)
            except json.JSONDecodeError:
                pass
    return out


# ── Help text ─────────────────────────────────────────────────────
HELP = """\
COMMANDS
  /new                  reset conversation history (keep system prompt)
  /sys <text>           set the system prompt
  /temp <float>         temperature (0 = greedy)
  /top-p /top-k /seed   sampling params
  /max <int>            max_tokens per turn
  /tools                toggle built-in tool calling
  /continue             resume after a finish=length cut
  /show /history        show params / transcript
  /save /load <file>    persist transcript to JSON
  /metrics              fetch server /metrics
  /help                 this
  /exit  /quit          leave

INPUT
  Plain text → user turn.
  Trailing \\ → multi-line continuation; empty line ends input.

TOOLS (when --tools / /tools enabled)
  shell <cmd>           safe read-only shell command (5s timeout)
  read_file <path>      read up to 4 KB from a file in $HOME
  calc <expr>           safe arithmetic eval
"""


# ── Main loop ─────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000",
                    help="base URL of the cb_api server (default: tunnel)")
    ap.add_argument("--sys", default=None, help="initial system prompt")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max", type=int, default=1024, help="max_tokens per turn")
    ap.add_argument("--tools", action="store_true",
                    help="enable built-in tool calling (shell, read_file, calc)")
    args = ap.parse_args()

    params = {"temperature": args.temp, "top_p": args.top_p, "top_k": args.top_k,
              "seed": args.seed, "max_tokens": args.max}
    tools_on = bool(args.tools)
    messages: list[dict] = []
    if args.sys:
        messages.append({"role": "system", "content": args.sys})

    code, body = _health(args.url)
    if code != 200:
        print(RED(f"[chat] cannot reach {args.url}: {body!r}"))
        print(DIM(f"[chat] is the daemon up and your ssh tunnel open?"))
        sys.exit(1)

    # Banner
    print()
    print(CYAN(BOLD(f"  tt-chat → {args.url}")))
    print(DIM(f"  server: {body}"))
    print(DIM(f"  /help for commands · multi-line: end with \\"))
    print(DIM(f"  tools: {'ON' if tools_on else 'OFF'}"))
    print()

    CONTINUE_PROMPT = ("Continue your previous response from exactly where you "
                       "left off. Do not restart, do not repeat any text.")

    while True:
        user = _read_input(BOLD(CYAN("> ")))
        if user is None:
            print(); break
        if not user:
            continue

        # Slash commands.
        if user.startswith("/"):
            cmd, _, arg = user[1:].partition(" "); arg = arg.strip()
            if cmd in ("exit", "quit"):
                break
            elif cmd == "help":
                print(DIM(HELP)); continue
            elif cmd == "continue":
                user = CONTINUE_PROMPT  # fall through
            elif cmd == "new":
                sys_msg = next((m for m in messages if m["role"] == "system"), None)
                messages = [sys_msg] if sys_msg else []
                print(DIM("history cleared")); continue
            elif cmd == "sys":
                messages = [m for m in messages if m["role"] != "system"]
                if arg:
                    messages.insert(0, {"role": "system", "content": arg})
                    print(DIM(f"system prompt set ({len(arg)} chars)"))
                else:
                    print(DIM("system prompt cleared"))
                continue
            elif cmd == "temp":
                params["temperature"] = float(arg); print(DIM(f"temp={params['temperature']}")); continue
            elif cmd == "top-p":
                params["top_p"] = float(arg); print(DIM(f"top_p={params['top_p']}")); continue
            elif cmd == "top-k":
                params["top_k"] = int(arg); print(DIM(f"top_k={params['top_k']}")); continue
            elif cmd == "seed":
                params["seed"] = int(arg) if arg else None; print(DIM(f"seed={params['seed']}")); continue
            elif cmd == "max":
                params["max_tokens"] = int(arg); print(DIM(f"max_tokens={params['max_tokens']}")); continue
            elif cmd == "tools":
                tools_on = not tools_on; print(DIM(f"tools={'ON' if tools_on else 'OFF'}")); continue
            elif cmd == "show":
                print(DIM(f"params={params}  history={len(messages)} msgs  tools={'ON' if tools_on else 'OFF'}"))
                continue
            elif cmd == "history":
                print(DIM("─" * 40))
                for m in messages:
                    role = m["role"].upper()
                    print(YELO(f"[{role}]") + " " + m["content"][:200])
                print(DIM("─" * 40))
                continue
            elif cmd == "save":
                path = arg or "chat.json"
                with open(path, "w") as f:
                    json.dump(messages, f, indent=2)
                print(DIM(f"saved {len(messages)} msgs to {path}")); continue
            elif cmd == "load":
                with open(arg) as f:
                    messages = json.load(f)
                print(DIM(f"loaded {len(messages)} msgs from {arg}")); continue
            elif cmd == "metrics":
                print(DIM(_metrics(args.url))); continue
            else:
                print(RED(f"unknown command /{cmd}")); continue

        # Send turn.
        messages.append({"role": "user", "content": user})
        for _round in range(4):   # tool-call loops capped at 4 per user turn
            print()
            print(GREEN(BOLD("assistant ")) + GREY("─" * 40))
            renderer = StreamRenderer()
            reply = []
            t0 = time.time(); ttft = None; n_chunks = 0; finish = None
            try:
                for kind, payload in _stream_chat(
                        args.url, messages, params,
                        tools=BUILTIN_TOOLS if tools_on else None):
                    if kind == "delta":
                        if ttft is None: ttft = time.time() - t0
                        renderer.feed(payload); reply.append(payload); n_chunks += 1
                    elif kind == "finish":
                        finish = payload
                    elif kind == "error":
                        print(); print(RED(f"error: {payload}")); reply = None; break
            except KeyboardInterrupt:
                print(); print(DIM("interrupted")); reply = None
            renderer.flush()
            elapsed = time.time() - t0
            print()
            if reply:
                full = "".join(reply)
                messages.append({"role": "assistant", "content": full})
                tps = n_chunks / elapsed if elapsed > 0 else 0.0
                stats = f"{n_chunks} chunks · {elapsed:.1f}s · {tps:.1f} chunk/s · TTFT {ttft*1000:.0f}ms · finish={finish}"
                print(GREY(f"  [{stats}]"))
                if finish == "length":
                    print(DIM(f"  hit max_tokens={params['max_tokens']} — type /continue to resume"))

                # Tool-call loop: detect, execute, feed result back.
                if tools_on:
                    calls = _parse_tool_calls(full)
                    if calls:
                        for call in calls:
                            print()
                            print(MAGE(BOLD("tool ")) + MAGE(call.get("name", "?"))
                                  + " " + DIM(json.dumps(call.get("arguments", {}))))
                            result = _exec_tool(call["name"], call.get("arguments", {}))
                            print(DIM(textwrap.indent(result[:800], "  ")))
                            messages.append({
                                "role": "tool",
                                "name": call["name"],
                                "content": result,
                            })
                        # Loop back to model so it can use the tool result.
                        continue
            else:
                messages.pop()  # roll back failed user turn
            break  # no tool calls (or reply failed); next user turn


if __name__ == "__main__":
    main()
