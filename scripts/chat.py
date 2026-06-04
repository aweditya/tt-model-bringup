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
import shutil
import subprocess
import sys
import textwrap
import time
from urllib.parse import urlparse


# ── Terminal width helper ─────────────────────────────────────────
def _cols(default: int = 80) -> int:
    """Best-effort terminal width. Honours $COLUMNS, falls back to shutil."""
    try:
        c = int(os.environ.get("COLUMNS", "0"))
        if c > 0:
            return c
    except ValueError:
        pass
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default


def _reset_terminal() -> None:
    """Restore cursor + colour + clear current line. Safe to call repeatedly."""
    if not TTY:
        return
    try:
        sys.stdout.write("\x1b[?25h\x1b[0m\x1b[2K\r")
        # Disable bracketed paste mode on exit (was enabled in raw paste path).
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
    except Exception:
        pass


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
    """POST /v1/chat/completions stream=true; yield (kind, payload).

    Network/server failures yield ("error", msg) instead of raising so the
    caller's prompt loop survives a flaky server.
    """
    body = {"messages": messages, "max_tokens": params["max_tokens"], "stream": True}
    if params["temperature"] > 0:
        body["temperature"] = params["temperature"]
        body["top_p"] = params["top_p"]
        body["top_k"] = params["top_k"]
        if params["seed"] is not None:
            body["seed"] = params["seed"]
    if tools:
        body["tools"] = tools
    try:
        conn = _conn(url)
    except OSError as e:
        yield ("error", f"connect failed: {e}")
        return
    try:
        try:
            conn.request("POST", "/v1/chat/completions",
                         json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            yield ("error", f"request failed: {e}")
            return
        if resp.status >= 500:
            yield ("error", f"server HTTP {resp.status}: "
                            f"{resp.read()[:200].decode('utf-8','replace')}")
            return
        if resp.status != 200:
            yield ("error", f"HTTP {resp.status} "
                            f"{resp.read()[:200].decode('utf-8','replace')}")
            return
        while True:
            try:
                line = resp.readline()
            except (OSError, http.client.HTTPException) as e:
                yield ("error", f"stream dropped: {e}")
                return
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
        try:
            conn.close()
        except Exception:
            pass


def _health(url):
    try:
        conn = _conn(url); conn.request("GET", "/health")
        r = conn.getresponse(); raw = r.read().decode("utf-8"); conn.close()
        return r.status, json.loads(raw) if raw else {}
    except Exception as e:
        return None, str(e)


def _metrics(url):
    try:
        conn = _conn(url); conn.request("GET", "/metrics")
        r = conn.getresponse(); raw = r.read().decode("utf-8"); conn.close()
    except Exception as e:
        return f"<error fetching /metrics: {e}>"
    return raw if r.status == 200 else f"<HTTP {r.status}>"


# ── Markdown-ish rendering of stream chunks ────────────────────────
# Track whether we're inside a fenced code block; flush a styled line.
_FENCE_RE = re.compile(r"^```(\w*)\s*$")

class StreamRenderer:
    """Light streaming markdown renderer — code fences get DIM colour.

    Prose lines word-wrap to $COLUMNS to avoid hard mid-word splits in
    narrow terminals. Code-fence lines are emitted verbatim (mangling
    code with soft wraps is worse than letting the terminal handle it).
    """
    def __init__(self):
        self.buf = ""
        self.in_code = False
        # Per-line word-wrap state: characters emitted since last "\n".
        self._col = 0

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

    def _wrap_prose(self, line: str) -> str:
        """Word-wrap a single line (with trailing \\n) to $COLUMNS."""
        width = max(20, _cols(80) - 2)
        had_nl = line.endswith("\n")
        text = line[:-1] if had_nl else line
        if not text.strip():
            return line
        # textwrap with break_long_words=False avoids mid-word splits.
        wrapped = textwrap.wrap(
            text, width=width,
            break_long_words=False, break_on_hyphens=False,
            drop_whitespace=False, replace_whitespace=False,
        )
        if not wrapped:
            return line
        out = "\n".join(wrapped)
        if had_nl:
            out += "\n"
        return out

    def _emit_line(self, line: str) -> None:
        m = _FENCE_RE.match(line.rstrip("\n"))
        if m:
            self.in_code = not self.in_code
            sys.stdout.write(GREY(line))
        elif self.in_code:
            sys.stdout.write(GREY(line))
        else:
            sys.stdout.write(self._wrap_prose(line))
        sys.stdout.flush()


# ── Multi-line input + bracketed paste ────────────────────────────
# We use bracketed paste mode (\x1b[?2004h) so a pasted block arrives
# wrapped as \x1b[200~ ... \x1b[201~ and contained newlines do NOT
# submit. For terminals that don't honour bracketed paste, a 50 ms
# inter-byte timeout collapses bursts into a single paste-equivalent.
BRACKETED_PASTE_ON  = "\x1b[?2004h"
BRACKETED_PASTE_OFF = "\x1b[?2004l"
PASTE_START = "\x1b[200~"
PASTE_END   = "\x1b[201~"
PASTE_BURST_MS = 50

try:
    import termios  # type: ignore
    import tty      # type: ignore
    import select   # type: ignore
    _HAVE_TERMIOS = True
except Exception:
    _HAVE_TERMIOS = False


def _read_input_raw(prompt: str) -> str | None:
    """Raw-mode prompt with bracketed-paste + burst-detection support.

    Returns the assembled line on Enter (outside a paste), None on Ctrl-D
    / Ctrl-C. Inside a bracketed paste (or a fast burst), newlines are
    kept as literal '\\n' in the buffer instead of submitting.

    Falls back to plain input() if stdin is not a TTY or termios isn't
    available (e.g. piped stdin, Windows).
    """
    if not (_HAVE_TERMIOS and sys.stdin.isatty() and TTY):
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return None

    sys.stdout.write(prompt); sys.stdout.flush()
    sys.stdout.write(BRACKETED_PASTE_ON); sys.stdout.flush()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf: list[str] = []
    in_paste = False
    last_byte_t = 0.0
    burst_mode = False  # heuristic for terminals w/o bracketed paste
    pending = ""        # rolling tail to spot \x1b[200~ / \x1b[201~

    def _redraw_tail(n_keep: int = 80) -> None:
        # Cheap redraw: erase line, reprint prompt + last n_keep chars of buf.
        s = "".join(buf)
        tail = s[-n_keep:] if len(s) > n_keep else s
        # Replace literal newlines with a visible glyph so the prompt
        # row doesn't actually wrap.
        tail_disp = tail.replace("\n", DIM("↵"))
        sys.stdout.write("\r\x1b[2K" + prompt + tail_disp)
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        while True:
            r, _, _ = select.select([fd], [], [], 0.1)
            if not r:
                # Idle: if we were in a burst, end it.
                if burst_mode and (time.time() - last_byte_t) * 1000 > PASTE_BURST_MS:
                    burst_mode = False
                continue
            ch = os.read(fd, 1).decode("utf-8", errors="replace")
            if not ch:
                return None  # EOF
            now = time.time()
            # Detect rapid arrival → likely paste.
            if last_byte_t and (now - last_byte_t) * 1000 < PASTE_BURST_MS:
                burst_mode = True
            last_byte_t = now

            # ESC sequence buffering for bracketed paste markers.
            if ch == "\x1b" or pending:
                pending += ch
                if pending.endswith(PASTE_START):
                    in_paste = True
                    pending = ""
                    continue
                if pending.endswith(PASTE_END):
                    in_paste = False
                    pending = ""
                    continue
                # Bail out of escape buffering once it's clearly not a
                # paste marker (any other escape we silently drop — keeps
                # arrow keys etc. from polluting the buffer).
                if len(pending) > 8 or (len(pending) >= 2 and pending[1] not in "[" ):
                    pending = ""
                continue

            # Ctrl-C cancels the line; Ctrl-D on empty buf → EOF.
            if ch == "\x03":  # Ctrl-C
                sys.stdout.write("\n"); sys.stdout.flush()
                return ""
            if ch == "\x04":  # Ctrl-D
                if not buf:
                    return None
                continue
            # Enter: submit only when NOT in a paste / burst.
            if ch in ("\r", "\n"):
                if in_paste or burst_mode:
                    buf.append("\n")
                    sys.stdout.write(DIM("↵"))
                    sys.stdout.flush()
                    continue
                sys.stdout.write("\n"); sys.stdout.flush()
                return "".join(buf)
            # Backspace / DEL.
            if ch in ("\x7f", "\x08"):
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b"); sys.stdout.flush()
                continue
            buf.append(ch)
            sys.stdout.write(ch); sys.stdout.flush()
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        try:
            sys.stdout.write(BRACKETED_PASTE_OFF); sys.stdout.flush()
        except Exception:
            pass


def _read_input(prompt: str) -> str | None:
    """Read possibly-multi-line input.

    Path A (TTY + termios): raw-mode reader with bracketed-paste +
    burst detection — newlines inside a paste/burst become literal '\\n'.
    Path B (fallback): legacy line-mode where trailing '\\' continues
    onto the next line.
    """
    if _HAVE_TERMIOS and sys.stdin.isatty() and TTY:
        s = _read_input_raw(prompt)
        return None if s is None else s.strip()
    try:
        first = input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None
    if not first.endswith("\\"):
        return first.strip()
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


def _read_paste_block(sentinels=(":end:",)) -> str:
    """Read a multi-line block in cooked mode until a sentinel line.

    Triple-blank-line is also a valid terminator. Returns the joined
    text WITHOUT the sentinel line.
    """
    print(DIM("  paste mode — end with ':end:' on its own line "
              "(or three blank lines)"))
    lines: list[str] = []
    blank_run = 0
    while True:
        try:
            ln = input(DIM("  ¶ "))
        except (EOFError, KeyboardInterrupt):
            break
        if ln.strip() in sentinels:
            break
        if not ln.strip():
            blank_run += 1
            if blank_run >= 3:
                lines = lines[:-2]  # drop the prior two blanks
                break
        else:
            blank_run = 0
        lines.append(ln)
    return "\n".join(lines).rstrip()


# ── Clipboard ────────────────────────────────────────────────────
def _clipboard_copy(text: str) -> tuple[bool, str]:
    """Copy text to the system clipboard. Returns (ok, tool_used)."""
    candidates = [
        ("pbcopy", ["pbcopy"]),
        ("wl-copy", ["wl-copy"]),
        ("xclip", ["xclip", "-selection", "clipboard"]),
        ("xsel", ["xsel", "--clipboard", "--input"]),
    ]
    for name, argv in candidates:
        if not shutil.which(argv[0]):
            continue
        try:
            p = subprocess.run(argv, input=text.encode("utf-8"),
                               timeout=2, check=False)
            if p.returncode == 0:
                return True, name
        except Exception:
            continue
    return False, ""


_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


def _last_code_block(text: str) -> str | None:
    matches = _CODE_BLOCK_RE.findall(text)
    return matches[-1].rstrip() if matches else None


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
  /metrics              fetch server /metrics (live dashboard)
  /paste [header]       multi-line paste mode (end with ':end:' or 3 blanks)
  /yank [code]          copy last reply (or its last code block) to clipboard
  /screenshot           save a screenshot of the terminal (presentation/)
  /help                 this
  /exit  /quit          leave

INPUT
  Plain text → user turn. Paste a block: bracketed paste is auto-detected;
  fall back to /paste for terminals that don't pass paste reliably.
  Trailing \\ → multi-line continuation (line-mode fallback only).

TOOLS (when --tools / /tools enabled)
  shell <cmd>           allow-listed read-only shell command (5s timeout)
  read_file path,start,n read a numbered line range from a project file
  write_file path,text   create/append a file inside the project (CWD-jail)
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
        print(DIM("[chat] is the daemon up and your ssh tunnel open?"))
        sys.exit(1)

    # Banner
    print()
    print(CYAN(BOLD(f"  tt-chat → {args.url}")))
    print(DIM(f"  server: {body}"))
    print(DIM("  /help for commands · multi-line: end with \\"))
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
            elif cmd == "paste":
                pasted = _read_paste_block()
                if not pasted:
                    print(DIM("paste cancelled")); continue
                # If user supplied an argument, treat it as a header.
                user = (arg + "\n\n" + pasted) if arg else pasted
                print(DIM(f"  [pasted {len(pasted)} chars, "
                          f"{pasted.count(chr(10))+1} lines]"))
                # fall through into the "send turn" path
            elif cmd == "yank":
                last_assistant = next(
                    (m["content"] for m in reversed(messages)
                     if m["role"] == "assistant" and m.get("content")),
                    None,
                )
                if not last_assistant:
                    print(DIM("nothing to yank")); continue
                want_code = arg.strip() in ("code", "block", "last")
                target = _last_code_block(last_assistant) if want_code else None
                if want_code and not target:
                    print(DIM("no code block in last reply")); continue
                if target is None:
                    target = last_assistant
                ok, tool = _clipboard_copy(target)
                if ok:
                    print(DIM(f"  [copied {len(target)} chars via {tool}]"))
                else:
                    print(RED("  [no clipboard tool found — install "
                              "pbcopy/xclip/wl-copy/xsel]"))
                continue
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
                        print()
                        print(panel("error", RED(str(payload)), color=RED))
                        reply = None; break
            except KeyboardInterrupt:
                print(); print(DIM("interrupted")); reply = None
            except Exception as e:
                print()
                print(panel("error", RED(f"{type(e).__name__}: {e}"), color=RED))
                reply = None
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
    try:
        main()
    finally:
        _reset_terminal()
