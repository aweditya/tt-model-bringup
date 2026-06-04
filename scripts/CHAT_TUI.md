# tt-chat TUI — `scripts/chat.py`

A stdlib-only terminal chat client for the CB OpenAI server. Designed to be
robust enough to live-demo at a tech conference.

## Quickstart

```
ssh -L 8000:localhost:8000 qb1
python3 scripts/chat.py                # default: greedy, max 1024, <think> hidden
python3 scripts/chat.py --tools        # enable shell / read_file / write_file / calc
python3 scripts/chat.py --show-think   # show the model's <think>…</think> reasoning
python3 scripts/chat.py --temp 0.7 --seed 42
```

The welcome panel is Claude-Code-styled: a closed box showing the URL, model,
working directory, and current settings. Each assistant turn opens with
`● assistant (<model_short>)` and a thin grey rule.

Qwen3.6 + Gemma 4 IT both emit `<think>…</think>` blocks. By default these
are suppressed from the rendered stream (replaced with a single dim
`(thinking…)` hint); the raw text is still appended to history so the
model has full context on the next turn. Use `/think` mid-session or
`--show-think` at launch to surface them.

`--url http://...` points at a different server (default
`http://localhost:8000`).

## Slash commands

| Command                | What it does                                              |
|------------------------|-----------------------------------------------------------|
| `/new` / `/clear`      | Clear history (keep the system prompt)                    |
| `/sys <text>`          | Set or replace the system prompt                          |
| `/temp <float>`        | Set temperature (0 = greedy)                              |
| `/top-p <float>`       | Set top_p                                                 |
| `/top-k <int>`         | Set top_k (0 disables)                                    |
| `/seed <int>`          | Fix the seed                                              |
| `/max <int>`           | Set `max_tokens` per turn                                 |
| `/tools`               | Toggle built-in tool calling                              |
| `/think`               | Toggle visibility of `<think>…</think>` blocks (default hidden) |
| `/continue`            | Resume after `finish=length`                              |
| `/status` / `/show`    | Panel of current url / model / params / history counts    |
| `/history`             | Dump transcript (truncated per message)                   |
| `/save <file>`         | Save transcript to JSON                                   |
| `/load <file>`         | Load transcript from JSON                                 |
| `/paste [header]`      | Multi-line paste mode (terminator: `:end:` or 3 blanks)   |
| `/yank [code]`         | Copy last reply (or its last code block) to clipboard     |
| `/metrics [N\|raw]`    | Live Prometheus dashboard (N cycles, default 10)          |
| `/screenshot`          | Save `presentation/screenshots/tui_<ts>.png`              |
| `/help`                | Show in-app help                                          |
| `/exit` / `/quit`      | Leave                                                     |

## Input modes

- **Plain prompt**: Type and press Enter. On a TTY the input is read in
  raw mode so paste is detected automatically.
- **Bracketed paste**: The TUI sets `\x1b[?2004h` on startup; modern
  terminals wrap pastes in `\x1b[200~ ... \x1b[201~`. Newlines inside
  the wrap are kept as literal `\n` — the message is only submitted
  when Enter is pressed *outside* a paste.
- **Burst heuristic**: Terminals that don't honour bracketed paste fall
  back to a 50 ms inter-byte timer. Bytes arriving faster than 50 ms
  apart are treated as paste content.
- **`/paste`**: Explicit multi-line mode for terminals that don't pass
  paste reliably. Each line is read normally; the block ends with a
  line whose only content is `:end:` (or 3 consecutive blank lines).
- **Fallback (no TTY / no termios)**: Trailing `\` continues onto the
  next line; an empty continuation submits.

`Ctrl-C` cancels the current input line. `Ctrl-D` on an empty line
exits.

## Clipboard (`/yank`)

`/yank` copies the most recent assistant reply to the system clipboard
via the first available of `pbcopy`, `wl-copy`, `xclip`, or `xsel`.
`/yank code` copies only the last fenced ```` ``` ```` block from that
reply.

## Built-in tools (`--tools` / `/tools`)

The CB server doesn't emit OpenAI-style `tool_calls` deltas yet, so the
TUI parses the assistant's plain-text output for any of:

```
```tool_code
{"name": "...", "arguments": {...}}
```
```

or `<tool_call>{...}</tool_call>` or a `json`-tagged code block with a
`"name"` field. Each detected call is executed, its (args, stdout) is
rendered in a labelled panel, and the result is appended as a
`role=tool` message so the model can take a second turn.

### `shell`

Read-only subprocess. Allow-listed binaries:

```
cat date diff echo egrep fgrep file find git grep head ls pwd
python python3 stat tail tree which wc
```

Anything else is refused. Token-level deny-list blocks `rm`, `kill`,
`shutdown`, `curl`, `wget`, pipes, redirects, command substitution,
and process chaining. `python`/`python3` is further capped to
`--version` / `-V` / `-c <snippet>` and the snippet must not contain
`open(`, `subprocess`, `os.system`, `import os`, `socket`, `urllib`,
`requests`, `http.client`, or `shutil`. CWD is pinned to the project
root. 5-second timeout.

### `read_file`

```
read_file(path, start=1, n=200)
```

Returns the slice `[start, start+n)` of lines as `NNN: text` lines so
the model can cite line numbers. `n` is capped at 1000; the file
itself is capped at 64 KB.

### `write_file`

```
write_file(path, content, mode="write")
```

- `path` is resolved relative to the project root and `realpath`-jailed
  inside it. Symlinks that point out are refused.
- Paths containing `.env`, `secret`, `key`, `token`, `credential`,
  `.pem`, `.p12`, or `.pfx` (case-insensitive) are refused.
- `mode="write"` refuses to overwrite an existing file. Use
  `mode="append"`, or read+rewrite, to modify one.

### `calc`

```
calc(expr)
```

Plain arithmetic only (`re.search(r"[A-Za-z_]", expr)` refuses
identifiers). No `__builtins__`.

## Live `/metrics` dashboard

`/metrics` polls `/metrics` once per second for 10 cycles by default
(`/metrics 30` overrides). Counters shown:

```
cb_tokens_generated_total
cb_step_seconds_sum / cb_step_seconds_count → avg ms/step
cb_prefix_cache_hits_total / cb_prefix_cache_misses_total
cb_slots_active
cb_requests_total
```

Deltas between refreshes are highlighted green (increase) or red
(decrease). `/metrics raw` dumps the unparsed Prometheus text.

## Robustness

- All HTTP failures (connection refused, 5xx, dropped stream) surface
  as a red error panel; the prompt loop keeps running.
- `main()` is wrapped in `try / finally` that resets cursor (`\x1b[?25h`),
  colour (`\x1b[0m`), clears the current line (`\x1b[2K\r`), and
  disables bracketed paste (`\x1b[?2004l`) on every exit path.
- The streaming markdown renderer word-wraps prose lines to
  `$COLUMNS` (falling back to `shutil.get_terminal_size`) with
  `textwrap.wrap(..., break_long_words=False)` to avoid mid-word
  splits. Code-fence content is emitted verbatim so syntax isn't
  mangled.
