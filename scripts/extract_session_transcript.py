#!/usr/bin/env python3
"""
extract_session_transcript.py

Stream-parse a Claude Code transcript JSONL into a single Markdown document
the user can read end-to-end (with optional HTML sibling).

Reusable artifact: pass `--jsonl <path>` and `--out-dir <dir>`; script writes
session_transcript.md (and optionally session_transcript.html).

Schema (per line in the JSONL):
  - type=user        message.role=user        message.content=str | [blocks]
  - type=assistant   message.role=assistant   message.content=[blocks]
  - blocks may be {type: text, text}, {type: tool_use, name, input, id},
                  {type: tool_result, tool_use_id, content}, {type: thinking, ...}
  - Other top-level types (permission-mode, file-history-snapshot, ai-title,
    last-prompt, system, queue-operation, bridge-session, attachment) are
    metadata noise and are skipped.

System reminders: real user messages often have content wrapping
<system-reminder>...</system-reminder> blocks injected by the harness. We
strip those tags and their contents (but keep the user's actual text). A user
turn whose content is ENTIRELY system reminders is dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any


SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL
)

# Harness-injected wrapper tags whose content is plumbing (not real user text).
# A user message whose body is ONLY these wrappers (after stripping system
# reminders) is dropped.
HARNESS_WRAPPER_TAGS = (
    "task-notification",
    "local-command-caveat",
    "local-command-stdout",
    "local-command-stderr",
    "command-name",
    "command-message",
    "command-args",
    "bash-input",
    "bash-stdout",
    "bash-stderr",
    "start",
    "path",
    "json",
    "fn",
)
HARNESS_WRAPPER_RE = re.compile(
    r"<(" + "|".join(HARNESS_WRAPPER_TAGS) + r")\b[^>]*>.*?</\1>",
    re.DOTALL,
)


def strip_system_reminders(text: str) -> str:
    """Remove <system-reminder>...</system-reminder> blocks from a user message."""
    return SYSTEM_REMINDER_RE.sub("", text).strip()


def strip_harness_wrappers(text: str) -> str:
    """Also strip task-notification and friends. Iterate to handle nesting."""
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = HARNESS_WRAPPER_RE.sub("", cur)
    return cur.strip()


def fmt_ts(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        # ISO 8601 with Z suffix
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ts


def md_escape(text: str) -> str:
    """Light markdown escape — only for inline characters that would break headings.
    We do NOT want to escape inside code blocks, so callers should only use this
    for prose-level text. Keep it minimal.
    """
    # Strip out NUL and other control bytes that confuse renderers.
    return text.replace("\x00", "").rstrip()


def extract_text_from_content(content: Any) -> str:
    """Return concatenated text from any content shape (str | list of blocks).
    Tool blocks are NOT included here; callers handle them separately.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                parts.append(b.get("text", "") or "")
            elif bt == "thinking":
                # Skip thinking blocks by default (they're internal).
                continue
        return "\n".join(parts)
    return ""


def collect_blocks(content: Any) -> list[dict]:
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def fence(lang: str, body: str) -> str:
    # Use a 4-backtick fence so embedded triple backticks survive.
    body = body.replace("\r\n", "\n").rstrip()
    return f"````{lang}\n{body}\n````"


def fmt_tool_input(inp: Any) -> str:
    try:
        s = json.dumps(inp, indent=2, ensure_ascii=False)
    except Exception:
        s = str(inp)
    if len(s) > 8000:
        s = s[:8000] + "\n... [truncated]"
    return s


def fmt_tool_result_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", "") or "")
                else:
                    parts.append(json.dumps(b, ensure_ascii=False))
            else:
                parts.append(str(b))
        return "\n".join(parts)
    return str(content)


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " ..."


def toc_heading_for(user_text: str) -> str:
    """First 60 chars of the (system-reminder-stripped) user message, single line."""
    cleaned = strip_system_reminders(user_text)
    # Collapse whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "(empty user turn)"
    return truncate(cleaned, 60)


def slugify(text: str, used: set[str]) -> str:
    s = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    s = re.sub(r"\s+", "-", s) or "turn"
    base = s
    i = 2
    while s in used:
        s = f"{base}-{i}"
        i += 1
    used.add(s)
    return s


def is_real_user_turn(message: dict) -> tuple[bool, str]:
    """Return (is_real, cleaned_text). A 'real' user turn has non-empty text after
    stripping system reminders AND harness wrapper tags. Pure tool_result-only
    user messages and pure harness-plumbing messages are NOT real user turns."""
    content = message.get("content")
    raw = extract_text_from_content(content)
    cleaned = strip_system_reminders(raw)
    # Drop if the entire message is harness plumbing (task-notification etc.).
    after_wrappers = strip_harness_wrappers(cleaned)
    if not after_wrappers.strip():
        return False, ""
    return True, after_wrappers


def has_tool_result(message: dict) -> bool:
    for b in collect_blocks(message.get("content")):
        if b.get("type") == "tool_result":
            return True
    return False


def parse_jsonl(path: str):
    """Yield (line_no, obj) for each parseable JSON line."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def build_transcript(jsonl_path: str) -> tuple[str, dict]:
    """First pass: collect all events (skipping noise). Second pass: emit markdown."""

    # Index tool_use_id -> tool_result block, so we can attach results to their calls.
    tool_results_by_id: dict[str, dict] = {}
    # Ordered list of events we care about.
    events: list[dict] = []

    first_ts: str | None = None
    last_ts: str | None = None

    tool_counter: Counter[str] = Counter()
    user_real_turns = 0
    assistant_turns = 0
    tool_calls_total = 0
    edits_total = 0
    commits_total = 0
    # Track largest text payload for the "anything surprising" report.
    largest_text = ("", 0)  # (label, bytes)

    for line_no, d in parse_jsonl(jsonl_path):
        t = d.get("type")
        if t not in ("user", "assistant"):
            # Skip metadata: permission-mode, file-history-snapshot, ai-title,
            # last-prompt, system, queue-operation, bridge-session, attachment.
            continue

        ts = d.get("timestamp")
        if ts:
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts > last_ts:
                last_ts = ts

        msg = d.get("message") or {}
        role = msg.get("role") or t
        content = msg.get("content")

        # Index tool_results inside any (user-role) message for later lookup.
        if role == "user":
            for b in collect_blocks(content):
                if b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    if tid:
                        tool_results_by_id[tid] = b

        # Decide whether this is a real user turn or tool plumbing.
        if role == "user":
            real, cleaned = is_real_user_turn(msg)
            if not real:
                continue
            # Track the largest payload — sometimes a paste is huge.
            size = len(cleaned.encode("utf-8"))
            if size > largest_text[1]:
                largest_text = (f"user turn @ {fmt_ts(ts) or 'unknown ts'}", size)
            user_real_turns += 1
            events.append(
                {
                    "kind": "user",
                    "ts": ts,
                    "text": cleaned,
                    "line_no": line_no,
                }
            )
            continue

        # role == 'assistant'
        text = extract_text_from_content(content)
        tool_uses = [b for b in collect_blocks(content) if b.get("type") == "tool_use"]

        # Skip empty assistant blips (no text, no tool calls).
        if not text.strip() and not tool_uses:
            continue

        assistant_turns += 1
        for tu in tool_uses:
            name = tu.get("name", "?")
            tool_counter[name] += 1
            tool_calls_total += 1
            if name in ("Edit", "Write", "NotebookEdit"):
                edits_total += 1
            if name == "Bash":
                cmd = (tu.get("input") or {}).get("command", "")
                if isinstance(cmd, str) and "git commit" in cmd:
                    commits_total += 1

        if text:
            size = len(text.encode("utf-8"))
            if size > largest_text[1]:
                largest_text = (f"assistant turn @ {fmt_ts(ts) or 'unknown ts'}", size)

        events.append(
            {
                "kind": "assistant",
                "ts": ts,
                "text": text,
                "tool_uses": tool_uses,
                "line_no": line_no,
            }
        )

    # Emit markdown.
    md_parts: list[str] = []
    used_slugs: set[str] = set()

    # 1. Title + metadata.
    md_parts.append("# Claude Code session transcript")
    md_parts.append("")
    md_parts.append(f"- **Session UUID**: `d666214a-44ea-4685-88ff-493934e2b315`")
    md_parts.append(f"- **Date range**: {fmt_ts(first_ts)} -> {fmt_ts(last_ts)}")
    md_parts.append(
        f"- **Events (real)**: {user_real_turns} user turns, "
        f"{assistant_turns} assistant turns, {tool_calls_total} tool calls"
    )
    md_parts.append(f"- **Source JSONL**: `{jsonl_path}`")
    md_parts.append("")

    # 2. Table of contents.
    md_parts.append("## Table of contents")
    md_parts.append("")
    turn_index = 0
    user_turn_slugs: list[tuple[str, str]] = []  # (heading_text, slug)
    for ev in events:
        if ev["kind"] != "user":
            continue
        turn_index += 1
        heading = toc_heading_for(ev["text"])
        slug = slugify(f"{turn_index:04d}-{heading}", used_slugs)
        user_turn_slugs.append((heading, slug))
        md_parts.append(f"{turn_index}. [{heading}](#{slug})")
        ev["_slug"] = slug
        ev["_turn_index"] = turn_index
    md_parts.append("")

    # 3. Body.
    md_parts.append("## Conversation")
    md_parts.append("")

    for ev in events:
        ts_str = fmt_ts(ev.get("ts"))
        if ev["kind"] == "user":
            slug = ev.get("_slug", "")
            md_parts.append(f'<a id="{slug}"></a>')
            md_parts.append("")
            md_parts.append(f"## User [{ts_str}]")
            md_parts.append("")
            md_parts.append(md_escape(ev["text"]))
            md_parts.append("")
            continue

        # assistant
        md_parts.append(f"### Assistant [{ts_str}]")
        md_parts.append("")
        if ev["text"].strip():
            md_parts.append(md_escape(ev["text"]))
            md_parts.append("")

        for tu in ev["tool_uses"]:
            name = tu.get("name", "?")
            inp = tu.get("input")
            tid = tu.get("id")
            md_parts.append(f"> **Tool:** `{name}`")
            md_parts.append("")
            md_parts.append(fence("json", fmt_tool_input(inp)))
            md_parts.append("")
            # Attach tool result if we found one.
            result = tool_results_by_id.get(tid)
            if result is not None:
                result_text = fmt_tool_result_content(result.get("content"))
                # Truncate very long outputs in the visible details summary.
                trunc = result_text
                if len(trunc) > 20000:
                    trunc = trunc[:20000] + "\n... [tool result truncated]"
                is_err = result.get("is_error", False)
                summary = "tool result"
                if is_err:
                    summary = "tool result (ERROR)"
                md_parts.append("<details>")
                md_parts.append(f"<summary>{summary}</summary>")
                md_parts.append("")
                md_parts.append(fence("", trunc))
                md_parts.append("")
                md_parts.append("</details>")
                md_parts.append("")

    # 4. Stats footer.
    md_parts.append("---")
    md_parts.append("")
    md_parts.append("## Stats")
    md_parts.append("")
    md_parts.append(f"- Total user turns (real, post-strip): **{user_real_turns}**")
    md_parts.append(f"- Total assistant turns: **{assistant_turns}**")
    md_parts.append(f"- Total tool calls: **{tool_calls_total}**")
    md_parts.append(f"- Total file edits (Edit/Write/NotebookEdit): **{edits_total}**")
    md_parts.append(f"- Git commits during session: **{commits_total}**")
    md_parts.append("")
    md_parts.append("### Top tools by count")
    md_parts.append("")
    md_parts.append("| Tool | Calls |")
    md_parts.append("| --- | ---: |")
    for name, n in tool_counter.most_common(10):
        md_parts.append(f"| `{name}` | {n} |")
    md_parts.append("")
    md_parts.append("### Notable")
    md_parts.append("")
    md_parts.append(
        f"- Largest single text payload: **{largest_text[1]:,} bytes** "
        f"({largest_text[0]})"
    )
    md_parts.append("")

    md_text = "\n".join(md_parts) + "\n"

    stats = {
        "user_real_turns": user_real_turns,
        "assistant_turns": assistant_turns,
        "tool_calls_total": tool_calls_total,
        "edits_total": edits_total,
        "commits_total": commits_total,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "top_tools": tool_counter.most_common(10),
        "largest_text": largest_text,
    }
    return md_text, stats


def maybe_write_html(md_text: str, out_path: str) -> bool:
    try:
        import markdown  # type: ignore
    except Exception:
        return False
    body = markdown.markdown(
        md_text,
        extensions=["fenced_code", "tables", "toc"],
        output_format="html5",
    )
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
           max-width: 1100px; margin: 2em auto; padding: 0 1em;
           line-height: 1.55; color: #222; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                font-size: 12.5px; }
    pre { background: #f5f5f5; padding: 0.9em; overflow-x: auto;
          border-radius: 6px; line-height: 1.4; }
    code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
    pre code { background: transparent; padding: 0; }
    h1, h2, h3 { line-height: 1.25; }
    h2 { border-bottom: 1px solid #ddd; padding-bottom: 0.2em; margin-top: 1.8em; }
    h3 { margin-top: 1.5em; }
    details { margin: 0.5em 0 1em; }
    summary { cursor: pointer; color: #555; }
    blockquote { border-left: 3px solid #888; margin: 0.5em 0; padding: 0.2em 0.8em;
                 color: #444; background: #fafafa; }
    table { border-collapse: collapse; margin: 0.8em 0; }
    th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: left; }
    th { background: #f0f0f0; }
    a { color: #0a58ca; }
    """
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Claude Code session transcript</title>"
        f"<style>{css}</style></head><body>{body}</body></html>"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        required=True,
        help="Path to the Claude Code transcript JSONL.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write session_transcript.md (and optional .html).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, "session_transcript.md")
    html_path = os.path.join(args.out_dir, "session_transcript.html")

    md_text, stats = build_transcript(args.jsonl)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    md_size = os.path.getsize(md_path)
    md_lines = md_text.count("\n")

    wrote_html = maybe_write_html(md_text, html_path)

    print(f"Wrote {md_path} ({md_size:,} bytes, {md_lines:,} lines)")
    if wrote_html:
        print(f"Wrote {html_path} ({os.path.getsize(html_path):,} bytes)")
    else:
        print("HTML: skipped (markdown library not installed)")
    print("Stats:")
    print(f"  user_real_turns:  {stats['user_real_turns']}")
    print(f"  assistant_turns:  {stats['assistant_turns']}")
    print(f"  tool_calls_total: {stats['tool_calls_total']}")
    print(f"  edits_total:      {stats['edits_total']}")
    print(f"  commits_total:    {stats['commits_total']}")
    print(f"  date range:       {fmt_ts(stats['first_ts'])} -> {fmt_ts(stats['last_ts'])}")
    print(f"  top tools:        {stats['top_tools']}")
    print(f"  largest payload:  {stats['largest_text'][1]:,} bytes ({stats['largest_text'][0]})")


if __name__ == "__main__":
    main()
