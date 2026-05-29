#!/usr/bin/env python3
"""Tenstorrent ecosystem scraper.

Two phases that can be run independently or together:

  1. ``docs``   — BFS-walk ``docs.tenstorrent.com``. Convert each HTML
                  page to markdown, store under ``tt_docs_corpus/docs.tenstorrent.com/``.
                  Idempotent: pages are overwritten in place, links are
                  deduped against the on-disk corpus.

  2. ``github`` — For each configured Tenstorrent repo, pull:
                  * open + recently closed issues  -> ``issues.jsonl``
                  * open + recently merged PRs     -> ``pulls.jsonl``
                  * branch names                   -> ``branches.jsonl``
                  Output lives at ``tt_docs_corpus/github/<repo>/``.

Use ``--phase docs|github|both`` to control phases, ``--max-pages N`` to
cap the docs crawl for testing.

Notes
-----
* Authenticated GitHub requests (``GITHUB_TOKEN`` env var) get 5000 req/h.
  Anonymous requests get 60 req/h — enough for a one-off scrape if we
  page conservatively, but the token is strongly recommended.
* Concurrency is bounded by a thread-pool size; we keep it small (5-10)
  to be polite to both targets and stay under the GitHub secondary
  rate-limit (~100 concurrent requests).
* This script runs LOCALLY. It does not touch the device. The existing
  ``scrape_tt_docs.py`` is left untouched; this is a superset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.parse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify as html_to_md
except ImportError as exc:  # pragma: no cover - import-time guard
    print(f"missing dep: {exc}", file=sys.stderr)
    print(
        "install with: pip install --user --break-system-packages "
        "requests beautifulsoup4 markdownify",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tt_docs_corpus"

DOCS_SEEDS: list[str] = [
    "https://docs.tenstorrent.com/",
    "https://docs.tenstorrent.com/tt-metal/latest/ttnn/",
    "https://docs.tenstorrent.com/tt-metal/latest/tt_metal/",
]
DOCS_DOMAIN = "docs.tenstorrent.com"

GITHUB_REPOS: list[str] = [
    "tenstorrent/tt-metal",
    "tenstorrent/tt-mlir",
    "tenstorrent/tt-forge-fe",
    "tenstorrent/tt-xla",
    "tenstorrent/tt-torch",
    "tenstorrent/tt-blacksmith",
]

DOCS_SKIP_EXTS = {
    ".zip", ".tar", ".tgz", ".gz", ".png", ".jpg", ".jpeg", ".gif",
    ".svg", ".pdf", ".mp4", ".webm", ".woff", ".woff2", ".ttf", ".eot",
    ".css", ".js", ".json", ".xml", ".ico", ".map",
}

DOCS_SKIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"/_sources/"),
    re.compile(r"/_static/"),
    re.compile(r"/genindex"),
    re.compile(r"/search\.html"),
]

REQUEST_TIMEOUT = 20
DOCS_DEFAULT_MAX_PAGES = 2500
DOCS_DEFAULT_MAX_DEPTH = 7
DOCS_SLEEP = 0.2  # per-worker per-request

GITHUB_API = "https://api.github.com"
GITHUB_PER_PAGE = 100
GITHUB_CLOSED_LOOKBACK_DAYS = 180  # only keep recently closed/merged

USER_AGENT = "tt-xla-ecosystem-scraper/0.2 (research)"


log = logging.getLogger("scrape_tt_ecosystem")


# ---------------------------------------------------------------------------
# Phase 1: docs BFS
# ---------------------------------------------------------------------------


@dataclass
class DocsResult:
    """Per-URL outcome from the docs crawl."""

    url: str
    status: str
    path: Path | None = None


def normalize_url(url: str) -> str:
    """Drop fragment; return the URL unchanged otherwise.

    Trailing slashes are preserved because they meaningfully select the
    directory index on most Sphinx-style doc sites.
    """
    parts = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(parts._replace(fragment=""))


def url_to_filepath(url: str) -> Path:
    """Map a docs URL to its on-disk markdown path under ``OUTPUT_DIR``."""
    parts = urllib.parse.urlparse(url)
    path = parts.path or "/"
    if path.endswith("/"):
        path += "index"
    if not path.endswith((".html", ".md", ".htm")):
        path += ".html"
    path = re.sub(r"\.html?$", ".md", path)
    return OUTPUT_DIR / parts.netloc / path.lstrip("/")


def is_docs_skippable(url: str) -> bool:
    """Return True if the URL is not worth fetching as a doc page."""
    parts = urllib.parse.urlparse(url)
    if not parts.scheme.startswith("http"):
        return True
    ext = os.path.splitext(parts.path)[1].lower()
    if ext in DOCS_SKIP_EXTS:
        return True
    return any(pat.search(url) for pat in DOCS_SKIP_PATTERNS)


def is_same_domain(url: str, domain: str) -> bool:
    """True if ``url`` is on ``domain`` (or a subdomain thereof)."""
    netloc = urllib.parse.urlparse(url).netloc
    return netloc == domain or netloc.endswith("." + domain)


def extract_links(html: str, base_url: str) -> list[str]:
    """Parse ``<a href>`` links from HTML, returning absolute URLs."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        absolute = urllib.parse.urljoin(base_url, a["href"])
        absolute = normalize_url(absolute)
        if absolute and not is_docs_skippable(absolute):
            out.append(absolute)
    return out


def html_to_markdown(html: str, source_url: str) -> str:
    """Convert a doc-site HTML page to a markdown string with frontmatter."""
    soup = BeautifulSoup(html, "html.parser")
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(role="main")
        or soup.find("div", class_="document")
        or soup.find("div", class_="body")
        or soup.find("div", class_="rst-content")
        or soup.find("div", class_="wy-nav-content")
        or soup.body
    )
    if main is None:
        return f"# (no content extracted)\n\nSource: {source_url}\n"

    for tag in main.find_all(["nav", "footer", "aside", "header", "script", "style"]):
        tag.decompose()
    for cls in ("edit-on-github", "breadcrumbs", "related", "sphinxsidebar"):
        for tag in main.find_all(class_=cls):
            tag.decompose()

    md = html_to_md(str(main), heading_style="ATX")
    md = re.sub(r"\n{4,}", "\n\n\n", md)
    header = (
        "---\n"
        f"source: {source_url}\n"
        f"scraped: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "---\n\n"
    )
    return header + md


def fetch_docs_page(session: requests.Session, url: str) -> tuple[str | None, str]:
    """Fetch one doc URL. Returns ``(html_or_none, status_string)``."""
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return None, f"error: {type(exc).__name__}"
    if r.status_code != 200:
        return None, f"http {r.status_code}"
    ctype = r.headers.get("Content-Type", "")
    if "html" not in ctype:
        return None, f"not-html ({ctype})"
    return r.text, "ok"


def crawl_docs(
    seeds: Iterable[str],
    max_pages: int,
    max_depth: int,
    workers: int,
) -> dict[str, DocsResult]:
    """BFS-walk the docs site with a bounded thread pool.

    Pages are written to disk as soon as they are fetched (idempotent
    overwrite). Returns a dict ``{url: DocsResult}`` summarizing every
    URL visited.
    """
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque()
    results: dict[str, DocsResult] = {}

    for seed in seeds:
        n = normalize_url(seed)
        if n not in seen:
            seen.add(n)
            queue.append((n, 0))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    pages_saved = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while queue and pages_saved < max_pages:
            # Pull a batch up to `workers` URLs to fetch in parallel.
            batch: list[tuple[str, int]] = []
            while queue and len(batch) < workers and pages_saved + len(batch) < max_pages:
                batch.append(queue.popleft())
            if not batch:
                break

            futures = {
                pool.submit(fetch_docs_page, session, url): (url, depth)
                for url, depth in batch
            }
            for fut in as_completed(futures):
                url, depth = futures[fut]
                html, status = fut.result()
                if html is None:
                    results[url] = DocsResult(url=url, status=status)
                    log.warning("skip %s (%s)", url, status)
                    continue

                out_path = url_to_filepath(url)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(html_to_markdown(html, url), encoding="utf-8")
                pages_saved += 1
                results[url] = DocsResult(url=url, status="saved", path=out_path)
                log.info("[%d/%d] d=%d %s", pages_saved, max_pages, depth, url)

                if depth < max_depth:
                    for link in extract_links(html, url):
                        if link in seen:
                            continue
                        if not is_same_domain(link, DOCS_DOMAIN):
                            continue
                        seen.add(link)
                        queue.append((link, depth + 1))

            time.sleep(DOCS_SLEEP)

    return results


# ---------------------------------------------------------------------------
# Phase 2: GitHub
# ---------------------------------------------------------------------------


def github_session(token: str | None) -> requests.Session:
    """Build a ``requests.Session`` with GitHub-appropriate headers."""
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        }
    )
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def github_paginate(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    max_pages: int = 50,
) -> Iterator[dict[str, Any]]:
    """Yield each item from a paginated GitHub list endpoint.

    Follows the ``Link: rel="next"`` header. Caps at ``max_pages`` to
    avoid runaway loops against e.g. ``tt-metal`` (5000+ issues).
    """
    params = dict(params or {})
    params.setdefault("per_page", GITHUB_PER_PAGE)
    page_count = 0
    while url and page_count < max_pages:
        try:
            r = session.get(url, params=params if page_count == 0 else None,
                            timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            log.error("github GET %s failed: %s", url, exc)
            return
        if r.status_code == 403 and "rate limit" in r.text.lower():
            reset = r.headers.get("X-RateLimit-Reset")
            log.error("rate-limited; X-RateLimit-Reset=%s", reset)
            return
        if r.status_code != 200:
            log.error("github GET %s -> http %d: %s", url, r.status_code, r.text[:200])
            return
        data = r.json()
        if not isinstance(data, list):
            log.error("expected list from %s, got %s", url, type(data).__name__)
            return
        for item in data:
            yield item
        page_count += 1
        url = _next_link(r.headers.get("Link", ""))


def _next_link(link_header: str) -> str:
    """Parse a ``Link`` header and return the ``rel="next"`` URL or ``""``."""
    if not link_header:
        return ""
    for part in link_header.split(","):
        bits = part.strip().split(";")
        if len(bits) < 2:
            continue
        url = bits[0].strip().lstrip("<").rstrip(">")
        rel = bits[1].strip()
        if rel == 'rel="next"':
            return url
    return ""


def _slim_issue(item: dict[str, Any]) -> dict[str, Any]:
    """Project an issue/PR API object down to the fields we care about."""
    return {
        "number": item.get("number"),
        "title": item.get("title"),
        "state": item.get("state"),
        "user": (item.get("user") or {}).get("login"),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "closed_at": item.get("closed_at"),
        "merged_at": item.get("pull_request", {}).get("merged_at")
        if item.get("pull_request")
        else None,
        "labels": [lab.get("name") for lab in item.get("labels") or []],
        "html_url": item.get("html_url"),
        "is_pull_request": "pull_request" in item,
        "body": item.get("body") or "",
    }


def _slim_pr(item: dict[str, Any]) -> dict[str, Any]:
    """Project a pull-request API object down to the fields we care about."""
    return {
        "number": item.get("number"),
        "title": item.get("title"),
        "state": item.get("state"),
        "user": (item.get("user") or {}).get("login"),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "closed_at": item.get("closed_at"),
        "merged_at": item.get("merged_at"),
        "draft": item.get("draft"),
        "base": (item.get("base") or {}).get("ref"),
        "head": (item.get("head") or {}).get("ref"),
        "labels": [lab.get("name") for lab in item.get("labels") or []],
        "html_url": item.get("html_url"),
        "body": item.get("body") or "",
    }


def _recent_enough(item: dict[str, Any], cutoff_iso: str) -> bool:
    """Closed/merged items older than ``cutoff_iso`` should be dropped."""
    closed = item.get("closed_at") or item.get("merged_at")
    if not closed:  # still open
        return True
    return closed >= cutoff_iso


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write ``rows`` to ``path`` as JSONL. Returns the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def scrape_one_repo(session: requests.Session, repo: str, out_dir: Path) -> dict[str, int]:
    """Fetch issues / PRs / branches for one repo into ``out_dir/<repo>/``.

    Returns a dict ``{"issues": N, "pulls": N, "branches": N}``.
    """
    repo_dir = out_dir / repo.split("/", 1)[1]
    repo_dir.mkdir(parents=True, exist_ok=True)

    cutoff = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() - GITHUB_CLOSED_LOOKBACK_DAYS * 86400),
    )

    # Issues: ``state=all`` returns both open and closed (including PRs,
    # because GitHub treats PRs as a special kind of issue). We split
    # them into the two files.
    log.info("[github] %s: issues", repo)
    issues_rows: list[dict[str, Any]] = []
    pulls_rows: list[dict[str, Any]] = []
    for item in github_paginate(
        session,
        f"{GITHUB_API}/repos/{repo}/issues",
        params={"state": "all", "sort": "updated", "direction": "desc"},
    ):
        if not _recent_enough(item, cutoff):
            continue
        slim = _slim_issue(item)
        if slim["is_pull_request"]:
            pulls_rows.append(slim)
        else:
            issues_rows.append(slim)

    # PRs via the dedicated endpoint give us merge metadata + base/head
    # branch refs that the /issues endpoint lacks.
    log.info("[github] %s: pulls", repo)
    pulls_full: list[dict[str, Any]] = []
    for item in github_paginate(
        session,
        f"{GITHUB_API}/repos/{repo}/pulls",
        params={"state": "all", "sort": "updated", "direction": "desc"},
    ):
        if not _recent_enough(item, cutoff):
            continue
        pulls_full.append(_slim_pr(item))

    # Branches: small list, single endpoint
    log.info("[github] %s: branches", repo)
    branches_rows = [
        {
            "name": b.get("name"),
            "commit_sha": (b.get("commit") or {}).get("sha"),
            "protected": b.get("protected"),
        }
        for b in github_paginate(
            session,
            f"{GITHUB_API}/repos/{repo}/branches",
            max_pages=20,
        )
    ]

    n_issues = write_jsonl(repo_dir / "issues.jsonl", issues_rows)
    # Prefer the richer /pulls payload; fall back to the issues-side slice
    # if /pulls came back empty (auth issues, repo with no PRs, etc.).
    n_pulls = write_jsonl(repo_dir / "pulls.jsonl", pulls_full or pulls_rows)
    n_branches = write_jsonl(repo_dir / "branches.jsonl", branches_rows)

    return {"issues": n_issues, "pulls": n_pulls, "branches": n_branches}


def scrape_github(
    repos: Iterable[str],
    token: str | None,
    workers: int,
) -> dict[str, dict[str, int]]:
    """Scrape every repo in ``repos`` in parallel; return per-repo counts."""
    session = github_session(token)
    out_dir = OUTPUT_DIR / "github"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, int]] = {}
    # We share a Session across threads. ``requests.Session`` is
    # documented as thread-safe for GET-only workloads, which is what
    # we do here.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scrape_one_repo, session, repo, out_dir): repo for repo in repos}
        for fut in as_completed(futures):
            repo = futures[fut]
            try:
                results[repo] = fut.result()
            except Exception:
                log.exception("repo %s failed", repo)
                results[repo] = {"issues": -1, "pulls": -1, "branches": -1}
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--phase",
        choices=["docs", "github", "both"],
        default="both",
        help="which phase(s) to run (default: both)",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=DOCS_DEFAULT_MAX_PAGES,
        help=f"docs: max pages to fetch (default {DOCS_DEFAULT_MAX_PAGES})",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=DOCS_DEFAULT_MAX_DEPTH,
        help=f"docs: max link-depth (default {DOCS_DEFAULT_MAX_DEPTH})",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="thread-pool size for each phase (default 6)",
    )
    p.add_argument(
        "--repos",
        nargs="+",
        default=GITHUB_REPOS,
        help="github: owner/repo slugs to scrape (default: built-in TT list)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="log DEBUG-level messages",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    log.info("output dir: %s", OUTPUT_DIR)

    t0 = time.time()

    if args.phase in ("docs", "both"):
        log.info("=== Phase 1: docs.tenstorrent.com ===")
        results = crawl_docs(
            DOCS_SEEDS,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            workers=args.workers,
        )
        saved = sum(1 for r in results.values() if r.status == "saved")
        errs = sum(1 for r in results.values() if r.status.startswith(("error", "http")))
        log.info("docs done: saved=%d errors=%d total_visited=%d", saved, errs, len(results))

        report_path = OUTPUT_DIR / "_crawl_report.txt"
        with report_path.open("w") as f:
            f.write(f"Crawl finished {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Saved={saved}, Errors={errs}, Total={len(results)}\n\n")
            for url, r in sorted(results.items()):
                f.write(f"{r.status:30s} {url}\n")
        log.info("wrote %s", report_path.relative_to(ROOT))

    if args.phase in ("github", "both"):
        log.info("=== Phase 2: GitHub ===")
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            log.warning("GITHUB_TOKEN not set; using 60 req/h anonymous limit")
        gh_results = scrape_github(args.repos, token=token, workers=min(args.workers, len(args.repos)))
        for repo, counts in sorted(gh_results.items()):
            log.info(
                "github %s: issues=%d pulls=%d branches=%d",
                repo, counts["issues"], counts["pulls"], counts["branches"],
            )

    log.info("total elapsed: %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
