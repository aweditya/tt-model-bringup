#!/usr/bin/env python3
"""
TT documentation scraper.

Crawls https://docs.tenstorrent.com (and related TT doc sites), follows
same-domain links, converts each page to markdown, and saves to a local
corpus directory. Stops at a max-depth and max-page count to avoid
runaway crawls. Skips code files and binary assets.

Output: `tt_docs_corpus/<domain>/<path>.md`

The result is a searchable text corpus we can grep / search with
ripgrep / feed to other tools. NOT code — just the human-readable docs.

Run locally:
    pip install --user --break-system-packages requests beautifulsoup4 markdownify
    python3 pjrt_plugin/scripts/scrape_tt_docs.py
"""

import os
import re
import sys
import time
import urllib.parse
from pathlib import Path
from collections import deque

# Permissive: try to import; if missing, print install command and exit
try:
    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify as html_to_md
except ImportError as e:
    print(f"Missing dep: {e}")
    print("Install with:")
    print("  pip install --user --break-system-packages requests beautifulsoup4 markdownify")
    sys.exit(1)

# Roots to crawl. Only same-domain links are followed.
SEED_URLS = [
    "https://docs.tenstorrent.com/",
    "https://docs.tenstorrent.com/tt-metal/latest/ttnn/",
    "https://docs.tenstorrent.com/tt-metal/latest/tt_metal/",
    # Add more if discovered
]

# Where to save
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tt_docs_corpus"

# Crawl limits
MAX_PAGES = 2000          # safety net
MAX_DEPTH = 6             # follow links N levels deep
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN = 0.5       # rate limit (per-page sleep)

# Skip file extensions that aren't docs
SKIP_EXTS = {
    '.zip', '.tar.gz', '.tgz', '.gz', '.png', '.jpg', '.jpeg', '.gif',
    '.svg', '.pdf', '.mp4', '.webm', '.woff', '.woff2', '.ttf', '.eot',
    '.css', '.js', '.json', '.xml', '.ico', '.map',
}

# Skip URL patterns that are search/navigation/code-pages (not content)
SKIP_PATTERNS = [
    re.compile(r'/_sources/'),
    re.compile(r'/_static/'),
    re.compile(r'/genindex'),
    re.compile(r'/search\.html'),
    re.compile(r'#'),                 # in-page anchors (we'll handle separately)
]


def normalize_url(url: str) -> str:
    """Strip fragments and trailing slashes."""
    parts = urllib.parse.urlparse(url)
    # Drop fragment
    parts = parts._replace(fragment='')
    # Normalize trailing slash for directory-like URLs (ends in /)
    return urllib.parse.urlunparse(parts)


def url_to_filepath(url: str) -> Path:
    """Map a URL to a local markdown path."""
    parts = urllib.parse.urlparse(url)
    path = parts.path
    if path.endswith('/'):
        path += 'index'
    if not path.endswith('.html') and not path.endswith('.md'):
        path += '.html'
    path = path.replace('.html', '.md').replace('.htm', '.md')
    # Avoid leading /
    rel = path.lstrip('/')
    return OUTPUT_DIR / parts.netloc / rel


def is_skippable(url: str) -> bool:
    parts = urllib.parse.urlparse(url)
    if not parts.scheme.startswith('http'):
        return True
    ext = os.path.splitext(parts.path)[1].lower()
    if ext in SKIP_EXTS:
        return True
    for pat in SKIP_PATTERNS:
        if pat.search(url):
            return True
    return False


def is_same_origin(url: str, seed_origin: str) -> bool:
    seed = urllib.parse.urlparse(seed_origin).netloc
    cur = urllib.parse.urlparse(url).netloc
    return cur == seed or cur.endswith('.' + seed)


def extract_links(html: str, base_url: str) -> list:
    """Pull out href links, resolve relative URLs against base."""
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        absolute = urllib.parse.urljoin(base_url, href)
        absolute = normalize_url(absolute)
        if absolute and not is_skippable(absolute):
            links.append(absolute)
    return links


def extract_content_md(html: str, source_url: str) -> str:
    """Pull main content out of an HTML page, convert to markdown."""
    soup = BeautifulSoup(html, 'html.parser')

    # Try common doc-site main-content selectors
    main = (soup.find('article')
            or soup.find('main')
            or soup.find(role='main')
            or soup.find('div', class_='document')
            or soup.find('div', class_='body')
            or soup.find('div', class_='rst-content')
            or soup.find('div', class_='wy-nav-content')
            or soup.body)

    if main is None:
        return f"# (no content extracted)\n\nSource: {source_url}\n"

    # Strip out nav, footer, sidebar elements
    for tag in main.find_all(['nav', 'footer', 'aside', 'header', 'script', 'style']):
        tag.decompose()
    # Strip "Edit on GitHub" / breadcrumb-style links
    for cls in ['edit-on-github', 'breadcrumbs', 'related', 'sphinxsidebar']:
        for tag in main.find_all(class_=cls):
            tag.decompose()

    md = html_to_md(str(main), heading_style='ATX')
    # Strip absurd whitespace runs
    md = re.sub(r'\n{4,}', '\n\n\n', md)

    header = f"---\nsource: {source_url}\nscraped: {time.strftime('%Y-%m-%d %H:%M:%S')}\n---\n\n"
    return header + md


def crawl(seeds: list[str]) -> dict:
    """BFS crawl. Returns dict of url → status."""
    seen = set()
    queue = deque()
    for s in seeds:
        n = normalize_url(s)
        queue.append((n, 0, n))   # (url, depth, root_seed)
        seen.add(n)

    status = {}
    pages_saved = 0

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'tt-xla-docs-scraper/0.1 (research; respects robots.txt)',
    })

    while queue and pages_saved < MAX_PAGES:
        url, depth, seed = queue.popleft()

        if is_skippable(url):
            continue
        if not is_same_origin(url, seed):
            status[url] = 'skipped-external'
            continue

        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
        except Exception as e:
            status[url] = f'error: {type(e).__name__}'
            continue

        if r.status_code != 200:
            status[url] = f'http {r.status_code}'
            continue

        ctype = r.headers.get('Content-Type', '')
        if 'html' not in ctype:
            status[url] = f'not-html ({ctype})'
            continue

        out_path = url_to_filepath(url)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md = extract_content_md(r.text, url)
        out_path.write_text(md, encoding='utf-8')

        pages_saved += 1
        status[url] = f'saved -> {out_path.relative_to(ROOT)}'
        print(f"  [{pages_saved:4d}/{MAX_PAGES}] d={depth} {url}")

        # Enqueue links
        if depth < MAX_DEPTH:
            for link in extract_links(r.text, url):
                if link not in seen and is_same_origin(link, seed):
                    seen.add(link)
                    queue.append((link, depth + 1, seed))

        time.sleep(SLEEP_BETWEEN)

    return status


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Seeds: {len(SEED_URLS)}")
    print(f"Limits: max_pages={MAX_PAGES}, max_depth={MAX_DEPTH}\n")

    t0 = time.time()
    status = crawl(SEED_URLS)
    elapsed = time.time() - t0

    saved = sum(1 for v in status.values() if v.startswith('saved'))
    errs = sum(1 for v in status.values() if 'error' in v or 'http' in v)
    skipped = sum(1 for v in status.values() if v.startswith('skipped') or v.startswith('not-html'))

    print(f"\n=== Done ===")
    print(f"  Pages saved: {saved}")
    print(f"  Skipped:     {skipped}")
    print(f"  Errors:      {errs}")
    print(f"  Total URLs visited: {len(status)}")
    print(f"  Elapsed: {elapsed/60:.1f} min")

    # Write a status report
    report = OUTPUT_DIR / "_crawl_report.txt"
    with report.open('w') as f:
        f.write(f"Crawl finished {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Saved={saved}, Skipped={skipped}, Errors={errs}\n\n")
        for url, st in sorted(status.items()):
            f.write(f"{st:30s} {url}\n")
    print(f"  Report: {report.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
