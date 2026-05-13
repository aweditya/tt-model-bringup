# TT ecosystem scrapers

Two scripts live in this directory:

- `scrape_tt_docs.py` — original single-threaded docs crawler (left in place; still works).
- `scrape_tt_ecosystem.py` — newer scraper covering both `docs.tenstorrent.com`
  and Tenstorrent GitHub repos, with concurrency, CLI flags, and idempotent output.

Use the ecosystem scraper for new work.

## Phases

### Phase 1 — `docs`

BFS-walks `docs.tenstorrent.com` starting from the documented seed URLs
(root, `tt-metal/latest/ttnn/`, `tt-metal/latest/tt_metal/`). For every
HTML page it:

1. Strips nav / footer / sidebar / scripts.
2. Converts the main `<article>` / `<main>` element to markdown.
3. Writes `tt_docs_corpus/docs.tenstorrent.com/<url-path>.md`,
   overwriting any prior copy (idempotent refresh).

Skips binary assets, `_sources/`, `_static/`, search and genindex pages.
Stays strictly within the `docs.tenstorrent.com` domain.

A summary report is written to `tt_docs_corpus/_crawl_report.txt`.

### Phase 2 — `github`

For each repo in the configured list it pulls, via the GitHub REST API:

- **Open + recently closed issues** (`issues.jsonl`) — title, body, labels, state, dates.
- **Open + recently merged PRs** (`pulls.jsonl`) — same plus base/head ref and merge metadata.
- **Branch names** (`branches.jsonl`) — name, commit sha, protected flag.

"Recently closed/merged" defaults to a 180-day lookback. Open items are
always included regardless of age. Output lives at
`tt_docs_corpus/github/<repo>/`. Files are overwritten in place each run.

Default repos:

```
tenstorrent/tt-metal
tenstorrent/tt-mlir
tenstorrent/tt-forge-fe
tenstorrent/tt-xla
tenstorrent/tt-torch
tenstorrent/tt-blacksmith
```

## Install

```
pip install --user --break-system-packages requests beautifulsoup4 markdownify
```

(Same deps as the original `scrape_tt_docs.py`.)

## Running

```
# Both phases, defaults
python3 pjrt_plugin/scripts/scrape_tt_ecosystem.py

# Just docs, capped at 50 pages for a quick smoke test
python3 pjrt_plugin/scripts/scrape_tt_ecosystem.py --phase docs --max-pages 50

# Just GitHub
python3 pjrt_plugin/scripts/scrape_tt_ecosystem.py --phase github

# Override repo list
python3 pjrt_plugin/scripts/scrape_tt_ecosystem.py --phase github \
    --repos tenstorrent/tt-metal tenstorrent/tt-mlir

# Verbose logging
python3 pjrt_plugin/scripts/scrape_tt_ecosystem.py -v
```

## Output layout

```
tt_docs_corpus/
  _crawl_report.txt                       # docs phase summary
  docs.tenstorrent.com/
    index.md
    tt-metal/latest/ttnn/...
    ...
  github/
    tt-metal/
      issues.jsonl
      pulls.jsonl
      branches.jsonl
    tt-mlir/
      ...
```

The entire `tt_docs_corpus/` tree is git-ignored — it is regenerable
from these scripts.

## Rate limits

- **docs**: a small per-batch sleep is built in. No documented public
  limit, but we stay polite by default (~6 concurrent requests).
- **GitHub**:
  - Anonymous: **60 requests/hour**. The script will refuse to retry
    once a 403 with `rate limit` is seen and will leave whatever it has
    fetched on disk.
  - With `GITHUB_TOKEN` exported: **5000 requests/hour**, plenty for the
    full repo list in one run.
  - Generate a fine-grained token with public-repo read access:
    https://github.com/settings/tokens — then `export GITHUB_TOKEN=...`.

## Adding new repos

Either pass `--repos owner/repo ...` on the CLI, or edit the
`GITHUB_REPOS` constant near the top of `scrape_tt_ecosystem.py` to
make the addition permanent.

## Adding new doc seed URLs

Edit `DOCS_SEEDS` at the top of the script. Any same-domain link
discovered during the BFS is automatically enqueued, so adding a seed
only matters if it lives in a corner of the site not linked from the
existing seeds.

## Notes / caveats

- The script uses a shared `requests.Session` across threads. This is
  fine for the GET-only workload but should not be cargo-culted into
  POST/PUT code.
- Output is overwritten per run, not incrementally merged. If you need
  an append-style history of closed issues, run periodically and snapshot.
- The script does **not** touch the Tenstorrent device — it is purely
  local. Per CLAUDE.md, device code stays on `qb1`/`qb2`.
