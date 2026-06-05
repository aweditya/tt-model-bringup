# Tenstorrent beamer color theme --- audit trail

File: `beamercolorthemetenstorrent.sty`
Drop-in for: `\usecolortheme{tenstorrent}` (replaces `\usecolortheme{stanford}`).
Structure mirrors the upstream gemini colorthemes (e.g. `beamercolorthememit.sty`,
`beamercolorthemelabsix.sty`) at
`https://github.com/anishathalye/gemini/tree/master/colorthemes`.

## How the colors were chosen

Tenstorrent does not publish a public brand-color reference page. Colors were
extracted by saving the rendered HTML of two pages on `tenstorrent.com`
(2026-06-04) and frequency-ranking inline hex codes:

- `https://tenstorrent.com` (homepage)
- `https://tenstorrent.com/hardware/blackhole` (Blackhole product page)

HTML snapshots live under `.cache/tt_home.html`, `.cache/tt_blackhole.html`,
`.cache/tt_news.html`.

### Homepage --- top six hex frequencies

| Count | Hex       | Role on site                                     |
|------:|-----------|--------------------------------------------------|
|  2882 | `#335C4B` | Dark teal-green --- primary brand surface        |
|   511 | `#EEEAE0` | Warm cream --- page background tone              |
|   458 | `#EDBE5A` | Warm gold --- CTAs and accent rules              |
|   456 | `#D9D9D9` | Neutral grey --- ancillary UI                    |
|   407 | `#92C7BF` | Light teal --- secondary tint                    |
|   404 | `#CFEEE8` | Very light teal --- block tint                   |
|   266 | `#1B3527` | Near-black green --- text / deep separators      |

### Blackhole page --- distinctive accent

| Count | Hex       | Role                                              |
|------:|-----------|---------------------------------------------------|
|    33 | `#202020` | Body text / near-black                            |
|     9 | `#3EB7DE` | TT product cyan --- the recurring hardware accent |
|     2 | `#574baa` | Historical Tenstorrent purple (legacy logo era)   |

## Role mapping (`.sty` -> beamer slots)

| Theme color (.sty)  | Hex        | Mapped to                                                                |
|---------------------|-----------:|--------------------------------------------------------------------------|
| `ttdarkteal`        | `#335C4B`  | `structure`, `headline` bg, `block title` fg, `item` fg, headers         |
| `ttnearblack`       | `#1B3527`  | `headline rule`, `block separator`, `palette tertiary` bg, `heading` fg  |
| `ttcream`           | `#EEEAE0`  | `headline` fg (text on dark teal)                                        |
| `ttgold`            | `#EDBE5A`  | `block separator alerted` (warm rule under alert titles)                 |
| `ttcyan`            | `#3EB7DE`  | `block separator example` (the TT product cyan)                          |
| `ttlightgold`       | `#FAF1DE`  | `block title/body alerted` bg --- derived (gold washed toward cream)     |
| `ttlightteal`       | `#CFEEE8`  | `block title/body example` bg --- observed on tenstorrent.com            |
| `ttlightgray`       | `RGB 240`  | Carry-over neutral, not currently bound                                  |

Block bodies stay on plain white so dense poster content (tables, equations,
code listings) keeps maximum legibility. The brand tone shows up in the
headline bar, block-title text, separator rules, and item bullets.

## Discrepancies vs. the original task prompt

The prompt anticipated a deep purple / magenta brand accent (e.g. `#6e2bff`,
`#bd00ff`). That historical Tenstorrent purple is real --- it surfaces on the
Blackhole product page as `#574baa` --- but it is no longer the dominant
brand surface on the live site as of 2026-06-04. The homepage has shifted to
the dark teal `#335C4B` + warm gold `#EDBE5A` + cream `#EEEAE0` palette
above. The theme follows what is actually on the site today rather than the
prior brand. If a future revision wants the legacy purple back, swap in:

```latex
\definecolor{ttpurple}{HTML}{574BAA}     % legacy TT purple (Blackhole page)
```

and rebind `structure`, `headline`, `block title`, `item` to `ttpurple`.

## How to switch back to the Stanford theme

Restore `\usecolortheme{stanford}` in `poster.tex` and ensure
`beamercolorthemestanford.sty` is reachable on the LaTeX search path.

## Verification checklist (do this before committing the rendered PDF)

- [ ] `latexmk -pdf poster.tex` succeeds with no missing-file warnings for
      `beamercolorthemetenstorrent.sty`.
- [ ] Headline bar reads as dark teal with cream title text.
- [ ] Block title rules and bullets are dark teal, not Stanford red.
- [ ] Alert blocks (`\begin{alertblock}`) show a warm-gold separator.
- [ ] Example blocks (`\begin{exampleblock}`) show a cyan separator.
