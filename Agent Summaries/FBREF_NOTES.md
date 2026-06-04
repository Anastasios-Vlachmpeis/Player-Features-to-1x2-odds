# FBref scraper — parsing notes & maintenance

Supplementary source for advanced metrics (xG, xA, npxG, progressive actions).
**Greek Super League coverage on FBref is patchy — missing data is normal, not
an error.** Rows are written with whatever metrics exist; the rest are NULL.

## Access basics
- `_get_soup()` fetches via `curl_cffi` (impersonate="chrome") with a plain
  `requests` fallback, `BeautifulSoup` parsing, realistic Chrome UA.
- **Conservative 4–6 s random delay** — FBref rate-limits and returns 429 when
  hit too fast. `_get_soup()` backs off 30 s+ on a 429 before retrying.

## ⚠️ IP-level 403 block (the big gotcha — currently blocking us)
FBref/Sports-Reference **blocks many VPN and datacenter IP ranges host-wide**.
When blocked, *every* fbref.com URL (homepage, robots.txt, schedule, match
reports) returns a ~6 KB 403 page across ALL TLS-impersonation profiles
(chrome/chrome124/120/110/safari/edge all tested → all 403). curl_cffi does
**not** help: the block is on the IP, not the TLS handshake.

Diagnosis: `curl_cffi.get("https://fbref.com/robots.txt", impersonate="chrome")`
returns 403 while `https://www.google.com/` returns 200 → you are IP-blocked.

Fix (environmental, not code):
- **Turn the VPN OFF** before running this scraper. This is the *opposite* of the
  Greek odds scrapers (which need VPN ON for geo access), so run FBref in a
  separate session with the VPN disabled.
- Or run from a residential / different network.

> Because of this block, the parsing details below (comp id, table ids,
> data-stat columns) are written from FBref's documented structure but have NOT
> been verified against a live page from this machine. Re-confirm on the first
> successful run from an unblocked IP.

## The comment-wrapped table gotcha
FBref defers loading by wrapping many stat tables inside HTML comments
(`<!-- <table ...> ... </table> -->`). BeautifulSoup won't see them as tables.
`_get_tables(soup)` handles this:
1. Collect all directly-rendered `<table id=...>`.
2. Walk every HTML `Comment` node; if it contains `<table`, re-parse the comment
   string as HTML and collect the tables inside.
3. Return a merged `{table_id: Tag}` dict so callers don't care which is which.

On match-report pages the player stat tables are frequently the commented kind.

## Pages and tables used

| Step | URL / table | Notes |
|---|---|---|
| Fixtures | `…/comps/84/schedule/Greek-Super-League-Scores-and-Fixtures` | No season in path = current season. Schedule table id starts `sched`. Only rows whose `match_report` cell links to text "Match Report" are played; others are skipped. match_id = hash in `/matches/{id}/…`. |
| Per match — summary | table id `stats_{teamId}_summary` | One per team. Player id = hash in `/players/{id}/…`. Team name from `<caption>` ("X Player Stats Table"). |
| Per match — possession | table id `stats_{teamId}_possession` | Merged into the summary rows by fbref_id for the two att-area metrics. |

## Stat column mapping (`data-stat` attribute → our column)

From **stats_{teamId}_summary**:
| Our column | data-stat |
|---|---|
| xg | `xg` |
| xa | `xg_assist`  (this is FBref's xAG) |
| npxg | `npxg` |
| sca | `sca` |
| gca | `gca` |
| progressive_carries | `progressive_carries` |
| progressive_passes | `progressive_passes` |

From **stats_{teamId}_possession**:
| Our column | data-stat |
|---|---|
| touches_att_pen | `touches_att_pen_area` |
| carries_final_third | `carries_into_final_third` |

> Note on `xa`: FBref's per-match player table exposes **xAG** (expected assisted
> goals, `xg_assist`), which is what we store as `xa`. True "xA" isn't in the
> match player tables; xAG is the standard FBref proxy.

Empty cells → `None` (stored NULL). `_f`/`_i` never raise on bad/blank values.

## Maintaining this when FBref changes
- **Comp id wrong / no fixtures?** Open `https://fbref.com/en/comps/`, find
  "Super League Greece", read the number in its URL, update `FBREF_COMP_ID` in
  `fbref_scraper.py` (currently `84`). Verify `COMP_SLUG` matches the URL slug.
- **A metric comes back all-NULL?** FBref renamed a `data-stat`. Open a match
  report, inspect the relevant table cell's `data-stat` attribute, update the
  mapping in `get_match_player_rows()`.
- **Tables not found?** They may have moved between commented/uncommented — but
  `_get_tables()` covers both, so first suspect a table-id rename
  (`stats_{teamId}_summary` / `_possession`).
