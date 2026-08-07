# Sofascore JSON Endpoints — Super League 1 scraper

All endpoints are under the base: `https://www.sofascore.com/api/v1`

Sofascore serves stats from these internal JSON APIs (not static HTML), so we
call them directly. **Cloudflare blocks plain `requests` with a 403 based on the
TLS/JA3 fingerprint — not headers** — so we use `curl_cffi` with
`impersonate="chrome"`, which performs a real Chrome TLS handshake and clears
the challenge without a browser. Install: `pip install curl_cffi`.

Headers still include a realistic User-Agent + `Referer`/`Origin` of
`https://www.sofascore.com`. A 2–3 s random delay sits between every call.

`_get_json()` prefers curl_cffi when installed and falls back to plain
`requests` (which will 403) with a warning otherwise.

## Endpoints used

| Purpose | Endpoint | Notes |
|---|---|---|
| List seasons | `/unique-tournament/{tid}/seasons` | `tid = 185` (Greek Super League 1). Response `seasons[]` is newest-first → `[0]` is current season. |
| Finished matches (paged) | `/unique-tournament/{tid}/season/{seasonId}/events/last/{page}` | `last` = already-played. `page` starts at 0; keep paging while `hasNextPage` is true. Filter on `status.type == "finished"`. |
| Match lineups + player stats | `/event/{matchId}/lineups` | Per-player stats live inline at `home.players[].statistics` and `away.players[].statistics`. `substitute: true` = bench start. |
| Match shotmap (xG/xGOT) | `/event/{matchId}/shotmap` | Per-shot array under `shotmap` (older payloads: `shots`). Used by `scrape_sofascore_xg.py` → `sofascore_xg` table. |

> Team names (home/away) come from the **events** payload (`homeTeam.name` /
> `awayTeam.name`), not the lineups payload. Match date = `startTimestamp`
> (UNIX seconds, UTC).

## Shotmap → sofascore_xg (xG / xGOT aggregation)

`get_match_xg_rows()` calls `/event/{matchId}/shotmap` and aggregates per shooter
to one row per player per match. Per-shot fields used:

| Field | Meaning | Our use |
|---|---|---|
| `player.id` | shooter's Sofascore id | `sofascore_id` (group key) |
| `player.name` | shooter name | (not stored here; name lives in `sofascore_players`) |
| `xg` | expected goals for the shot (float) | summed → `xg` |
| `xgot` | expected goals on target; **0 for off-target shots** | summed → `xgot` |
| `isHome` | true = home side | maps to event home/away name → `player_team` |

`shots` = count of shot objects for that player. Sums rounded to 4 dp.

**Validated (2025-09-27, Olympiacos vs Levadiakos, match 14151968):** 25/25 shots
had non-null numeric `xg` and present `xgot` for this Greek match — so Sofascore
is a valid xG source for the Greek SL and FotMob was NOT needed.

Behaviour / edge cases:
- Match with no shotmap → `get_match_xg_rows()` returns `None` → logged "missing
  shotmap", nothing written, run continues. (Distinct from an empty shot array.)
- A player who took no shots simply has no row — correct, not an error.
- `xgot == 0` is a real value (off-target shot), not missing data.
- If a future payload drops `xg` on a shot, that shot still counts toward `shots`
  but contributes 0 to the xG/xGOT sums (guarded in code).

Maintenance: if xG comes back all-null someday, re-run the one-match validation
above; if Greek coverage regressed, the fallback plan is FotMob (league id 135).

## Statistics field mapping (lineups → our schema)

| Our column | Sofascore key |
|---|---|
| rating | `rating` |
| minutes_played | `minutesPlayed` |
| goals | `goals` |
| assists | `goalAssist` |
| key_passes | `keyPass` |
| total_passes | `totalPass` |
| accurate_passes | `accuratePass` |
| tackles | `tackles` |
| interceptions | `interceptionWon` |
| clearances | `totalClearance` |
| aerial_won | `aerialWon` |
| aerial_total | `aerialWon` + `aerialLost` |
| is_starter | `not substitute` |

**Important:** Sofascore omits a stat key entirely when its value is 0, so
`_stat()` defaults missing keys to 0. `rating` is left as `None` when absent
(a player with no rating genuinely has none, e.g. very short cameos).

## Maintaining this when Sofascore changes

- **Tournament re-IDed?** Hit `/search/all?q=super%20league` (or the site
  search) and read the `uniqueTournament.id` for Greece. Update
  `UNIQUE_TOURNAMENT_ID` in `sofascore_scraper.py`.
- **Stat key renamed?** Inspect a live `/event/{id}/lineups` response in the
  browser Network tab and update the mapping in `_extract_side()`.
- **403 / Cloudflare block?** First confirm `curl_cffi` is installed — without
  it the code falls back to plain `requests`, which Cloudflare always 403s.
  If curl_cffi itself starts getting blocked, bump the `_IMPERSONATE` target in
  `sofascore_scraper.py` to a newer Chrome build (e.g. `"chrome124"`). As a last
  resort escalate to `undetected_chromedriver`: drive headless Chrome to the same
  API URLs (they return raw JSON in the page body) and parse `driver.page_source`.
  Only `_get_json()` needs to change — the rest of the logic is unaffected.
