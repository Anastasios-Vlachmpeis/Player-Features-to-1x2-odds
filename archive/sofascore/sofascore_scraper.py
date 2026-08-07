"""
Sofascore scraping logic for Greek Super League 1 per-match player stats.

Layer 2 (current form) + part of Layer 3 (physical state via minutes played).

Sofascore renders stats from internal JSON API endpoints — we call those
directly with requests rather than scraping the DOM.  See ENDPOINTS.md for the
exact endpoint structure and how to maintain it when Sofascore changes the API.

Public functions consumed by scrape_sofascore.py:
  get_current_season_id()                 -> int
  get_season_events(season_id, date_range)-> list[dict]   (finished matches)
  get_match_player_rows(event)            -> list[dict]    (one row per player)
"""

import time
import random
import logging
import re
from datetime import datetime, date
from typing import Optional

import requests

# Sofascore sits behind Cloudflare, which blocks plain `requests` on TLS/JA3
# fingerprint (not headers) — every call returns 403. curl_cffi impersonates a
# real Chrome TLS handshake and gets through without a full browser. We prefer
# it when installed and fall back to requests otherwise.
try:
    from curl_cffi import requests as cffi_requests
    _USE_CFFI = True
except ImportError:  # pragma: no cover
    cffi_requests = None
    _USE_CFFI = False

log = logging.getLogger(__name__)

API_BASE = "https://www.sofascore.com/api/v1"

# curl_cffi impersonation target — keep roughly in step with a current Chrome.
_IMPERSONATE = "chrome"

# Greek Super League 1 unique-tournament id on Sofascore.
# Fragile: verify via /api/v1/search if Sofascore ever re-IDs the competition.
UNIQUE_TOURNAMENT_ID = 185

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}

if _USE_CFFI:
    _SESSION = cffi_requests.Session(impersonate=_IMPERSONATE)
    _SESSION.headers.update(_HEADERS)
else:
    _SESSION = requests.Session()
    _SESSION.headers.update(_HEADERS)
    log.warning(
        "curl_cffi not installed — falling back to requests, which Cloudflare "
        "will likely 403. Install with: pip install curl_cffi"
    )


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get_json(url: str, retries: int = 3) -> Optional[dict]:
    """GET a JSON endpoint with polite delay + retry. Returns None on failure.

    Uses curl_cffi (Chrome TLS impersonation) to clear Cloudflare. If Sofascore
    ever hardens further, escalate to undetected_chromedriver — see
    SOFASCORE_ENDPOINTS.md fallback note.
    """
    time.sleep(random.uniform(2.0, 3.0))
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=20)
            if resp.status_code == 404:
                # Some matches legitimately have no lineups endpoint yet
                log.debug("404 for %s", url)
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # curl_cffi & requests raise different types
            log.warning("Attempt %d failed for %s: %s", attempt + 1, url, exc)
            if attempt < retries - 1:
                time.sleep(random.uniform(3.0, 6.0))
    return None


# ---------------------------------------------------------------------------
# Season + event discovery
# ---------------------------------------------------------------------------

def get_seasons() -> list[dict]:
    """Return the Sofascore season catalogue for Super League 1."""
    url = f"{API_BASE}/unique-tournament/{UNIQUE_TOURNAMENT_ID}/seasons"
    data = _get_json(url)
    if not data or not data.get("seasons"):
        raise RuntimeError("Could not fetch seasons list from Sofascore")
    return data["seasons"]


def _season_years(season: dict) -> tuple[int, int] | None:
    """Extract (start year, end year) from names such as ``2025/2026``."""
    text = " ".join(
        str(season.get(key, "")) for key in ("name", "year")
    )
    match = re.search(r"(20\d{2})\D+(20\d{2})", text)
    if match:
        return int(match.group(1)), int(match.group(2))

    short = re.search(r"\b(\d{2})\s*[/\-]\s*(\d{2})\b", text)
    if short:
        start = 2000 + int(short.group(1))
        end = 2000 + int(short.group(2))
        return start, end
    return None


def select_seasons(
    seasons: list[dict], start_year: int, end_year: int
) -> list[dict]:
    """Select seasons fully contained in a calendar-year interval.

    ``start_year=2015, end_year=2026`` selects 2015/16 through 2025/26.
    This function is deliberately network-free so the selection can be tested.
    """
    if start_year >= end_year:
        raise ValueError("start_year must be earlier than end_year")

    selected = []
    for season in seasons:
        years = _season_years(season)
        if years and years[0] >= start_year and years[1] <= end_year:
            selected.append({**season, "start_year": years[0], "end_year": years[1]})
    return sorted(selected, key=lambda item: item["start_year"])


def get_seasons_between(start_year: int, end_year: int) -> list[dict]:
    """Fetch and select seasons from ``start_year`` through ``end_year``."""
    selected = select_seasons(get_seasons(), start_year, end_year)
    if not selected:
        raise RuntimeError(
            f"No Sofascore seasons found between {start_year} and {end_year}"
        )
    log.info(
        "Selected %d seasons: %s",
        len(selected),
        ", ".join(str(season.get("name")) for season in selected),
    )
    return selected


def get_current_season_id() -> int:
    """Return the most recent season id for Super League 1.

    Endpoint: /unique-tournament/{id}/seasons
    The seasons list is newest-first, so element [0] is the current season.
    """
    season = get_seasons()[0]
    log.info("Current season: %s (id=%s)", season.get("name"), season["id"])
    return season["id"]


def _parse_date_arg(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def get_season_events(
    season_id: int,
    date_from=None,
    date_to=None,
    *,
    season_name: str | None = None,
) -> list:
    """Return finished events for the season, optionally filtered by date range.

    Endpoint: /unique-tournament/{id}/season/{seasonId}/events/last/{page}
    'last' = already-played matches, paginated. We page until exhausted.

    date_from / date_to: 'YYYY-MM-DD' strings (or date objects) to restrict the
    range — lets you re-scrape only recent matches without pulling the season.
    """
    d_from = _parse_date_arg(date_from)
    d_to = _parse_date_arg(date_to)

    events = []
    page = 0
    while True:
        url = (
            f"{API_BASE}/unique-tournament/{UNIQUE_TOURNAMENT_ID}"
            f"/season/{season_id}/events/last/{page}"
        )
        data = _get_json(url)
        if not data or not data.get("events"):
            break

        for ev in data["events"]:
            # Only finished matches carry complete player stats
            status = ev.get("status", {}).get("type")
            if status != "finished":
                continue

            ts = ev.get("startTimestamp")
            ev_date = datetime.utcfromtimestamp(ts).date() if ts else None

            if d_from and ev_date and ev_date < d_from:
                continue
            if d_to and ev_date and ev_date > d_to:
                continue

            events.append(
                {
                    "match_id": ev["id"],
                    "season_id": season_id,
                    "season_name": season_name,
                    "match_date": ev_date.isoformat() if ev_date else None,
                    "home_team": ev.get("homeTeam", {}).get("name"),
                    "away_team": ev.get("awayTeam", {}).get("name"),
                    "home_score": ev.get("homeScore", {}).get("current"),
                    "away_score": ev.get("awayScore", {}).get("current"),
                }
            )

        # Sofascore signals the last page via hasNextPage in the meta block
        if not data.get("hasNextPage"):
            break
        page += 1

    log.info("Collected %d finished events for season %s", len(events), season_id)
    return events


# ---------------------------------------------------------------------------
# Per-match player stats
# ---------------------------------------------------------------------------

def _stat(stats: dict, key: str, default=0):
    """Safe getter — many stat keys are absent when the value is zero."""
    val = stats.get(key)
    return default if val is None else val


def _extract_side(side: dict, team_name: str, event: dict) -> list:
    """Build player rows for one side (home or away) of a match."""
    rows = []
    for entry in side.get("players", []):
        player = entry.get("player", {})
        stats = entry.get("statistics") or {}

        # Players who didn't feature have no statistics block — skip them
        if not stats:
            continue

        sofascore_id = player.get("id")
        if sofascore_id is None:
            continue

        aerial_won = _stat(stats, "aerialWon")
        aerial_lost = _stat(stats, "aerialLost")

        rows.append(
            {
                "sofascore_id": sofascore_id,
                "player_name": player.get("name"),
                "match_id": event["match_id"],
                "match_date": event["match_date"],
                "home_team": event["home_team"],
                "away_team": event["away_team"],
                "player_team": team_name,
                "rating": stats.get("rating"),
                "minutes_played": _stat(stats, "minutesPlayed"),
                "goals": _stat(stats, "goals"),
                "assists": _stat(stats, "goalAssist"),
                "key_passes": _stat(stats, "keyPass"),
                "total_passes": _stat(stats, "totalPass"),
                "accurate_passes": _stat(stats, "accuratePass"),
                "tackles": _stat(stats, "tackles"),
                "interceptions": _stat(stats, "interceptionWon"),
                "clearances": _stat(stats, "totalClearance"),
                "aerial_won": aerial_won,
                "aerial_total": aerial_won + aerial_lost,
                # substitute=True means they started on the bench
                "is_starter": not entry.get("substitute", False),
            }
        )
    return rows


def get_match_player_rows(event: dict) -> list:
    """Fetch lineups for one event and return one row per player who appeared.

    Endpoint: /event/{match_id}/lineups
    The lineups payload carries each player's per-match statistics inline under
    home.players[].statistics and away.players[].statistics.
    """
    url = f"{API_BASE}/event/{event['match_id']}/lineups"
    data = _get_json(url)
    if not data:
        log.warning("No lineups for match %s", event["match_id"])
        return []

    # Team names come from the event payload, not the lineups payload
    rows = []
    if "home" in data:
        rows += _extract_side(data["home"], event["home_team"], event)
    if "away" in data:
        rows += _extract_side(data["away"], event["away_team"], event)

    return rows


# ---------------------------------------------------------------------------
# Per-match xG / xGOT — aggregated from the shotmap endpoint
# ---------------------------------------------------------------------------

def get_match_xg_rows(event: dict) -> Optional[list]:
    """Fetch the shotmap for one event and aggregate xG/xGOT per player.

    Endpoint: /event/{match_id}/shotmap
    Each shot object carries:
      - player.id / player.name : who took the shot
      - xg                       : expected goals for that shot (float)
      - xgot                     : expected goals on target (0 for off-target)
      - isHome                   : True => home side, used for team attribution
    We sum xg and xgot and count shots per player.

    Returns:
      list[dict] of per-player aggregates (one row per player who shot), or
      None if the match has no shotmap (distinct from an empty match) so the
      caller can log "missing shotmap" vs "shotmap present but empty".
    """
    url = f"{API_BASE}/event/{event['match_id']}/shotmap"
    data = _get_json(url)
    if not data:
        return None

    # Sofascore returns the shots under "shotmap" (older payloads: "shots")
    shots = data.get("shotmap")
    if shots is None:
        shots = data.get("shots")
    if shots is None:
        return None  # endpoint responded but carries no shot array

    agg = {}  # sofascore_id -> running totals
    for shot in shots:
        player = shot.get("player") or {}
        sofascore_id = player.get("id")
        if sofascore_id is None:
            continue

        xg = shot.get("xg")
        xgot = shot.get("xgot")
        # xg should be present for Greek matches (validated), but guard anyway:
        # a shot with no xg still counts as a shot, contributing 0 to the sums.
        xg_val = float(xg) if isinstance(xg, (int, float)) else 0.0
        xgot_val = float(xgot) if isinstance(xgot, (int, float)) else 0.0

        # Shot on target = a goal or a saved shot. Blocked/miss/post are NOT on
        # target. shotType is Sofascore's own classification (goal/save/block/
        # miss/post), so we read it directly rather than inferring from xgot.
        is_on_target = 1 if shot.get("shotType") in ("goal", "save") else 0

        # isHome on the shot maps to the event's home/away team name
        team = (
            event["home_team"] if shot.get("isHome") else event["away_team"]
        )

        rec = agg.get(sofascore_id)
        if rec is None:
            agg[sofascore_id] = {
                "sofascore_id": sofascore_id,
                "match_id": event["match_id"],
                "match_date": event["match_date"],
                "player_team": team,
                "xg": xg_val,
                "xgot": xgot_val,
                "shots": 1,
                "sot": is_on_target,
            }
        else:
            rec["xg"] += xg_val
            rec["xgot"] += xgot_val
            rec["shots"] += 1
            rec["sot"] += is_on_target

    # Round the sums to avoid long float tails in the DB
    rows = list(agg.values())
    for r in rows:
        r["xg"] = round(r["xg"], 4)
        r["xgot"] = round(r["xgot"], 4)

    return rows
