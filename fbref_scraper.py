"""
FBref scraping logic for Greek Super League 1 advanced per-match metrics.

Supplementary source (Layer 2+): xG / xA / npxG / SCA / GCA / progressive
actions. FBref's Greek coverage is patchy — MISSING DATA IS THE NORMAL CASE,
never an error. We capture what exists and leave the rest NULL.

Two FBref scraping quirks handled here:
  1. Many stat tables are wrapped inside HTML comments (<!-- ... -->) to defer
     rendering. _get_tables() strips the comment markers and parses the inner
     HTML so commented + uncommented tables look identical to the caller.
  2. FBref rate-limits aggressively — we use a conservative 4-6 s delay.

Public functions consumed by scrape_fbref.py:
  get_season_fixtures(date_from, date_to) -> list[dict]   (matches w/ reports)
  get_match_player_rows(fixture)          -> list[dict]    (one row per player)

See FBREF_NOTES.md for the exact tables / data-stat columns pulled.
"""

import re
import time
import random
import logging
from datetime import datetime, date
from typing import Optional

import requests
from bs4 import BeautifulSoup, Comment

# FBref (like Sofascore) blocks plain `requests` with a 403 on TLS/JA3
# fingerprint, not headers. curl_cffi impersonates a real Chrome handshake and
# clears it without a browser. Prefer it when installed; fall back to requests.
try:
    from curl_cffi import requests as cffi_requests
    _USE_CFFI = True
except ImportError:  # pragma: no cover
    cffi_requests = None
    _USE_CFFI = False

log = logging.getLogger(__name__)

BASE_URL = "https://fbref.com"
_IMPERSONATE = "chrome"

# Greek Super League competition id on FBref.
# FRAGILE: verify by opening https://fbref.com/en/comps/ and finding "Super
# League Greece" — the number in its URL is this id. Update if it ever changes.
FBREF_COMP_ID = "84"
COMP_SLUG = "Greek-Super-League"

# No season in the path = current season's schedule.
SCHEDULE_URL = (
    f"{BASE_URL}/en/comps/{FBREF_COMP_ID}/schedule/"
    f"{COMP_SLUG}-Scores-and-Fixtures"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://fbref.com/",
}

if _USE_CFFI:
    _SESSION = cffi_requests.Session(impersonate=_IMPERSONATE)
    _SESSION.headers.update(_HEADERS)
else:
    _SESSION = requests.Session()
    _SESSION.headers.update(_HEADERS)
    log.warning(
        "curl_cffi not installed — falling back to requests, which FBref will "
        "likely 403. Install with: pip install curl_cffi"
    )


# ---------------------------------------------------------------------------
# HTTP helper — conservative delay, FBref blocks fast scraping
# ---------------------------------------------------------------------------

def _get_soup(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    time.sleep(random.uniform(4.0, 6.0))
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=30)
            if resp.status_code == 404:
                log.debug("404 for %s", url)
                return None
            if resp.status_code == 429:
                # Rate limited — back off hard before retrying
                wait = 30 * (attempt + 1)
                log.warning("429 rate limit on %s — backing off %ds", url, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as exc:  # curl_cffi & requests raise different types
            log.warning("Attempt %d failed for %s: %s", attempt + 1, url, exc)
            if attempt < retries - 1:
                time.sleep(random.uniform(6.0, 12.0))
    return None


# ---------------------------------------------------------------------------
# Comment-wrapped table handling
# ---------------------------------------------------------------------------

def _get_tables(soup: BeautifulSoup) -> dict:
    """Return {table_id: Tag} for every stat table on the page.

    FBref hides many tables inside HTML comments to defer load. We collect both
    the directly-rendered tables and the ones buried in comments, so callers
    never have to care which is which.
    """
    tables = {}

    # Directly rendered tables
    for tbl in soup.find_all("table"):
        if tbl.get("id"):
            tables[tbl["id"]] = tbl

    # Tables inside HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" not in comment:
            continue
        inner = BeautifulSoup(comment, "html.parser")
        for tbl in inner.find_all("table"):
            if tbl.get("id"):
                tables.setdefault(tbl["id"], tbl)

    return tables


# ---------------------------------------------------------------------------
# Value parsers — empty / missing cells become None, not zero
# ---------------------------------------------------------------------------

def _cell(row, data_stat: str):
    """Return the stripped text of a cell by its data-stat, or None if absent."""
    el = row.find(["td", "th"], attrs={"data-stat": data_stat})
    if el is None:
        return None
    text = el.get_text(strip=True)
    return text if text != "" else None


def _f(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _i(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value.replace(",", ""))
    except (TypeError, ValueError, AttributeError):
        return None


def _fbref_id_from_href(href: str) -> Optional[str]:
    m = re.search(r"/players/([a-z0-9]+)/", href or "")
    return m.group(1) if m else None


def _match_id_from_href(href: str) -> Optional[str]:
    m = re.search(r"/matches/([a-z0-9]+)/", href or "")
    return m.group(1) if m else None


def _parse_date_arg(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def get_season_fixtures(date_from=None, date_to=None) -> list:
    """Return current-season fixtures that have a Match Report (i.e. played).

    Only matches with a "Match Report" link carry player stats; scheduled
    matches link to "Head-to-Head" instead and are skipped.
    """
    d_from = _parse_date_arg(date_from)
    d_to = _parse_date_arg(date_to)

    soup = _get_soup(SCHEDULE_URL)
    if soup is None:
        raise RuntimeError(f"Could not fetch FBref schedule: {SCHEDULE_URL}")

    tables = _get_tables(soup)
    # The fixtures table id varies by view: 'sched_{season}_{comp}_1' on some
    # pages, 'results{season}{comp}1_overall' on the schedule view. Rather than
    # match a fragile id, pick whichever table actually has a match_report column.
    sched = None
    for tid, t in tables.items():
        if t.find(attrs={"data-stat": "match_report"}):
            sched = t
            break
    if sched is None:
        raise RuntimeError("Could not locate the schedule table on FBref")

    fixtures = []
    for row in sched.select("tbody tr"):
        if "thead" in (row.get("class") or []):
            continue

        report_cell = row.find("td", attrs={"data-stat": "match_report"})
        if report_cell is None:
            continue
        link = report_cell.find("a")
        # Played matches say "Match Report"; everything else is not yet playable
        if link is None or link.get_text(strip=True) != "Match Report":
            continue

        match_id = _match_id_from_href(link.get("href", ""))
        if match_id is None:
            continue

        date_text = _cell(row, "date")
        ev_date = None
        if date_text:
            try:
                ev_date = datetime.strptime(date_text, "%Y-%m-%d").date()
            except ValueError:
                ev_date = None

        if d_from and ev_date and ev_date < d_from:
            continue
        if d_to and ev_date and ev_date > d_to:
            continue

        fixtures.append(
            {
                "match_id": match_id,
                "match_date": ev_date.isoformat() if ev_date else date_text,
                "home_team": _cell(row, "home_team"),
                "away_team": _cell(row, "away_team"),
                "report_url": BASE_URL + link["href"],
            }
        )

    log.info("Found %d played fixtures with match reports", len(fixtures))
    return fixtures


# ---------------------------------------------------------------------------
# Per-match player rows
# ---------------------------------------------------------------------------

def _caption_team(table) -> Optional[str]:
    """Team name from a stats table caption ('Olympiacos Player Stats Table')."""
    cap = table.find("caption")
    if cap is None:
        return None
    text = cap.get_text(strip=True)
    return text.replace("Player Stats Table", "").strip() or None


def _match_team_name(caption_name, fixture) -> Optional[str]:
    """Map a caption team name to the home/away name from the schedule.

    FBref captions and schedule names usually agree, but we fuzzy-match on a
    substring in case of minor punctuation differences, falling back to the
    caption text itself.
    """
    if not caption_name:
        return None
    for side in (fixture["home_team"], fixture["away_team"]):
        if not side:
            continue
        if caption_name == side or caption_name in side or side in caption_name:
            return side
    return caption_name


def get_match_player_rows(fixture: dict) -> dict:
    """Scrape one match report. Returns {'rows': [...], 'with_xg': int}.

    One row per player, merging the summary table (xG/xA/npxG/SCA/GCA/progressive
    passes+carries) with the possession table (touches in att. pen area, carries
    into final third). Any metric a table doesn't provide stays None.
    """
    soup = _get_soup(fixture["report_url"])
    if soup is None:
        log.warning("Could not fetch match report %s", fixture["match_id"])
        return {"rows": [], "with_xg": 0}

    tables = _get_tables(soup)

    # Player-level tables are named stats_{teamid}_{type}. Group team ids.
    team_ids = set()
    for tid in tables:
        m = re.match(r"stats_([a-z0-9]+)_summary$", tid)
        if m:
            team_ids.add(m.group(1))

    if not team_ids:
        log.warning("No player stat tables on match report %s",
                    fixture["match_id"])
        return {"rows": [], "with_xg": 0}

    rows = []
    with_xg = 0

    for team_id in team_ids:
        summary = tables.get(f"stats_{team_id}_summary")
        possession = tables.get(f"stats_{team_id}_possession")
        team_name = _match_team_name(_caption_team(summary), fixture)

        # Index possession rows by fbref_id for merging
        poss_by_id = {}
        if possession is not None:
            for prow in possession.select("tbody tr"):
                link = prow.find("a", href=re.compile(r"/players/"))
                if link is None:
                    continue
                pid = _fbref_id_from_href(link.get("href", ""))
                if pid:
                    poss_by_id[pid] = prow

        for srow in summary.select("tbody tr"):
            link = srow.find("a", href=re.compile(r"/players/"))
            if link is None:
                continue  # skip team-total / spacer rows
            fbref_id = _fbref_id_from_href(link.get("href", ""))
            if fbref_id is None:
                continue

            player_name = link.get_text(strip=True)
            prow = poss_by_id.get(fbref_id)

            xg = _f(_cell(srow, "xg"))
            if xg is not None:
                with_xg += 1

            rows.append(
                {
                    "fbref_id": fbref_id,
                    "player_name": player_name,
                    "match_id": fixture["match_id"],
                    "match_date": fixture["match_date"],
                    "home_team": fixture["home_team"],
                    "away_team": fixture["away_team"],
                    "player_team": team_name,
                    "xg": xg,
                    "xa": _f(_cell(srow, "xg_assist")),       # FBref xAG
                    "npxg": _f(_cell(srow, "npxg")),
                    "sca": _f(_cell(srow, "sca")),
                    "gca": _f(_cell(srow, "gca")),
                    "progressive_carries": _i(_cell(srow, "progressive_carries")),
                    "progressive_passes": _i(_cell(srow, "progressive_passes")),
                    # These two only exist in the possession table
                    "touches_att_pen": _i(
                        _cell(prow, "touches_att_pen_area") if prow else None
                    ),
                    "carries_final_third": _i(
                        _cell(prow, "carries_into_final_third") if prow else None
                    ),
                }
            )

    return {"rows": rows, "with_xg": with_xg}
