"""Historical top-division results and 1X2 odds from Football-Data.

The network-facing function is intentionally separate from CSV normalization so
the parser can be tested without downloading anything.
"""

from __future__ import annotations

import csv
import io
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://www.football-data.co.uk/mmz4281"
SOURCE = "football-data.co.uk"

# Football-Data division codes for target non-Big-5 top divisions.
LEAGUES: dict[str, dict[str, str]] = {
    "greece": {"division": "G1", "name": "Greece Super League"},
    "turkey": {"division": "T1", "name": "Turkey Super Lig"},
    "netherlands": {"division": "N1", "name": "Netherlands Eredivisie"},
    "portugal": {"division": "P1", "name": "Portugal Primeira Liga"},
    "belgium": {"division": "B1", "name": "Belgium Pro League"},
    "scotland": {"division": "SC0", "name": "Scotland Premiership"},
}

DEFAULT_LEAGUE_KEYS: tuple[str, ...] = tuple(LEAGUES)


def resolve_league_keys(keys: list[str] | None = None) -> list[str]:
    """Validate league keys and return them in registry order."""
    selected = list(keys) if keys else list(DEFAULT_LEAGUE_KEYS)
    unknown = [key for key in selected if key not in LEAGUES]
    if unknown:
        known = ", ".join(DEFAULT_LEAGUE_KEYS)
        raise ValueError(f"Unknown league key(s): {', '.join(unknown)}. Known: {known}")
    # Preserve registry order regardless of CLI ordering.
    return [key for key in DEFAULT_LEAGUE_KEYS if key in selected]


def league_division(league_key: str) -> str:
    return LEAGUES[league_key]["division"]

# Prefer a market-average close. Older files may only expose Pinnacle closing
# or pre-closing prices. The selected type is always retained in odds_source and
# odds_is_closing; downstream evaluation must filter odds_is_closing = 1.
ODDS_CANDIDATES = (
    ("market_average_closing", ("AvgCH", "AvgCD", "AvgCA"), True),
    ("bet365_closing", ("B365CH", "B365CD", "B365CA"), True),
    ("pinnacle_closing", ("PSCH", "PSCD", "PSCA"), True),
    ("market_average_preclosing", ("AvgH", "AvgD", "AvgA"), False),
    ("bet365_preclosing", ("B365H", "B365D", "B365A"), False),
    ("pinnacle_preclosing", ("PSH", "PSD", "PSA"), False),
)


def season_code(start_year: int, end_year: int) -> str:
    """Return Football-Data's compact code, e.g. 2015/16 -> ``1516``."""
    if end_year != start_year + 1:
        raise ValueError("A football season must span consecutive calendar years")
    if not (2000 <= start_year <= 2098):
        raise ValueError("Season years must be in the 2000s")
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def season_label(start_year: int, end_year: int) -> str:
    season_code(start_year, end_year)
    return f"{start_year}-{end_year % 100:02d}"


def season_url(start_year: int, end_year: int, division: str) -> str:
    return f"{BASE_URL}/{season_code(start_year, end_year)}/{division}.csv"


def iter_seasons(start_year: int, end_year: int):
    """Yield seasons fully inside a calendar range.

    ``iter_seasons(2015, 2026)`` yields 2015/16 through 2025/26.
    """
    if start_year >= end_year:
        raise ValueError("start_year must be earlier than end_year")
    for season_start in range(start_year, end_year):
        yield season_start, season_start + 1


def fetch_season_csv(
    start_year: int,
    end_year: int,
    division: str,
    *,
    retries: int = 3,
) -> tuple[str, str]:
    """Download one season and return ``(decoded_csv, source_url)``."""
    url = season_url(start_year, end_year, division)
    headers = {
        "User-Agent": "SuperLeagueResearch/1.0 (historical academic download)",
        "Accept": "text/csv,text/plain,*/*",
    }
    last_error = None
    for attempt in range(retries):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=30) as response:
                content = response.read()
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = content.decode("cp1252")
            return text, url
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def _parse_date(value: str) -> str:
    value = value.strip()
    for pattern in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Unsupported match date: {value!r}")


def _integer(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _positive_float(value: str | None) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 1.0 else None


def _select_odds(row: dict) -> tuple[str | None, bool, tuple[float, float, float] | None]:
    for name, columns, is_closing in ODDS_CANDIDATES:
        values = tuple(_positive_float(row.get(column)) for column in columns)
        if all(value is not None for value in values):
            return name, is_closing, values
    return None, False, None


def _no_vig_probabilities(odds: tuple[float, float, float]):
    inverse = [1.0 / value for value in odds]
    total = sum(inverse)
    return tuple(value / total for value in inverse)


def parse_season_csv(
    csv_text: str,
    *,
    start_year: int,
    end_year: int,
    division: str,
    source_url: str | None = None,
) -> list[dict]:
    """Normalize official scores and the best available 1X2 odds per match."""
    label = season_label(start_year, end_year)
    url = source_url or season_url(start_year, end_year, division)
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    rows = []

    for raw in reader:
        row = {(key or "").strip(): value for key, value in raw.items()}
        home = (row.get("HomeTeam") or "").strip()
        away = (row.get("AwayTeam") or "").strip()
        raw_date = (row.get("Date") or "").strip()
        home_goals = _integer(row.get("FTHG") or row.get("HG"))
        away_goals = _integer(row.get("FTAG") or row.get("AG"))
        if not home or not away or not raw_date:
            continue
        if home_goals is None or away_goals is None:
            continue

        derived_result = (
            "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
        )
        supplied_result = (row.get("FTR") or row.get("Res") or "").strip().upper()
        if supplied_result and supplied_result != derived_result:
            raise ValueError(
                f"Result mismatch for {home} vs {away} on {raw_date}: "
                f"goals imply {derived_result}, CSV says {supplied_result}"
            )

        odds_source, is_closing, odds = _select_odds(row)
        probabilities = _no_vig_probabilities(odds) if odds else (None, None, None)
        home_odds, draw_odds, away_odds = odds or (None, None, None)

        rows.append(
            {
                "source": SOURCE,
                "season": label,
                "division": (row.get("Div") or division).strip(),
                "match_date": _parse_date(raw_date),
                "home_team": home,
                "away_team": away,
                "full_time_home": home_goals,
                "full_time_away": away_goals,
                "result_3way": derived_result,
                "odds_source": odds_source,
                "odds_is_closing": bool(is_closing),
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "market_p_home": probabilities[0],
                "market_p_draw": probabilities[1],
                "market_p_away": probabilities[2],
                "source_url": url,
            }
        )
    return rows
