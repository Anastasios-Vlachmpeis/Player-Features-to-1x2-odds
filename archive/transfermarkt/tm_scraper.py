"""
Transfermarkt scraping logic for Greek Super League 1 player ability baselines.

Three public functions consumed by scrape_transfermarkt.py:
  get_clubs()                           -> list[dict]
  get_squad(club_id, club_slug, club)   -> list[dict]
  get_player_profile(tm_id, slug)       -> dict
  get_player_injuries(tm_id, slug)      -> list[dict]
"""

import re
import time
import random
import logging
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

BASE_URL = "https://www.transfermarkt.com"
LEAGUE_URL = f"{BASE_URL}/super-league/startseite/wettbewerb/GR1"

# Realistic Chrome UA — TM blocks the default requests UA immediately.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Referer": "https://www.transfermarkt.com/",
    "DNT": "1",
}

_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch *url* with polite delay + retry.  Returns None on permanent failure."""
    time.sleep(random.uniform(2.0, 3.0))
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.HTTPError as exc:
            # 404 / 403 won't recover — bail out early
            if exc.response is not None and exc.response.status_code in (403, 404):
                log.warning("HTTP %s for %s", exc.response.status_code, url)
                return None
            log.warning("HTTP error on attempt %d for %s: %s", attempt + 1, url, exc)
        except requests.RequestException as exc:
            log.warning("Request error on attempt %d for %s: %s", attempt + 1, url, exc)
        if attempt < retries - 1:
            time.sleep(random.uniform(3.0, 6.0))
    return None


# ---------------------------------------------------------------------------
# Value / date parsers
# ---------------------------------------------------------------------------

def _parse_market_value(text: str) -> Optional[int]:
    """Convert TM value strings ('€5.00m', '€500k', '-') to integer euros."""
    text = text.strip().replace("\xa0", "").replace(",", ".")
    if not text or text in ("-", "€-", "0", ""):
        return None
    text = text.lstrip("€").strip()
    try:
        if "m" in text.lower():
            return int(float(text.lower().replace("m", "")) * 1_000_000)
        if "k" in text.lower():
            return int(float(text.lower().replace("k", "")) * 1_000)
        return int(float(text))
    except ValueError:
        return None


def _parse_dob(text: str) -> tuple:
    """Return (iso_date_str, age_int) from TM DOB cells like 'Jan 1, 1990  (35)'."""
    text = text.strip()
    age: Optional[int] = None
    age_m = re.search(r"\((\d+)\)", text)
    if age_m:
        age = int(age_m.group(1))
        text = text[: age_m.start()].strip()

    for fmt in ("%b %d, %Y", "%d.%m.%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d"), age
        except ValueError:
            continue
    return text or None, age


def _parse_date(text: str) -> Optional[str]:
    """Parse a standalone date cell (injury from/until) to ISO format."""
    text = text.strip().replace("\xa0", "")
    if not text or text == "-":
        return None
    for fmt in ("%b %d, %Y", "%d.%m.%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text or None


def _parse_int(text: str) -> Optional[int]:
    text = text.strip().replace("\xa0", "")
    if not text or text == "-":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _slug_from_href(href: str) -> str:
    """Extract the slug (first path segment) from a TM relative href."""
    return href.strip("/").split("/")[0]


def _tm_id_from_href(href: str) -> Optional[int]:
    """Extract numeric TM id from hrefs like '/name/profil/spieler/12345'."""
    m = re.search(r"/spieler/(\d+)", href)
    if m:
        return int(m.group(1))
    # Club pages use /verein/
    m = re.search(r"/verein/(\d+)", href)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Public scraping functions
# ---------------------------------------------------------------------------

def get_clubs() -> list:
    """
    Scrape the Super League 1 league page and return every current club.

    Returns list of dicts: {club_name, club_id, club_slug}
    """
    soup = _get(LEAGUE_URL)
    if soup is None:
        raise RuntimeError(f"Could not fetch league page: {LEAGUE_URL}")

    clubs = []
    seen = set()

    # Club links live in td.hauptlink > a with href containing /startseite/verein/
    # Fragile: TM sometimes changes the table layout across seasons.
    for a in soup.select("td.hauptlink a[href*='/startseite/verein/']"):
        href = a.get("href", "")
        club_id = _tm_id_from_href(href)
        if club_id is None or club_id in seen:
            continue
        seen.add(club_id)
        clubs.append(
            {
                "club_name": a.get_text(strip=True),
                "club_id": club_id,
                "club_slug": _slug_from_href(href),
            }
        )

    log.info("Found %d clubs", len(clubs))
    return clubs


def get_squad(club_id: int, club_slug: str, club_name: str) -> list:
    """
    Scrape the squad page for *club_id* and return basic player stubs.

    Returns list of dicts: {tm_id, slug, full_name, primary_position,
                             shirt_number, dob, age, nationality,
                             market_value_eur, club}
    """
    url = f"{BASE_URL}/{club_slug}/kader/verein/{club_id}"
    soup = _get(url)
    if soup is None:
        log.error("Could not fetch squad page for %s (%s)", club_name, url)
        return []

    players = []
    # Each player row alternates odd/even in table.items
    for row in soup.select("table.items tbody tr.odd, table.items tbody tr.even"):
        try:
            player = _parse_squad_row(row, club_name)
            if player:
                players.append(player)
        except Exception as exc:
            log.warning("Error parsing squad row for %s: %s", club_name, exc)

    return players


def _parse_squad_row(row, club_name: str) -> Optional[dict]:
    """Parse a single squad-table <tr> into a player dict."""
    # Player link — most reliable anchor for name + tm_id
    # Fragile: TM sometimes wraps the name link in an inline-table inside td.posrela
    player_link = row.select_one("td.posrela a[href*='/profil/spieler/']")
    if player_link is None:
        return None

    href = player_link.get("href", "")
    tm_id = _tm_id_from_href(href)
    if tm_id is None:
        return None

    full_name = player_link.get_text(strip=True)
    slug = _slug_from_href(href)

    # Primary position sits in a second <tr> inside the inline-table within td.posrela
    # Fragile: the inner table structure can shift; fall back to None gracefully
    primary_position = None
    inner_rows = row.select("td.posrela table tr")
    if len(inner_rows) >= 2:
        pos_cell = inner_rows[1].select_one("td")
        if pos_cell:
            primary_position = pos_cell.get_text(strip=True) or None

    # Shirt number — div.rn_nummer inside the first td
    shirt_number = None
    rn = row.select_one("div.rn_nummer")
    if rn:
        shirt_number = _parse_int(rn.get_text())

    # Date of birth — a td whose text matches a date-like pattern
    # Fragile: column order varies; we search all tds rather than indexing by position
    dob, age = None, None
    for td in row.find_all("td"):
        text = td.get_text(strip=True)
        if re.search(r"\b\d{4}\b", text) and re.search(r"\(\d+\)", text):
            dob, age = _parse_dob(text)
            break

    # Nationality — first flaggenrahmen (or img with title) in the row
    nationality = None
    flag_img = row.select_one("img.flaggenrahmen")
    if flag_img:
        nationality = flag_img.get("title") or flag_img.get("alt")

    # Market value — td.rechts.hauptlink (sometimes just td.hauptlink on squad pages)
    # Fragile: class combination varies by TM version
    market_value_eur = None
    for selector in ("td.rechts.hauptlink a", "td.hauptlink a[href*='marktwertverlauf']"):
        mv_el = row.select_one(selector)
        if mv_el:
            market_value_eur = _parse_market_value(mv_el.get_text())
            break

    return {
        "tm_id": tm_id,
        "slug": slug,
        "full_name": full_name,
        "primary_position": primary_position,
        "shirt_number": shirt_number,
        "dob": dob,
        "age": age,
        "nationality": nationality,
        "market_value_eur": market_value_eur,
        "club": club_name,
        "secondary_position": None,  # filled in by get_player_profile
    }


def get_player_profile(tm_id: int, slug: str) -> dict:
    """
    Scrape the player profile page for enriched data — primarily secondary position.
    Falls back to empty dict on failure (caller merges with squad-row data).
    """
    url = f"{BASE_URL}/{slug}/profil/spieler/{tm_id}"
    soup = _get(url)
    if soup is None:
        log.warning("Could not fetch profile for tm_id=%s", tm_id)
        return {}

    data: dict = {}

    # --- Market value (more accurate than squad-page value) ---
    # Fragile: TM redesigns this section periodically
    mv_el = soup.select_one(
        "div.tm-player-market-value-development__current-value"
    )
    if mv_el:
        mv_text = mv_el.get_text(strip=True)
        parsed = _parse_market_value(mv_text)
        if parsed is not None:
            data["market_value_eur"] = parsed

    # --- Positions (primary + secondary) ---
    # The player-detail info table uses label spans followed by value spans.
    # We search for the label text "Position:" and "Other position:" case-insensitively.
    # Fragile: relies on span ordering; breaks if TM moves to a definition list.
    info_spans = soup.select(
        "div.info-table span.info-table__content, "
        "table.auflistung td"  # older TM layout fallback
    )
    label_text = [s.get_text(strip=True) for s in info_spans]

    for i, text in enumerate(label_text):
        if re.search(r"^position:?$", text, re.I):
            if i + 1 < len(label_text):
                data["primary_position"] = label_text[i + 1] or None
        elif re.search(r"other position", text, re.I):
            if i + 1 < len(label_text):
                data["secondary_position"] = label_text[i + 1] or None

    # --- Full name (displayed name may differ from squad-page short name) ---
    h1 = soup.select_one("h1.data-header__headline-wrapper--oswald")
    if h1:
        data["full_name"] = h1.get_text(strip=True)

    # --- DOB / age (profile page is canonical) ---
    for span in info_spans:
        text = span.get_text(strip=True)
        if re.search(r"\b\d{4}\b", text) and re.search(r"\(\d+\)", text):
            dob, age = _parse_dob(text)
            data["dob"] = dob
            data["age"] = age
            break

    # --- Nationality ---
    # Profile shows flag images in a dedicated nationality section
    nat_imgs = soup.select(
        "span.info-table__content img.flaggenrahmen, "
        "td img.flaggenrahmen"
    )
    if nat_imgs:
        data["nationality"] = (
            nat_imgs[0].get("title") or nat_imgs[0].get("alt")
        )

    return data


def get_player_injuries(tm_id: int, slug: str) -> list:
    """
    Scrape the injury history page.  Returns list of dicts ready for upsert_injuries().
    """
    url = f"{BASE_URL}/{slug}/verletzungen/spieler/{tm_id}"
    soup = _get(url)
    if soup is None:
        log.warning("Could not fetch injuries for tm_id=%s", tm_id)
        return []

    injuries = []
    # Injury table has one row per incident; skip header rows and summary rows
    for row in soup.select("table.items tbody tr"):
        cells = row.find_all("td")
        # Expected columns: Season | Injury | From | Until | Days | Games | Club
        # Fragile: if TM adds/removes columns, indices shift — check cell count
        if len(cells) < 6:
            continue
        injury_type = cells[1].get_text(strip=True)
        if not injury_type or injury_type == "-":
            continue
        injuries.append(
            {
                "tm_id": tm_id,
                "injury_type": injury_type,
                "date_from": _parse_date(cells[2].get_text(strip=True)),
                "date_to": _parse_date(cells[3].get_text(strip=True)),
                "matches_missed": _parse_int(cells[5].get_text(strip=True)),
            }
        )

    return injuries
