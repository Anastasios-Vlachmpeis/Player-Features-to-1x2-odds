"""
Novibet Super League 1 odds scraper.

Strategy
--------
1. Try navigating directly to the Super League competition page.
2. If the direct URL yields no matches, fall back to the Daily Coupon
   feature (Novibet's compact multi-sport listing) and filter for the
   "Super League 1" championship group.

SELECTOR NOTES (update these if the scraper returns 0 matches):
  - Cookie button  : ".acceptCookies_button" (Angular component class)
                     — try "[id*='acceptCookies']" or an XPath on button
                       text if the component name changes
  - Login modal    : "[data-cy='closeBtn']" (Cypress test attr — usually stable)
                     — fall back to "[class*='close'][class*='btn']" or
                       "[aria-label='Close']"
  - Daily coupon   : "a[title='Daily coupon']" or "a[title='Ημερήσιο κουπόνι']"
                     — inspect <a> elements in the top nav if this fails
  - Coupon body    : ".dailyCoupon_body"
                     — if the component was renamed, check for
                       "[class*='dailyCoupon']" or "[class*='daily-coupon']"
  - Direct league  : ".events-list", "[class*='event-list']", etc.

TEXT FORMAT (Daily Coupon)
--------------------------
Novibet's daily coupon emits a text blob structured as championship
blocks separated by " - " (e.g. "Ελλάδα - Super League 1").
Within each block, football matches follow an 18-token rhythm:
    Team1, Team2, HH:MM,
    "1", odds1, "X", oddsX, "2", odds2,
    "O", O_odds, "U", U_odds,
    "GG", GG_odds, "NG", NG_odds

TEXT FORMAT (Direct competition page)
--------------------------------------
The direct page typically shows:
    <date header>   e.g. "11/01" or Greek date string
    HH:MM
    Team1
    Team2
    odds1  oddsX  odds2  (sometimes all on one line, sometimes separate)
"""

import logging
import re
import time
from datetime import datetime, timedelta

import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

BOOKMAKER = "novibet"

# ── COMPETITION TARGET ────────────────────────────────────────────────────────
# Uncomment the league you want and comment out the rest.
# Novibet URLs often contain numeric category IDs — if a URL 404s, browse the
# site, navigate to the competition, and copy the URL from the address bar.

DIRECT_URLS = [
    "https://www.novibet.gr/stoixima/podosfairo/champions-league",        # Champions League (verify path)
    "https://www.novibet.gr/stoixima/podosfairo/uefa-champions-league",   # alt slug
]

# Super League 1 (Greece)
# DIRECT_URLS = [
#     "https://www.novibet.gr/stoixima/podosfairo/4372606/greece/super-league-1/5909217"
# ]

# Ligue 1 (France) — find the numeric IDs from the address bar
# DIRECT_URLS = [
#     "https://www.novibet.gr/stoixima/podosfairo/france/ligue-1",
# ]
# ─────────────────────────────────────────────────────────────────────────────

MAIN_URL   = "https://www.novibet.gr/"
COUPON_URL = "https://www.novibet.gr/sports/daily-coupon"

logger = logging.getLogger(__name__)

_DATE_RE   = re.compile(r"^(\d{1,2})[/.](\d{1,2})([/.](\d{2,4}))?$")
_TIME_RE   = re.compile(r"^\d{2}:\d{2}$")
_ODDS_RE   = re.compile(r"^\d+[.,]\d+$")
_GREEK_DATE_RE = re.compile(
    r"(\d{1,2})\s+(Ιαν|Φεβ|Μαρ|Απρ|Μαϊ|Μάϊ|Ιουν|Ιουλ|Αυγ|Σεπ|Οκτ|Νοε|Δεκ"
    r"|Ιανουαρίου|Φεβρουαρίου|Μαρτίου|Απριλίου|Μαΐου|Ιουνίου|Ιουλίου"
    r"|Αυγούστου|Σεπτεμβρίου|Οκτωβρίου|Νοεμβρίου|Δεκεμβρίου)",
    re.IGNORECASE,
)
_GREEK_MONTHS = {
    "ιαν": 1,  "φεβ": 2,  "μαρ": 3,  "απρ": 4,
    "μαϊ": 5,  "μάϊ": 5,  "ιουν": 6, "ιουλ": 7,
    "αυγ": 8,  "σεπ": 9,  "οκτ": 10, "νοε": 11,  "δεκ": 12,
    "ιανουαρίου": 1, "φεβρουαρίου": 2, "μαρτίου": 3,  "απριλίου": 4,
    "μαΐου": 5,      "ιουνίου": 6,    "ιουλίου": 7,  "αυγούστου": 8,
    "σεπτεμβρίου": 9,"οκτωβρίου": 10, "νοεμβρίου": 11,"δεκεμβρίου": 12,
}

_NOISE = frozenset({
    "1", "x", "2", "1x", "x2", "12",
    "gg", "ng", "o", "u",
    "live", "αρχική", "αθλήματα", "ποδόσφαιρο", "μπάσκετ", "τένις",
    "αγαπημένα", "εγγραφή", "σύνδεση", "στατιστικά", "εναλλακτικά",
    "αποτελέσματα", "αγορές", "κουπόνι", "ζωντανά", "επόμενα",
    "super league 1", "ελλάδα", "στοίχημα", "κεφάλαιο",
    "no_bet", "no bet", "-", "—",
    "κύπελλο", "φάσεις κυπέλλου", "ψηλά", "χαμηλά", "ψηλά/χαμηλά",
    "ημερήσιο κουπόνι", "daily coupon",
    "markets are not available",
})

# ── COUPON FILTER KEYWORDS ───────────────────────────────────────────────────
# These are matched against championship-group headers in the daily coupon
# fallback.  Update when you change DIRECT_URLS above.

_TARGET_KEYWORDS = frozenset({
    "champions league", "champion's league", "champions-league",
    "uefa champions", "τσάμπιονς λιγκ",  # Greek transliteration
})

# Super League 1 (Greece)
# _TARGET_KEYWORDS = frozenset({
#     "super league 1", "super league", "super-league", "superleague",
#     "ελλάδα",
# })

# Ligue 1 (France)
# _TARGET_KEYWORDS = frozenset({
#     "ligue 1", "ligue1", "γαλλία", "france",
# })
# ─────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_driver(headless: bool = False) -> uc.Chrome:
    opts = uc.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    return uc.Chrome(options=opts, version_main=148)


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------

def _click_first_found(driver, candidates, timeout=8) -> bool:
    for by, sel in candidates:
        try:
            btn = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((by, sel))
            )
            btn.click()
            time.sleep(1.2)
            return True
        except TimeoutException:
            continue
    return False


def _dismiss_overlays(driver):
    # Cookie consent
    found = _click_first_found(driver, [
        (By.CSS_SELECTOR, ".acceptCookies_button"),
        (By.CSS_SELECTOR, "[id*='acceptCookies']"),
        (By.CSS_SELECTOR, "[class*='acceptCookies']"),
        (By.ID, "onetrust-accept-btn-handler"),
        (By.XPATH, "//button[contains(translate(.,'αβγδεζηθικλμνξοπρστυφχψω','ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'),'ΑΠΟΔΟΧ')]"),
        (By.XPATH, "//button[contains(text(),'Accept')]"),
    ])
    if found:
        logger.info("  cookies accepted")

    # Login modal
    found = _click_first_found(driver, [
        (By.CSS_SELECTOR, "[data-cy='closeBtn']"),
        (By.CSS_SELECTOR, "[data-cy='close-btn']"),
        (By.CSS_SELECTOR, "[class*='closeBtn']"),
        (By.CSS_SELECTOR, "button[aria-label='Close']"),
        (By.CSS_SELECTOR, "button[aria-label='Κλείσιμο']"),
        (By.XPATH, "//button[@aria-label='Close']"),
    ], timeout=6)
    if found:
        logger.info("  modal dismissed")


def _scroll_to_load(driver):
    prev = -1
    for _ in range(30):
        driver.execute_script("window.scrollBy(0, 900);")
        time.sleep(0.9)
        cur = driver.execute_script(
            "return window.pageYOffset || document.documentElement.scrollTop;"
        )
        if cur == prev:
            break
        prev = cur
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.8)


def _open_daily_coupon(driver):
    """Click the Daily Coupon link or navigate directly."""
    # Try clicking the nav link first
    found = _click_first_found(driver, [
        (By.CSS_SELECTOR, "a[title='Daily coupon']"),
        (By.CSS_SELECTOR, "a[title='Ημερήσιο κουπόνι']"),
        (By.XPATH, "//a[contains(text(),'Daily coupon')]"),
        (By.XPATH, "//a[contains(text(),'Ημερήσιο κουπόνι')]"),
        (By.XPATH, "//a[contains(@class,'ng-star-inserted') and contains(@title,'coupon')]"),
    ], timeout=8)
    if found:
        logger.info("  clicked daily coupon link")
        time.sleep(2)
        return True

    # Fall back to direct URL
    logger.info("  daily coupon link not found – navigating directly")
    driver.get(COUPON_URL)
    time.sleep(3)
    return True


def _get_coupon_text(driver) -> str | None:
    """Return text from the daily coupon body element."""
    for sel in [
        ".dailyCoupon_body",
        "[class*='dailyCoupon']",
        "[class*='daily-coupon']",
        "[class*='couponBody']",
    ]:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        if els:
            logger.info("  coupon body matched: %s", sel)
            return els[0].text
    return None


def _get_direct_page_text(driver) -> str:
    """Return text from the main event list on a direct competition page."""
    for sel in [
        "[class*='events-list']",
        "[class*='event-list']",
        "[class*='competition-events']",
        "[class*='eventsPanel']",
        "[class*='grid__column--main']",
        "main",
        "[role='main']",
    ]:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        if els:
            logger.info("  direct page container matched: %s", sel)
            return els[0].text
    return driver.find_element(By.TAG_NAME, "body").text


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _resolve_date(raw: str) -> str | None:
    raw = raw.strip()
    low = raw.lower()

    if low in ("σήμερα", "today"):
        return datetime.now().date().strftime("%Y-%m-%d")
    if low in ("αύριο", "tomorrow"):
        return (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")

    m = _DATE_RE.match(raw)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        yr_raw = m.group(4)
        if yr_raw:
            yr = int(yr_raw)
            if yr < 100:
                yr += 2000
        else:
            now = datetime.now()
            yr = now.year
            try:
                candidate = datetime(yr, month, day).date()
                if candidate < now.date() - timedelta(days=1):
                    yr += 1
            except ValueError:
                return None
        try:
            return datetime(yr, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    mg = _GREEK_DATE_RE.search(raw)
    if mg:
        day = int(mg.group(1))
        month_key = mg.group(2).lower()
        month = _GREEK_MONTHS.get(month_key)
        if month:
            yr_m = re.search(r"\b(20\d{2})\b", raw)
            yr = int(yr_m.group(1)) if yr_m else datetime.now().year
            try:
                return datetime(yr, month, day).strftime("%Y-%m-%d")
            except ValueError:
                return None

    return None


def _within_window(date_str: str, days: int = 14) -> bool:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        return today <= d <= today + timedelta(days=days)
    except ValueError:
        return False


def _make_match_id(home: str, away: str, date: str) -> str:
    def slug(s):
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return f"{slug(home)}_{slug(away)}_{date}"


# ---------------------------------------------------------------------------
# Parser: daily coupon text
# ---------------------------------------------------------------------------

def _parse_coupon_text(text: str) -> list[tuple]:
    """
    Parse the Novibet daily coupon text blob.

    The blob is structured as championship blocks separated by
    the pattern "COUNTRY - COMPETITION" (e.g. "Ελλάδα - Super League 1").
    Within a Super League block each football match is 18 tokens:
        Team1, Team2, HH:MM,
        "1", odds1, "X", oddsX, "2", odds2,
        "O", O_odds, "U", U_odds,
        "GG", GG_odds, "NG", NG_odds

    Tokens containing "No_bet" / "Markets are not available" mean that
    market is unavailable; we substitute None for missing odds.
    """
    rows = []
    today = datetime.now().date().strftime("%Y-%m-%d")

    # Split into championship groups on the separator pattern
    # e.g. text contains "Ελλάδα\nSuper League 1\n..."
    # Group headers appear as lines that don't match odds/time/team patterns
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Find the target competition section by matching _TARGET_KEYWORDS
    # against championship-group header lines.
    in_target = False
    block_lines: list[str] = []

    championship_re = re.compile(
        r".{2,30}\s*[-–]\s*.{2,50}",  # "Country - Competition"
    )

    for line in lines:
        low = line.lower()

        # Detect championship header
        if championship_re.match(line) and not _ODDS_RE.match(line) and not _TIME_RE.match(line):
            # Flush previous block if it matched our target competition
            if in_target and block_lines:
                rows.extend(_parse_coupon_block(block_lines, today))
                block_lines = []
            # Check if this header matches our target competition
            in_target = any(kw in low for kw in _TARGET_KEYWORDS)
            continue

        # Also detect competition name on its own line (some sites split the header)
        if any(kw in low for kw in _TARGET_KEYWORDS) and not _ODDS_RE.match(line):
            in_target = True
            block_lines = []
            continue

        if in_target:
            block_lines.append(line)

    # Flush last block
    if in_target and block_lines:
        rows.extend(_parse_coupon_block(block_lines, today))

    return rows


def _parse_coupon_block(lines: list[str], date: str) -> list[tuple]:
    """
    Parse one championship block from the coupon (18 tokens per match).
    date is used as the match date (daily coupon = today's matches).
    """
    rows = []
    # Filter obvious noise
    clean = [l for l in lines if l.lower() not in _NOISE and l]

    i = 0
    while i + 17 < len(clean):
        chunk = clean[i: i + 18]
        team1    = chunk[0]
        team2    = chunk[1]
        match_time = chunk[2]
        # positions 3-17: "1", odds1, "X", oddsX, "2", odds2, ...
        lbl1, odd1 = chunk[3], chunk[4]
        lblx, oddx = chunk[5], chunk[6]
        lbl2, odd2 = chunk[7], chunk[8]

        def safe_float(s):
            if not s or s.lower() in ("no_bet", "no bet", "markets are not available"):
                return None
            try:
                return float(s.replace(",", "."))
            except ValueError:
                return None

        if (
            lbl1 == "1" and lblx.upper() == "X" and lbl2 == "2"
            and _TIME_RE.match(match_time)
            and not _ODDS_RE.match(team1)
            and not _ODDS_RE.match(team2)
            and len(team1) > 1 and len(team2) > 1
        ):
            hw = safe_float(odd1)
            dr = safe_float(oddx)
            aw = safe_float(odd2)
            if hw and dr and aw and _within_window(date):
                row = (
                    _make_match_id(team1, team2, date),
                    team1, team2, date, BOOKMAKER, hw, dr, aw,
                )
                rows.append(row)
                logger.info(
                    "  match: %s vs %s  %s  1=%.2f X=%.2f 2=%.2f",
                    team1, team2, date, hw, dr, aw,
                )
            i += 18
        else:
            i += 1  # re-sync on bad alignment

    return rows


# ---------------------------------------------------------------------------
# Parser: direct competition page (same rhythm as Stoiximan parser)
# ---------------------------------------------------------------------------

def _parse_direct_text(lines: list[str]) -> list[tuple]:
    """
    Parse text from Novibet's direct competition page.
    Layout is similar to Stoiximan's: date header → HH:MM → team1 →
    team2 → "1" → odds → "X" → odds → "2" → odds.
    """
    rows = []
    current_date: str | None = None
    n = len(lines)

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.lower() in _NOISE:
            continue

        date_attempt = _resolve_date(line)
        if date_attempt:
            current_date = date_attempt
            continue

        if re.match(
            r"^(Δευτέρα|Τρίτη|Τετάρτη|Πέμπτη|Παρασκευή|Σάββατο|Κυριακή)",
            line,
        ):
            continue

        if not _TIME_RE.match(line) or current_date is None:
            continue

        # Look ahead, strip noise
        remaining = [l.strip() for l in lines[i + 1:] if l.strip()]
        remaining = [l for l in remaining if l.lower() not in _NOISE]

        if len(remaining) < 8:
            continue

        team1, team2 = remaining[0], remaining[1]
        lbl1,  odd1  = remaining[2], remaining[3]
        lblx,  oddx  = remaining[4], remaining[5]
        lbl2,  odd2  = remaining[6], remaining[7]

        if (
            lbl1 == "1" and lblx.upper() == "X" and lbl2 == "2"
            and _ODDS_RE.match(odd1)
            and _ODDS_RE.match(oddx)
            and _ODDS_RE.match(odd2)
            and not _ODDS_RE.match(team1)
            and not _ODDS_RE.match(team2)
            and len(team1) > 1 and len(team2) > 1
        ):
            if _within_window(current_date):
                row = (
                    _make_match_id(team1, team2, current_date),
                    team1, team2, current_date, BOOKMAKER,
                    float(odd1.replace(",", ".")),
                    float(oddx.replace(",", ".")),
                    float(odd2.replace(",", ".")),
                )
                rows.append(row)
                logger.info(
                    "  match: %s vs %s  %s  1=%.2f X=%.2f 2=%.2f",
                    team1, team2, current_date, row[5], row[6], row[7],
                )

    return rows


# ---------------------------------------------------------------------------
# Scrape strategies
# ---------------------------------------------------------------------------

def _try_direct_urls(driver) -> list[tuple]:
    """Try each candidate Super League URL; return rows from the first success."""
    for url in DIRECT_URLS:
        try:
            logger.info("[novibet] trying direct URL: %s", url)
            driver.get(url)
            time.sleep(4)
            _dismiss_overlays(driver)
            time.sleep(2)
            _scroll_to_load(driver)

            page_text = _get_direct_page_text(driver)
            lines = page_text.split("\n")
            logger.info("[novibet] direct page raw lines: %d", len(lines))

            rows = _parse_direct_text(lines)
            if rows:
                logger.info("[novibet] direct URL success: %d matches", len(rows))
                return rows
            logger.info("[novibet] direct URL yielded 0 matches, trying next")
        except Exception as exc:
            logger.warning("[novibet] direct URL %s failed: %s", url, exc)

    return []


def _try_daily_coupon(driver) -> list[tuple]:
    """Fall back to the Daily Coupon approach and filter to Super League."""
    try:
        logger.info("[novibet] falling back to daily coupon")
        driver.get(MAIN_URL)
        time.sleep(4)
        _dismiss_overlays(driver)
        time.sleep(2)

        _open_daily_coupon(driver)
        _scroll_to_load(driver)

        coupon_text = _get_coupon_text(driver)
        if not coupon_text:
            logger.warning("[novibet] could not find daily coupon body")
            return []

        logger.info("[novibet] coupon text length: %d chars", len(coupon_text))
        rows = _parse_coupon_text(coupon_text)
        logger.info("[novibet] coupon parsed %d Super League rows", len(rows))
        return rows
    except Exception as exc:
        logger.error("[novibet] daily coupon fallback failed: %s", exc, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def scrape(driver: uc.Chrome | None = None) -> list[tuple]:
    """
    Scrape Super League 1 odds from Novibet.
    Returns a list of rows ready for db.insert_odds().
    """
    own_driver = driver is None
    if own_driver:
        driver = make_driver()

    rows: list[tuple] = []
    try:
        # Strategy 1: direct competition page
        rows = _try_direct_urls(driver)

        # Strategy 2: daily coupon fallback
        if not rows:
            rows = _try_daily_coupon(driver)

        if not rows:
            logger.warning(
                "[novibet] 0 rows — check DIRECT_URLS list and coupon selectors in "
                "novibet_scraper.py. Run with headless=False to watch the browser."
            )

    except Exception as exc:
        logger.error("[novibet] scrape failed: %s", exc, exc_info=True)
    finally:
        if own_driver:
            driver.quit()

    return rows
