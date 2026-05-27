"""
Stoiximan Super League 1 odds scraper.

Navigates directly to the Super League 1 competition page, scrolls to
load all fixtures, extracts the text blob from the main events container,
and parses home/draw/away odds for matches within the next 14 days.

SELECTOR NOTES (update these if the scraper returns 0 matches):
  - Cookie button  : OneTrust ID "onetrust-accept-btn-handler"
                     — very stable; only change if they switch consent provider
  - Login modal    : class fragment "sb-modal__close" or aria-label "Close"
                     — Stoiximan has redesigned this; inspect .sb-modal or
                       [data-testid] attributes if both fallbacks miss
  - Content div    : class fragment "grid__column--main" or "events-list"
                     — if you get 0 matches, open DevTools, find the div
                       wrapping all event rows, and add its selector here
  - Date format    : page emits DD/MM lines as date headers; "Σήμερα"
                     and "Αύριο" are also handled as Today/Tomorrow
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

BOOKMAKER = "stoiximan"
SUPER_LEAGUE_URL = (
    "https://www.stoiximan.gr/sports/podosfairo/ellada/super-league-1/"
)

logger = logging.getLogger(__name__)

# --- regex helpers ---
_DATE_RE  = re.compile(r"^(\d{1,2})[/.](\d{1,2})([/.](\d{2,4}))?$")
_TIME_RE  = re.compile(r"^\d{2}:\d{2}$")
_ODDS_RE  = re.compile(r"^\d+[.,]\d+$")
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

# Lines that carry no match data and should be skipped wholesale
_NOISE = frozenset({
    "1", "x", "2", "1x", "x2", "12",
    "gg", "ng", "o", "u",
    "live", "αρχική", "αθλήματα", "ποδόσφαιρο", "μπάσκετ", "τένις",
    "αγαπημένα", "εγγραφή", "σύνδεση", "στατιστικά", "εναλλακτικά",
    "αποτελέσματα", "αγορές", "κουπόνι", "ζωντανά", "επόμενα",
    "super league 1", "ελλάδα", "στοίχημα", "κεφάλαιο",
    "no_bet", "no bet", "-", "—",
    "κύπελλο", "φάσεις κυπέλλου", "ψηλά", "χαμηλά",
    "ψηλά/χαμηλά", "έναρξη αγώνα",
})


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_driver(headless: bool = False) -> uc.Chrome:
    """
    headless=False is recommended: fully headless Chrome is more easily
    fingerprinted than a visible window.  On a headless server use
    Xvfb + headless=False, or pass headless=True and accept the risk.
    """
    opts = uc.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    return uc.Chrome(options=opts)


# ---------------------------------------------------------------------------
# Page interaction helpers
# ---------------------------------------------------------------------------

def _click_first_found(driver, candidates, timeout=8) -> bool:
    """Try each (By, selector) until one is clickable; return True on success."""
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
    # 1. Cookie consent (OneTrust — very common across Betsson group sites)
    found = _click_first_found(driver, [
        (By.ID, "onetrust-accept-btn-handler"),
        (By.XPATH, "//button[contains(translate(., 'αβγδεζηθικλμνξοπρστυφχψω','ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'),'ΑΠΟΔΟΧ')]"),
        (By.XPATH, "//button[contains(text(),'Accept')]"),
        (By.XPATH, "//button[contains(text(),'ACCEPT')]"),
    ])
    if found:
        logger.info("  cookies accepted")

    # 2. Login/promo modal  (class names change; aria-label is more stable)
    found = _click_first_found(driver, [
        (By.CSS_SELECTOR, "[class*='sb-modal__close']"),
        (By.CSS_SELECTOR, "[data-testid='modal-close-btn']"),
        (By.CSS_SELECTOR, "button[aria-label='Close']"),
        (By.CSS_SELECTOR, "button[aria-label='Κλείσιμο']"),
        (By.XPATH, "//button[@aria-label='Close']"),
    ], timeout=6)
    if found:
        logger.info("  modal dismissed")


def _scroll_to_load(driver):
    """
    Scroll down in steps to trigger lazy-loaded event rows, then return
    to the top so text extraction reads in DOM order.
    """
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


def _get_content_text(driver) -> str:
    """
    Return the text content of the main event list container.
    Tries progressively broader selectors; falls back to <body>.
    """
    for sel in [
        "[class*='grid__column--main']",
        "[class*='events-list']",
        "[class*='competition-events']",
        "[class*='event-group']",
        "[class*='eventsPanel']",
        "main",
        "[role='main']",
    ]:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        if els:
            logger.info("  content container matched: %s", sel)
            return els[0].text
    logger.warning("  falling back to <body> text")
    return driver.find_element(By.TAG_NAME, "body").text


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _resolve_date(raw: str) -> str | None:
    """
    Parse a raw date line into 'YYYY-MM-DD', or None if unrecognisable.
    Handles: DD/MM, DD/MM/YY, DD/MM/YYYY, DD Month(Greek),
             'Σήμερα' (today), 'Αύριο' (tomorrow).
    """
    raw = raw.strip()
    low = raw.lower()

    if low in ("σήμερα", "today"):
        return datetime.now().date().strftime("%Y-%m-%d")
    if low in ("αύριο", "tomorrow"):
        return (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")

    # DD/MM[/YYYY]
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
            # If the date already passed this year, it must be next year
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

    # DD MonthNameGreek [YYYY]
    mg = _GREEK_DATE_RE.search(raw)
    if mg:
        day = int(mg.group(1))
        month_key = mg.group(2).lower()
        month = _GREEK_MONTHS.get(month_key)
        if month:
            # Look for a 4-digit year in the original string
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


# ---------------------------------------------------------------------------
# match_id builder
# ---------------------------------------------------------------------------

def _make_match_id(home: str, away: str, date: str) -> str:
    def slug(s):
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return f"{slug(home)}_{slug(away)}_{date}"


# ---------------------------------------------------------------------------
# Text parser
# ---------------------------------------------------------------------------

def _parse_lines(lines: list[str]) -> list[tuple]:
    """
    Walk the text lines emitted by Stoiximan's event list and extract
    1X2 odds rows.

    Stoiximan's typical text rhythm (after noise removal):
        <date header>        e.g. "11/01" or "Σήμερα"
        <HH:MM>
        <home team name>
        <away team name>
        1                    literal label
        <odds>               e.g. "2.10"
        X                    literal label
        <odds>
        2                    literal label
        <odds>
        ... (more markets – ignored)

    The parser is intentionally lenient: it uses the HH:MM line as an
    anchor and looks ahead for the expected pattern rather than assuming
    strict offset positions.
    """
    rows = []
    current_date: str | None = None
    n = len(lines)

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue

        # Skip known noise (case-insensitive)
        if line.lower() in _NOISE:
            continue

        # Date header?
        date_attempt = _resolve_date(line)
        if date_attempt:
            current_date = date_attempt
            continue

        # Day-of-week labels in Greek – just skip them, the DD/MM follows
        if re.match(
            r"^(Δευτέρα|Τρίτη|Τετάρτη|Πέμπτη|Παρασκευή|Σάββατο|Κυριακή)",
            line,
        ):
            continue

        # Match anchor: HH:MM time
        if not _TIME_RE.match(line) or current_date is None:
            continue

        # Look ahead for: team1, team2, "1", odds, "X", odds, "2", odds
        # Allow up to 3 skippable noise lines between anchor and teams
        remaining = [l.strip() for l in lines[i + 1 :] if l.strip()]
        # Strip any immediate noise from the front
        remaining = [l for l in remaining if l.lower() not in _NOISE]

        if len(remaining) < 8:
            continue

        team1 = remaining[0]
        team2 = remaining[1]
        lbl1  = remaining[2]
        odd1  = remaining[3]
        lblx  = remaining[4]
        oddx  = remaining[5]
        lbl2  = remaining[6]
        odd2  = remaining[7]

        # Validate structure
        if (
            lbl1 == "1"
            and lblx.upper() == "X"
            and lbl2 == "2"
            and _ODDS_RE.match(odd1)
            and _ODDS_RE.match(oddx)
            and _ODDS_RE.match(odd2)
            # team names must not be odds or single-char labels
            and not _ODDS_RE.match(team1)
            and not _ODDS_RE.match(team2)
            and len(team1) > 1
            and len(team2) > 1
        ):
            if _within_window(current_date):
                row = (
                    _make_match_id(team1, team2, current_date),
                    team1,
                    team2,
                    current_date,
                    BOOKMAKER,
                    float(odd1.replace(",", ".")),
                    float(oddx.replace(",", ".")),
                    float(odd2.replace(",", ".")),
                )
                rows.append(row)
                logger.info(
                    "  match: %s vs %s  %s  1=%.2f X=%.2f 2=%.2f",
                    team1, team2, current_date,
                    row[5], row[6], row[7],
                )

    return rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def scrape(driver: uc.Chrome | None = None) -> list[tuple]:
    """
    Scrape Super League 1 odds from Stoiximan.
    Returns a list of rows ready for db.insert_odds().
    Pass an existing driver to reuse it; otherwise a new one is created
    and closed after this call.
    """
    own_driver = driver is None
    if own_driver:
        driver = make_driver()

    rows: list[tuple] = []
    try:
        logger.info("[stoiximan] navigating → %s", SUPER_LEAGUE_URL)
        driver.get(SUPER_LEAGUE_URL)
        time.sleep(4)

        _dismiss_overlays(driver)
        time.sleep(2)
        _scroll_to_load(driver)

        page_text = _get_content_text(driver)
        lines = page_text.split("\n")
        logger.info("[stoiximan] raw lines from container: %d", len(lines))

        rows = _parse_lines(lines)
        logger.info("[stoiximan] parsed %d match rows", len(rows))

        if not rows:
            logger.warning(
                "[stoiximan] 0 rows — likely stale selectors or page structure change. "
                "Check SUPER_LEAGUE_URL and the content container selector in "
                "_get_content_text()."
            )

    except Exception as exc:
        logger.error("[stoiximan] scrape failed: %s", exc, exc_info=True)
    finally:
        if own_driver:
            driver.quit()

    return rows
