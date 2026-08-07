#!/usr/bin/env python3
"""
collect_odds.py  —  Super League 1 odds snapshot collector

Scrapes home/draw/away odds for all Super League 1 fixtures within the
next 14 days from Stoiximan and Novibet, then persists them to odds.db.

Usage:
    python collect_odds.py

Cron (daily at 08:00):
    0 8 * * * cd /path/to/project && python collect_odds.py >> logs/odds.log 2>&1
"""

import logging
import sys

import db
import novibet_scraper
import stoiximan_scraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("collect_odds.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("collect_odds")


def run_scraper(name: str, scrape_fn) -> int:
    """Run one scraper, store results, return row count (0 on failure)."""
    logger.info("─── %s ───", name)
    try:
        rows = scrape_fn()
    except Exception as exc:
        logger.error("%s scraper raised: %s", name, exc, exc_info=True)
        return 0

    if not rows:
        logger.warning("%s: 0 matches scraped", name)
        return 0

    try:
        db.insert_odds(rows)
    except Exception as exc:
        logger.error("%s: DB insert failed: %s", name, exc, exc_info=True)
        return 0

    logger.info("%s: %d rows stored", name, len(rows))
    return len(rows)


def main():
    logger.info("══════════════════════════════════════════")
    logger.info("  Super League 1 odds collection — start  ")
    logger.info("══════════════════════════════════════════")

    db.init_db()

    total = 0
    total += run_scraper("Stoiximan", stoiximan_scraper.scrape)
    total += run_scraper("Novibet",   novibet_scraper.scrape)

    logger.info("══ Done. Total rows inserted: %d ══", total)
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
