"""
Entry point: python scrape_fbref.py [--from YYYY-MM-DD] [--to YYYY-MM-DD]

Iterates current-season Greek Super League 1 matches that have an FBref match
report and populates fbref_players + fbref_match_stats in player_stats.db
(one row per player per match where data exists).

FBref's Greek coverage is patchy by design — missing advanced metrics are
stored as NULL, not treated as failures. The log reports, per match, how many
players carried xG vs how many were missing it.

Optional date range for incremental re-scraping (same pattern as Sofascore):
    python scrape_fbref.py --from 2026-05-01
    python scrape_fbref.py --from 2026-05-01 --to 2026-05-15
"""

import sys
import argparse
import logging

from db import (
    init_fbref_db,
    upsert_fbref_player,
    upsert_fbref_match_stats,
)
from fbref_scraper import get_season_fixtures, get_match_player_rows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def run(date_from=None, date_to=None) -> None:
    init_fbref_db()

    fixtures = get_season_fixtures(date_from, date_to)
    if not fixtures:
        log.error("No fixtures with match reports found — check comp id, "
                  "date range, or connectivity")
        sys.exit(1)

    total_rows = 0
    total_failures = 0

    for fx in fixtures:
        label = f"{fx['home_team']} vs {fx['away_team']} ({fx['match_date']})"
        log.info(">> %s", label)
        try:
            result = get_match_player_rows(fx)
            rows = result["rows"]
            if not rows:
                log.warning("   no player rows (FBref has no detail for this match)")
                continue

            for r in rows:
                if r.get("player_name"):
                    upsert_fbref_player(r["fbref_id"], r["player_name"])

            upsert_fbref_match_stats(rows)
            total_rows += len(rows)

            # Sparse coverage is expected — surface the gap rather than hide it
            with_xg = result["with_xg"]
            log.info("   %d players saved (%d with xG, %d missing advanced metrics)",
                     len(rows), with_xg, len(rows) - with_xg)
        except Exception as exc:
            total_failures += 1
            log.error("   FAILED match %s: %s", fx["match_id"], exc)

    log.info(
        "Done — %d player-match rows across %d matches, %d failures",
        total_rows,
        len(fixtures),
        total_failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape FBref advanced per-match player metrics for Super League 1"
    )
    parser.add_argument("--from", dest="date_from", default=None,
                        help="Only matches on/after this date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", default=None,
                        help="Only matches on/before this date (YYYY-MM-DD)")
    args = parser.parse_args()
    run(args.date_from, args.date_to)


if __name__ == "__main__":
    main()
