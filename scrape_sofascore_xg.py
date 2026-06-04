"""
Entry point: python scrape_sofascore_xg.py [--from YYYY-MM-DD] [--to YYYY-MM-DD]

Extension to the Sofascore scraper (Prompt 2). For every finished Super League 1
match — the SAME match list as scrape_sofascore.py — additionally calls the
shotmap endpoint, extracts per-shot xG/xGOT, and aggregates to per-player
per-match totals in the sofascore_xg table of player_stats.db.

Why standalone rather than folded into scrape_sofascore.py:
  - The shotmap is a separate endpoint / separate concern (advanced metrics vs
    base match stats). Keeping it separate means you can re-pull xG incrementally
    without re-scraping every lineup, and a shotmap outage never blocks the
    primary stats run. It reuses the existing season/event discovery and the
    existing request layer (_get_json/session) in sofascore_scraper.py — no new
    request plumbing, no parallel scraper.

Behaviour:
  - Match with no shotmap -> logged as "missing shotmap", nothing written, continue.
  - Players who took no shots simply get no row (correct, not an error).
  - Upsert on (sofascore_id, match_id), same date-range arg as the base script.
"""

import sys
import argparse
import logging

from db import init_sofascore_xg_db, upsert_sofascore_xg
from sofascore_scraper import (
    get_current_season_id,
    get_season_events,
    get_match_xg_rows,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def run(date_from=None, date_to=None) -> None:
    init_sofascore_xg_db()

    season_id = get_current_season_id()
    events = get_season_events(season_id, date_from, date_to)
    if not events:
        log.error("No events found — check season id, date range, or connectivity")
        sys.exit(1)

    total_rows = 0
    matches_with_data = 0
    matches_missing = 0
    total_failures = 0

    for ev in events:
        label = f"{ev['home_team']} vs {ev['away_team']} ({ev['match_date']})"
        log.info(">> %s", label)
        try:
            rows = get_match_xg_rows(ev)

            # None => no shotmap for this match (distinct from "0 shooters")
            if rows is None:
                matches_missing += 1
                log.warning("   no shotmap data for match %s", ev["match_id"])
                continue

            if not rows:
                # Shotmap present but no attributable shooters — rare, not an error
                matches_with_data += 1
                log.info("   shotmap present, 0 players with shots")
                continue

            upsert_sofascore_xg(rows)
            matches_with_data += 1
            total_rows += len(rows)

            match_xg = round(sum(r["xg"] for r in rows), 2)
            log.info("   %d players with shots, total xG %.2f",
                     len(rows), match_xg)
        except Exception as exc:
            total_failures += 1
            log.error("   FAILED match %s: %s", ev["match_id"], exc)

    log.info(
        "Done — %d player-match xG rows across %d matches "
        "(%d with shotmap, %d missing), %d failures",
        total_rows,
        len(events),
        matches_with_data,
        matches_missing,
        total_failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Sofascore shotmap xG/xGOT per player per match"
    )
    parser.add_argument("--from", dest="date_from", default=None,
                        help="Only matches on/after this date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", default=None,
                        help="Only matches on/before this date (YYYY-MM-DD)")
    args = parser.parse_args()
    run(args.date_from, args.date_to)


if __name__ == "__main__":
    main()
