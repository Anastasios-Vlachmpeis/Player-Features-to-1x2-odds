"""
Entry point: python scrape_sofascore.py [--from YYYY-MM-DD] [--to YYYY-MM-DD]

Iterates every finished Greek Super League 1 match in the current season and
populates the sofascore_players and sofascore_match_stats tables in
player_stats.db (one row per player per match they appeared in).

Optional date range lets you re-scrape only recent matches:
    python scrape_sofascore.py --from 2026-05-01
    python scrape_sofascore.py --from 2026-05-01 --to 2026-05-15
"""

import sys
import argparse
import logging

from db import (
    init_sofascore_db,
    upsert_sofascore_player,
    upsert_sofascore_match_stats,
)
from sofascore_scraper import (
    get_current_season_id,
    get_season_events,
    get_match_player_rows,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def run(date_from=None, date_to=None) -> None:
    init_sofascore_db()

    season_id = get_current_season_id()
    events = get_season_events(season_id, date_from, date_to)
    if not events:
        log.error("No events found — check season id, date range, or connectivity")
        sys.exit(1)

    total_rows = 0
    total_failures = 0

    for ev in events:
        label = f"{ev['home_team']} vs {ev['away_team']} ({ev['match_date']})"
        log.info(">> %s", label)
        try:
            rows = get_match_player_rows(ev)
            if not rows:
                log.warning("   no player rows for match %s", ev["match_id"])
                continue

            # Maintain the player-name lookup table
            for r in rows:
                if r.get("player_name"):
                    upsert_sofascore_player(r["sofascore_id"], r["player_name"])

            upsert_sofascore_match_stats(rows)
            total_rows += len(rows)
            log.info("   %d players saved", len(rows))
        except Exception as exc:
            total_failures += 1
            log.error("   FAILED match %s: %s", ev["match_id"], exc)

    log.info(
        "Done — %d player-match rows across %d matches, %d failures",
        total_rows,
        len(events),
        total_failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Sofascore per-match player stats for Super League 1"
    )
    parser.add_argument("--from", dest="date_from", default=None,
                        help="Only matches on/after this date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", default=None,
                        help="Only matches on/before this date (YYYY-MM-DD)")
    args = parser.parse_args()
    run(args.date_from, args.date_to)


if __name__ == "__main__":
    main()
