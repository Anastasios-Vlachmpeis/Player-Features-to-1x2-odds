"""Collect Greek Super League lineups, player stats, and official results.

With no season arguments, only the current season is collected. For a historical
backfill, pass both calendar boundaries; ``2015`` through ``2026`` means seasons
2015/16, 2016/17, ..., 2025/26.
"""

import argparse
import logging
import sys

from db import (
    init_sofascore_db,
    upsert_sofascore_matches,
    upsert_sofascore_match_stats,
    upsert_sofascore_player,
)
from sofascore_scraper import (
    get_current_season_id,
    get_match_player_rows,
    get_season_events,
    get_seasons_between,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _season_targets(start_year=None, end_year=None) -> list[dict]:
    if start_year is None and end_year is None:
        return [{"id": get_current_season_id(), "name": None}]
    if start_year is None or end_year is None:
        raise ValueError("--start-year and --end-year must be supplied together")
    return get_seasons_between(start_year, end_year)


def run(date_from=None, date_to=None, start_year=None, end_year=None) -> None:
    init_sofascore_db()
    seasons = _season_targets(start_year, end_year)

    events = []
    for season in seasons:
        events.extend(
            get_season_events(
                season["id"],
                date_from,
                date_to,
                season_name=season.get("name"),
            )
        )

    if not events:
        log.error("No events found -- check season range, date range, or connectivity")
        raise SystemExit(1)

    # Store official scores before the slower per-match lineup requests.
    upsert_sofascore_matches(events)

    total_rows = 0
    total_failures = 0
    for event in events:
        label = (
            f"{event['home_team']} vs {event['away_team']} "
            f"({event['match_date']})"
        )
        log.info(">> %s", label)
        try:
            rows = get_match_player_rows(event)
            if not rows:
                log.warning("   no player rows for match %s", event["match_id"])
                continue

            for row in rows:
                if row.get("player_name"):
                    upsert_sofascore_player(
                        row["sofascore_id"], row["player_name"]
                    )

            upsert_sofascore_match_stats(rows)
            total_rows += len(rows)
            log.info("   %d players saved", len(rows))
        except Exception as exc:
            total_failures += 1
            log.error("   FAILED match %s: %s", event["match_id"], exc)

    log.info(
        "Done -- %d player-match rows across %d matches, %d failures",
        total_rows,
        len(events),
        total_failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Sofascore player stats and results for Super League 1"
    )
    parser.add_argument(
        "--from",
        dest="date_from",
        default=None,
        help="Only matches on/after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        default=None,
        help="Only matches on/before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First season start year, e.g. 2015",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last season end year, e.g. 2026",
    )
    args = parser.parse_args()
    run(args.date_from, args.date_to, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
