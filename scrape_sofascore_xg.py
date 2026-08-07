"""Collect per-player xG/xGOT aggregates from Sofascore shotmaps."""

import argparse
import logging
import sys

from db import init_sofascore_xg_db, upsert_sofascore_xg
from sofascore_scraper import (
    get_current_season_id,
    get_match_xg_rows,
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
    init_sofascore_xg_db()
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

    total_rows = 0
    matches_with_data = 0
    matches_missing = 0
    total_failures = 0

    for event in events:
        label = (
            f"{event['home_team']} vs {event['away_team']} "
            f"({event['match_date']})"
        )
        log.info(">> %s", label)
        try:
            rows = get_match_xg_rows(event)
            if rows is None:
                matches_missing += 1
                log.warning("   no shotmap data for match %s", event["match_id"])
                continue
            if not rows:
                matches_with_data += 1
                log.info("   shotmap present, 0 players with shots")
                continue

            upsert_sofascore_xg(rows)
            matches_with_data += 1
            total_rows += len(rows)
            match_xg = round(sum(row["xg"] for row in rows), 2)
            log.info("   %d players with shots, total xG %.2f", len(rows), match_xg)
        except Exception as exc:
            total_failures += 1
            log.error("   FAILED match %s: %s", event["match_id"], exc)

    log.info(
        "Done -- %d player-match xG rows across %d matches "
        "(%d with shotmap, %d missing), %d failures",
        total_rows,
        len(events),
        matches_with_data,
        matches_missing,
        total_failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Sofascore shotmap xG/xGOT for Super League 1"
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
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()
    run(args.date_from, args.date_to, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
