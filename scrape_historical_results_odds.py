"""Download historical Greek Super League results and odds into odds.db."""

from __future__ import annotations

import argparse
import logging
import time

from db import init_historical_results_odds_db, upsert_historical_results_odds
from football_data_scraper import fetch_season_csv, iter_seasons, parse_season_csv

log = logging.getLogger(__name__)


def run(start_year: int, end_year: int, pause_seconds: float = 2.0) -> None:
    init_historical_results_odds_db()
    seasons = list(iter_seasons(start_year, end_year))

    total_rows = 0
    total_closing = 0
    for index, (season_start, season_end) in enumerate(seasons):
        if index:
            time.sleep(pause_seconds)
        log.info("Downloading %d/%d", season_start, season_end)
        csv_text, url = fetch_season_csv(
            season_start,
            season_end,
        )
        rows = parse_season_csv(
            csv_text,
            start_year=season_start,
            end_year=season_end,
            source_url=url,
        )
        upsert_historical_results_odds(rows)
        closing = sum(bool(row["odds_is_closing"]) for row in rows)
        total_rows += len(rows)
        total_closing += closing
        log.info(
            "%d/%d: stored %d results (%d with selected closing odds)",
            season_start,
            season_end,
            len(rows),
            closing,
        )

    log.info(
        "Done: %d official results, %d with selected closing odds",
        total_rows,
        total_closing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Football-Data Greek results and historical 1X2 odds"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="First season start year (default: 2015)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="Last season end year (default: 2026)",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=2.0,
        help="Delay between season downloads (default: 2)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    run(args.start_year, args.end_year, args.pause_seconds)


if __name__ == "__main__":
    main()
