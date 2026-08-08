"""Download historical top-division results and odds into odds.db."""

from __future__ import annotations

import argparse
import logging
import time

from db import init_historical_results_odds_db, upsert_historical_results_odds
from football_data_scraper import (
    DEFAULT_LEAGUE_KEYS,
    LEAGUES,
    fetch_season_csv,
    iter_seasons,
    league_division,
    parse_season_csv,
    resolve_league_keys,
)

log = logging.getLogger(__name__)


def run(
    start_year: int,
    end_year: int,
    league_keys: list[str] | None = None,
    pause_seconds: float = 2.0,
) -> None:
    init_historical_results_odds_db()
    seasons = list(iter_seasons(start_year, end_year))
    leagues = resolve_league_keys(league_keys)

    total_rows = 0
    total_closing = 0
    request_index = 0
    for league_key in leagues:
        division = league_division(league_key)
        league_name = LEAGUES[league_key]["name"]
        for season_start, season_end in seasons:
            if request_index:
                time.sleep(pause_seconds)
            request_index += 1
            log.info(
                "Downloading %s (%s) %d/%d",
                league_name,
                division,
                season_start,
                season_end,
            )
            csv_text, url = fetch_season_csv(season_start, season_end, division)
            rows = parse_season_csv(
                csv_text,
                start_year=season_start,
                end_year=season_end,
                division=division,
                source_url=url,
            )
            upsert_historical_results_odds(rows)
            closing = sum(bool(row["odds_is_closing"]) for row in rows)
            total_rows += len(rows)
            total_closing += closing
            log.info(
                "%s %d/%d: stored %d results (%d with selected closing odds)",
                division,
                season_start,
                season_end,
                len(rows),
                closing,
            )

    log.info(
        "Done: %d official results across %d league(s), %d with selected closing odds",
        total_rows,
        len(leagues),
        total_closing,
    )


def main() -> None:
    league_help = ", ".join(
        f"{key} ({LEAGUES[key]['division']})" for key in DEFAULT_LEAGUE_KEYS
    )
    parser = argparse.ArgumentParser(
        description=(
            "Collect Football-Data official results and historical 1X2 odds "
            "for selected top divisions"
        )
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
        "--leagues",
        nargs="+",
        choices=DEFAULT_LEAGUE_KEYS,
        metavar="LEAGUE",
        help=(
            "League keys to download (default: all). "
            f"Choices: {league_help}"
        ),
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
    run(args.start_year, args.end_year, args.leagues, args.pause_seconds)


if __name__ == "__main__":
    main()
