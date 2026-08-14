"""Fetch 2025-26 matches for all six research leagues and merge them into CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "statsapi"
ENV_FILE = PROJECT_ROOT / ".env"

BASE_URL = "https://api.thestatsapi.com/api"
LEAGUES = {
    "greece": {"name": "Greece", "competition_id": "comp_4008"},
    "turkey": {"name": "Turkey", "competition_id": "comp_9235"},
    "netherlands": {"name": "Netherlands", "competition_id": "comp_3809"},
    "portugal": {"name": "Portugal", "competition_id": "comp_8385"},
    "belgium": {"name": "Belgium", "competition_id": "comp_8531"},
    "scotland": {"name": "Scotland", "competition_id": "comp_6387"},
}
SEASON_START_YEARS = (2025,)
PER_PAGE = 100
# 2.1 seconds caps the theoretical request rate below the API limit of 30/minute.
DEFAULT_SLEEP_SECONDS = 2.1
MINIMUM_SLEEP_SECONDS = 2.1
REQUEST_TIMEOUT_SECONDS = 60
MAX_429_RETRIES = 5

MATCH_COLUMNS = [
    "competition_id",
    "season_id",
    "season",
    "match_id",
    "utc_date",
    "status",
    "matchday",
    "stage_name",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_score",
    "away_score",
    "xg_available",
]


class ApiError(RuntimeError):
    pass


def load_api_key() -> str:
    if value := os.environ.get("THESTATSAPI_KEY"):
        return value

    if ENV_FILE.exists():
        for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "THESTATSAPI_KEY":
                return value.strip().strip('"').strip("'")

    raise RuntimeError(f"THESTATSAPI_KEY was not found in the environment or {ENV_FILE}")


class StatsApiClient:
    def __init__(self, api_key: str, sleep_seconds: float) -> None:
        if sleep_seconds < MINIMUM_SLEEP_SECONDS:
            raise ValueError(
                f"--sleep must be at least {MINIMUM_SLEEP_SECONDS} seconds "
                "to remain below 30 requests per minute"
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.sleep_seconds = sleep_seconds
        self.requests_used = 0

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        for attempt in range(MAX_429_RETRIES + 1):
            if self.requests_used:
                time.sleep(self.sleep_seconds)

            response = self.session.get(
                url,
                params=params or {},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            self.requests_used += 1

            if response.status_code == 429 and attempt < MAX_429_RETRIES:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(60, 5 * (2**attempt))
                print(f"Rate limited; waiting {wait:.0f} seconds")
                time.sleep(wait)
                continue

            if not response.ok:
                try:
                    error = response.json().get("error", {})
                    message = error.get("message", response.text)
                except ValueError:
                    message = response.text
                raise ApiError(f"{response.status_code} {path}: {message}")

            return response.json()

        raise ApiError(f"Repeated rate limits for {path}")

    def paginate(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        base_params = dict(params or {})
        base_params["per_page"] = PER_PAGE
        rows: list[dict[str, Any]] = []
        page = 1

        while True:
            payload = self.get(path, {**base_params, "page": page})
            rows.extend(payload.get("data", []))
            total_pages = int(payload.get("meta", {}).get("total_pages", page))
            if page >= total_pages:
                return rows
            page += 1


def selected_leagues(values: list[str] | None) -> list[str]:
    return list(dict.fromkeys(values)) if values else list(LEAGUES)


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def find_seasons(
    client: StatsApiClient,
    league_name: str,
    competition_id: str,
) -> dict[int, str]:
    seasons = client.paginate(f"/football/competitions/{competition_id}/seasons")
    result: dict[int, str] = {}
    for start_year in SEASON_START_YEARS:
        match = next(
            (
                season
                for season in seasons
                if season.get("start_year") == start_year
                and season.get("end_year") == start_year + 1
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                f"TheStatsAPI has no {season_label(start_year)} season for {league_name}"
            )
        result[start_year] = str(match["id"])
    return result


def nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def score_value(match: dict[str, Any], side: str) -> Any:
    candidates = [
        nested_value(match, "score", "final_score", side),
        nested_value(match, "score", "full_time", side),
        nested_value(match, "score", side),
        nested_value(match, f"{side}_team", "score"),
        match.get(f"{side}_score"),
    ]
    return next((value for value in candidates if value is not None), None)


def match_row(
    match: dict[str, Any],
    competition_id: str,
    season_id: str,
    label: str,
) -> dict[str, Any]:
    home = match.get("home_team") or match.get("home") or {}
    away = match.get("away_team") or match.get("away") or {}
    stage = match.get("stage") or {}

    return {
        "competition_id": competition_id,
        "season_id": season_id,
        "season": label,
        "match_id": match.get("id") or match.get("match_id"),
        "utc_date": match.get("utc_date") or match.get("kickoff_utc"),
        "status": match.get("status"),
        "matchday": match.get("matchday"),
        "stage_name": stage.get("name") if isinstance(stage, dict) else stage,
        "home_team_id": home.get("id") or home.get("team_id"),
        "home_team": home.get("name") or home.get("team"),
        "away_team_id": away.get("id") or away.get("team_id"),
        "away_team": away.get("name") or away.get("team"),
        "home_score": score_value(match, "home"),
        "away_score": score_value(match, "away"),
        "xg_available": match.get("xg_available"),
    }


def write_matches(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def load_existing_matches(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MATCH_COLUMNS:
            raise RuntimeError(
                f"Unexpected columns in {path}; expected {MATCH_COLUMNS}, "
                f"found {reader.fieldnames}"
            )
        return list(reader)


def merge_matches(
    existing: list[dict[str, Any]],
    fetched: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in [*existing, *fetched]:
        match_id = str(row.get("match_id") or "").strip()
        if not match_id:
            raise RuntimeError("At least one match did not contain a match ID")
        merged[match_id] = row
    return sorted(
        merged.values(),
        key=lambda row: (str(row["utc_date"]), str(row["match_id"])),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league",
        action="append",
        choices=list(LEAGUES),
        help="Fetch one league; repeat for several. Default: all six.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help=f"Seconds between requests; minimum {MINIMUM_SLEEP_SECONDS}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sleep < MINIMUM_SLEEP_SECONDS:
        raise SystemExit(
            f"--sleep must be at least {MINIMUM_SLEEP_SECONDS} seconds "
            "to remain below 30 requests per minute"
        )

    requested = selected_leagues(args.league)
    client = StatsApiClient(load_api_key(), args.sleep)
    failures: list[tuple[str, str]] = []

    for league in requested:
        config = LEAGUES[league]
        league_name = config["name"]
        competition_id = config["competition_id"]
        output = DATA_ROOT / league / "matches.csv"
        try:
            season_ids = find_seasons(client, league_name, competition_id)
            fetched_rows: list[dict[str, Any]] = []
            for start_year, season_id in season_ids.items():
                label = season_label(start_year)
                print(f"Fetching {league_name} {label}...")
                matches = client.paginate(
                    "/football/matches",
                    {
                        "competition_id": competition_id,
                        "season_id": season_id,
                        "status": "finished",
                    },
                )
                fetched_rows.extend(
                    match_row(match, competition_id, season_id, label)
                    for match in matches
                )
                print(f"  {len(matches)} matches")

            existing_rows = load_existing_matches(output)
            rows = merge_matches(existing_rows, fetched_rows)
            write_matches(rows, output)
            print(
                f"Merged {len(fetched_rows)} 2025-26 matches into {output}; "
                f"CSV now contains {len(rows)} matches"
            )
        except (ApiError, RuntimeError) as error:
            failures.append((league_name, str(error)))
            print(f"FAILED {league_name}: {error}")

    print(f"API requests used: {client.requests_used}")
    if failures:
        print("Failed leagues can be rerun without refetching completed leagues:")
        for league_name, error in failures:
            print(f"  {league_name}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
