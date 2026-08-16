"""Fetch missing 2025-26 player stats for all six saved research leagues."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "statsapi"
ENV_FILE = PROJECT_ROOT / ".env"

BASE_URL = "https://api.thestatsapi.com/api"
LEAGUES = ["greece", "turkey", "netherlands", "portugal", "belgium", "scotland"]
TARGET_SEASON = "2025-26"
# 2.1 seconds caps the theoretical request rate below the API limit of 30/minute.
DEFAULT_SLEEP_SECONDS = 2.1
MINIMUM_SLEEP_SECONDS = 2.1
REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_MAX_REQUESTS = 8_500
MAX_429_RETRIES = 5

MATCH_CONTEXT_COLUMNS = [
    "match_id",
    "season",
    "utc_date",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
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
    def __init__(self, api_key: str, sleep_seconds: float, max_requests: int) -> None:
        if sleep_seconds < MINIMUM_SLEEP_SECONDS:
            raise ValueError(
                f"--sleep must be at least {MINIMUM_SLEEP_SECONDS} seconds "
                "to remain below 30 requests per minute"
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.sleep_seconds = sleep_seconds
        self.max_requests = max_requests
        self.requests_used = 0

    def get(self, path: str) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        for attempt in range(MAX_429_RETRIES + 1):
            if self.requests_used >= self.max_requests:
                raise ApiError(f"Stopped at the {self.max_requests}-request safety limit")
            if self.requests_used:
                time.sleep(self.sleep_seconds)

            response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
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


def selected_leagues(values: list[str] | None) -> list[str]:
    return list(dict.fromkeys(values)) if values else LEAGUES


def league_paths(league: str) -> dict[str, Path]:
    data_dir = DATA_ROOT / league
    return {
        "matches": data_dir / "matches.csv",
        "raw": data_dir / "player_stats_raw",
        "player_stats": data_dir / "player_match_stats.csv",
        "failures": data_dir / "failed_requests.csv",
    }


def load_matches(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Run fetch_matches.py first; {path} does not exist")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{path} contains no matches")
    return rows


def raw_path(raw_dir: Path, match: dict[str, str]) -> Path:
    return raw_dir / match["season"] / f"{match['match_id']}.json"


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def flatten_dict(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        output_key = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(flatten_dict(item, output_key))
        elif isinstance(item, list):
            flattened[output_key] = json.dumps(item, ensure_ascii=False)
        else:
            flattened[output_key] = item
    return flattened


def extract_player_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data", [])
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if isinstance(data, dict):
        if isinstance(data.get("players"), list):
            return [row for row in data["players"] if isinstance(row, dict)]

        rows: list[dict[str, Any]] = []
        for side in ("home", "away"):
            side_data = data.get(side)
            if isinstance(side_data, list):
                rows.extend(row for row in side_data if isinstance(row, dict))
            elif isinstance(side_data, dict) and isinstance(side_data.get("players"), list):
                rows.extend(row for row in side_data["players"] if isinstance(row, dict))
        return rows

    return []


def normalize_saved_files(
    matches: list[dict[str, str]],
    raw_dir: Path,
    output_path: Path,
) -> int:
    records: list[dict[str, Any]] = []
    for match in matches:
        path = raw_path(raw_dir, match)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for player in extract_player_rows(payload):
            record = {column: match.get(column) for column in MATCH_CONTEXT_COLUMNS}
            player_values = flatten_dict(player)
            for key, value in player_values.items():
                output_key = key if key not in record else f"player.{key}"
                record[output_key] = value
            records.append(record)

    if not records:
        return 0

    extra_columns = sorted(
        {key for record in records for key in record}.difference(MATCH_CONTEXT_COLUMNS)
    )
    columns = MATCH_CONTEXT_COLUMNS + extra_columns
    temporary_path = output_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(output_path)
    return len(records)


def write_failures(failures: list[dict[str, str]], path: Path) -> None:
    if not failures:
        if path.exists():
            path.unlink()
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["match_id", "season", "error"])
        writer.writeheader()
        writer.writerows(failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league",
        action="append",
        choices=LEAGUES,
        help="Fetch one league; repeat for several. Default: all six.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help=f"Seconds between requests; minimum {MINIMUM_SLEEP_SECONDS}.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
        help=f"Maximum new HTTP requests across this run (default: {DEFAULT_MAX_REQUESTS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sleep < MINIMUM_SLEEP_SECONDS:
        raise SystemExit(
            f"--sleep must be at least {MINIMUM_SLEEP_SECONDS} seconds "
            "to remain below 30 requests per minute"
        )
    if args.max_requests < 1:
        raise SystemExit("--max-requests must be positive")

    requested = selected_leagues(args.league)
    league_matches: dict[str, list[dict[str, str]]] = {}
    league_missing: dict[str, list[dict[str, str]]] = {}
    paths_by_league: dict[str, dict[str, Path]] = {}

    for league in requested:
        paths = league_paths(league)
        matches = load_matches(paths["matches"])
        target_matches = [
            match for match in matches if match.get("season") == TARGET_SEASON
        ]
        if not target_matches:
            raise RuntimeError(
                f"{paths['matches']} contains no {TARGET_SEASON} matches; "
                "run fetch_matches.py first"
            )
        missing = [
            match
            for match in target_matches
            if not raw_path(paths["raw"], match).exists()
        ]
        paths_by_league[league] = paths
        league_matches[league] = matches
        league_missing[league] = missing
        print(
            f"{league}: {len(target_matches)} {TARGET_SEASON} matches, "
            f"{len(missing)} still to fetch"
        )

    total_missing = sum(len(matches) for matches in league_missing.values())
    if not total_missing:
        for league in requested:
            paths = paths_by_league[league]
            normalized = normalize_saved_files(
                league_matches[league], paths["raw"], paths["player_stats"]
            )
            print(f"{league}: no API requests needed; normalized {normalized} rows")
        return

    client = StatsApiClient(load_api_key(), args.sleep, args.max_requests)
    failures_by_league: dict[str, list[dict[str, str]]] = {
        league: [] for league in requested
    }
    stop_requested = False

    try:
        for league in requested:
            missing = league_missing[league]
            paths = paths_by_league[league]
            for number, match in enumerate(missing, start=1):
                match_id = match["match_id"]
                try:
                    payload = client.get(f"/football/matches/{match_id}/player-stats")
                    save_json_atomic(raw_path(paths["raw"], match), payload)
                    player_count = len(extract_player_rows(payload))
                    print(
                        f"{league} [{number}/{len(missing)}] "
                        f"{match_id}: saved {player_count} players"
                    )
                except ApiError as error:
                    failures_by_league[league].append(
                        {
                            "match_id": match_id,
                            "season": match["season"],
                            "error": str(error),
                        }
                    )
                    print(
                        f"{league} [{number}/{len(missing)}] "
                        f"{match_id}: FAILED - {error}"
                    )
                    if client.requests_used >= client.max_requests:
                        stop_requested = True
                        break
            if stop_requested:
                break
    except KeyboardInterrupt:
        print("\nStopped by user. Saved match files will be reused next time.")

    for league in requested:
        paths = paths_by_league[league]
        write_failures(failures_by_league[league], paths["failures"])
        normalized = normalize_saved_files(
            league_matches[league], paths["raw"], paths["player_stats"]
        )
        saved = sum(
            raw_path(paths["raw"], match).exists()
            for match in league_matches[league]
        )
        print(
            f"{league}: saved {saved}/{len(league_matches[league])} raw responses; "
            f"normalized {normalized} player rows"
        )

    print(f"API requests used this run: {client.requests_used}")
    if stop_requested:
        print("Request safety limit reached. Rerun the same command to resume.")


if __name__ == "__main__":
    main()
