"""Fetch and normalize per-match player stats for saved Scottish Premiership matches."""

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
DATA_DIR = PROJECT_ROOT / "data" / "statsapi" / "scotland"
MATCHES_CSV = DATA_DIR / "matches.csv"
RAW_DIR = DATA_DIR / "player_stats_raw"
PLAYER_STATS_CSV = DATA_DIR / "player_match_stats.csv"
FAILED_CSV = DATA_DIR / "failed_requests.csv"
ENV_FILE = PROJECT_ROOT / ".env"

BASE_URL = "https://api.thestatsapi.com/api"
DEFAULT_SLEEP_SECONDS = 5.5
REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_MAX_REQUESTS = 2_000
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


def load_matches() -> list[dict[str, str]]:
    if not MATCHES_CSV.exists():
        raise RuntimeError(f"Run fetch_matches.py first; {MATCHES_CSV} does not exist")
    with MATCHES_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{MATCHES_CSV} contains no matches")
    return rows


def raw_path(match: dict[str, str]) -> Path:
    return RAW_DIR / match["season"] / f"{match['match_id']}.json"


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


def normalize_saved_files(matches: list[dict[str, str]]) -> int:
    records: list[dict[str, Any]] = []
    for match in matches:
        path = raw_path(match)
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
        if PLAYER_STATS_CSV.exists():
            PLAYER_STATS_CSV.unlink()
        return 0

    extra_columns = sorted(
        {key for record in records for key in record}.difference(MATCH_CONTEXT_COLUMNS)
    )
    columns = MATCH_CONTEXT_COLUMNS + extra_columns
    temporary_path = PLAYER_STATS_CSV.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(PLAYER_STATS_CSV)
    return len(records)


def write_failures(failures: list[dict[str, str]]) -> None:
    if not failures:
        if FAILED_CSV.exists():
            FAILED_CSV.unlink()
        return
    with FAILED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["match_id", "season", "error"])
        writer.writeheader()
        writer.writerows(failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help=f"Seconds between requests (default: {DEFAULT_SLEEP_SECONDS})",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
        help=f"Maximum new HTTP requests in this run (default: {DEFAULT_MAX_REQUESTS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = load_matches()
    missing = [match for match in matches if not raw_path(match).exists()]
    print(f"Matches in list: {len(matches)}")
    print(f"Already saved: {len(matches) - len(missing)}")
    print(f"Still to fetch: {len(missing)}")

    if not missing:
        normalized = normalize_saved_files(matches)
        print(f"No API requests needed. Normalized {normalized} player-match rows.")
        return

    client = StatsApiClient(load_api_key(), args.sleep, args.max_requests)
    failures: list[dict[str, str]] = []

    try:
        for number, match in enumerate(missing, start=1):
            match_id = match["match_id"]
            try:
                payload = client.get(f"/football/matches/{match_id}/player-stats")
                save_json_atomic(raw_path(match), payload)
                player_count = len(extract_player_rows(payload))
                print(f"[{number}/{len(missing)}] {match_id}: saved {player_count} players")
            except ApiError as error:
                failures.append(
                    {"match_id": match_id, "season": match["season"], "error": str(error)}
                )
                print(f"[{number}/{len(missing)}] {match_id}: FAILED - {error}")
                if client.requests_used >= client.max_requests:
                    break
    except KeyboardInterrupt:
        print("\nStopped by user. Successfully saved match files will be reused next time.")

    write_failures(failures)
    normalized = normalize_saved_files(matches)
    saved = sum(raw_path(match).exists() for match in matches)
    print(f"Saved raw responses: {saved}/{len(matches)}")
    print(f"Normalized player-match rows: {normalized}")
    print(f"API requests used this run: {client.requests_used}")
    if failures:
        print(f"Failures written to {FAILED_CSV}")


if __name__ == "__main__":
    main()
