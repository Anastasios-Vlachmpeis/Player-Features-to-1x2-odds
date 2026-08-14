#Validate the local Scotland match and player-stat backfill

from __future__ import annotations

import argparse
import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd

from feature_req import NPXG_FIELD, NPXG_MAX_SEASON_COVERAGE_DROP, NPXG_MIN_PLAYER_COVERAGE, USE_NPXG_FEATURE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "statsapi" / "scotland"
MATCHES_CSV = DATA_DIR / "matches.csv"
PLAYER_STATS_CSV = DATA_DIR / "player_match_stats.csv"
RAW_DIR = DATA_DIR / "player_stats_raw"
ODDS_DB = PROJECT_ROOT / "odds.db"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "scotland_data_validation"

DIVISION = "SC0"
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
PLAYER_DATA_THRESHOLD = 0.90
STARTER_COVERAGE_THRESHOLD = 0.90
IDENTITY_THRESHOLD = 0.99

CORE_FIELDS = [
    "minutes_played",
    "rating",
    "shooting.expected_goals",
    "shooting.expected_assists",
    "shooting.total_shots",
    NPXG_FIELD
]

MATCH_REQUIRED_COLUMNS = {
    "match_id",
    "season",
    "utc_date",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_score",
    "away_score",
}

PLAYER_REQUIRED_COLUMNS = {
    "match_id",
    "season",
    "player_id",
    "team_id",
    "started",
    "played",
    *CORE_FIELDS,
}

TEAM_ALIASES = {
    "hamiltonacademical": "hamilton",
    "heartofmidlothian": "hearts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for validation CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def normalize_team_name(name: object) -> str:
    value = unicodedata.normalize("NFKD", str(name or ""))
    value = value.encode("ascii", "ignore").decode("ascii").lower().strip()
    value = re.sub(r"\b(fc|cf|sc|afc|fk|sk)\b", "", value)
    value = re.sub(r"[^a-z0-9]", "", value)
    return TEAM_ALIASES.get(value, value)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.lower().map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    ).fillna(False)


def match_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["season"].astype("string")
        + "|"
        + frame["match_date"].astype("string")
        + "|"
        + frame["home_team"].map(normalize_team_name)
        + "|"
        + frame["away_team"].map(normalize_team_name)
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (MATCHES_CSV, PLAYER_STATS_CSV, ODDS_DB):
        require_file(path)

    matches = pd.read_csv(MATCHES_CSV, dtype="string", keep_default_na=True)
    players = pd.read_csv(PLAYER_STATS_CSV, dtype="string", keep_default_na=True)
    require_columns(matches, MATCH_REQUIRED_COLUMNS, MATCHES_CSV)
    require_columns(players, PLAYER_REQUIRED_COLUMNS, PLAYER_STATS_CSV)

    matches = matches[matches["season"].isin(SEASONS)].copy()
    players = players[players["season"].isin(SEASONS)].copy()
    matches["match_date"] = pd.to_datetime(matches["utc_date"], utc=True).dt.strftime("%Y-%m-%d")
    matches["home_score"] = pd.to_numeric(matches["home_score"], errors="coerce")
    matches["away_score"] = pd.to_numeric(matches["away_score"], errors="coerce")

    players["started"] = as_bool(players["started"])
    players["played"] = as_bool(players["played"])
    players["minutes_played"] = pd.to_numeric(players["minutes_played"], errors="coerce")
    for field in CORE_FIELDS[1:]:
        players[field] = pd.to_numeric(players[field], errors="coerce")

    placeholders = ",".join("?" for _ in SEASONS)
    query = f"""
        SELECT season, match_date, home_team, away_team,
               full_time_home, full_time_away, result_3way,
               odds_is_closing, home_odds, draw_odds, away_odds
        FROM historical_results_odds
        WHERE division = ? AND season IN ({placeholders})
    """
    with sqlite3.connect(ODDS_DB) as connection:
        football_data = pd.read_sql_query(query, connection, params=[DIVISION, *SEASONS])

    football_data["match_date"] = football_data["match_date"].astype("string")
    football_data["season"] = football_data["season"].astype("string")
    football_data["join_key"] = match_key(football_data)
    matches["join_key"] = match_key(matches)
    return matches, players, football_data


def find_player_duplicates(players: pd.DataFrame) -> pd.DataFrame:
    identified = players[players["player_id"].notna()].copy()
    mask = identified.duplicated(["match_id", "player_id"], keep=False)
    columns = [
        "season",
        "match_id",
        "player_id",
        "player_name",
        "team_id",
        "started",
        "minutes_played",
    ]
    existing = [column for column in columns if column in identified.columns]
    return identified.loc[mask, existing].sort_values(["season", "match_id", "player_id"])


def build_match_validation(
    matches: pd.DataFrame,
    players: pd.DataFrame,
    football_data: pd.DataFrame,
) -> pd.DataFrame:
    if matches["match_id"].duplicated().any():
        duplicate_ids = matches.loc[matches["match_id"].duplicated(False), "match_id"].tolist()
        raise ValueError(f"Duplicate match IDs in {MATCHES_CSV}: {duplicate_ids[:10]}")
    if football_data["join_key"].duplicated().any():
        raise ValueError("Football-Data contains duplicate Scotland season/date/team join keys")

    odds_columns = [
        "join_key",
        "full_time_home",
        "full_time_away",
        "result_3way",
        "odds_is_closing",
        "home_odds",
        "draw_odds",
        "away_odds",
    ]
    report = matches.merge(
        football_data[odds_columns],
        on="join_key",
        how="left",
        validate="one_to_one",
    )
    report["football_data_match"] = report["result_3way"].notna()
    report["closing_odds_available"] = (
        report["odds_is_closing"].fillna(0).astype(bool)
        & report[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1)
    )
    report["score_matches_football_data"] = (
        report["football_data_match"]
        & report["home_score"].eq(report["full_time_home"])
        & report["away_score"].eq(report["full_time_away"])
    )

    player_groups = players.groupby("match_id", sort=False)
    player_summary = player_groups.agg(
        player_rows=("player_id", "size"),
        players_with_id=("player_id", "count"),
        unique_player_ids=("player_id", "nunique"),
        played_players=("played", "sum"),
        starters=("started", "sum"),
    )

    starters = players[players["started"]].copy()
    starter_summary = starters.groupby("match_id", sort=False).agg(
        starters_with_id=("player_id", "count"),
        starters_with_minutes=("minutes_played", "count"),
    )
    report = report.merge(player_summary, left_on="match_id", right_index=True, how="left")
    report = report.merge(starter_summary, left_on="match_id", right_index=True, how="left")

    starter_counts = starters.groupby(["match_id", "team_id"], sort=False).size()
    report["home_starters"] = [
        int(starter_counts.get((match_id, team_id), 0))
        for match_id, team_id in zip(report["match_id"], report["home_team_id"])
    ]
    report["away_starters"] = [
        int(starter_counts.get((match_id, team_id), 0))
        for match_id, team_id in zip(report["match_id"], report["away_team_id"])
    ]

    count_columns = [
        "player_rows",
        "players_with_id",
        "unique_player_ids",
        "played_players",
        "starters",
        "starters_with_id",
        "starters_with_minutes",
    ]
    report[count_columns] = report[count_columns].fillna(0).astype(int)
    report["player_data_available"] = report["player_rows"] > 0
    report["twenty_two_starters"] = (
        report["home_starters"].eq(11) & report["away_starters"].eq(11)
    )
    report["starter_identity_complete"] = (
        report["starters"] > 0
    ) & report["starters_with_id"].eq(report["starters"])
    report["starter_minutes_complete"] = (
        report["starters"] > 0
    ) & report["starters_with_minutes"].eq(report["starters"])

    raw_paths = [RAW_DIR / season / f"{match_id}.json" for season, match_id in zip(report["season"], report["match_id"])]
    report["raw_json_exists"] = [path.exists() for path in raw_paths]

    valid_team_ids: dict[str, bool] = {}
    team_lookup = report.set_index("match_id")[["home_team_id", "away_team_id"]].to_dict("index")
    for match_id, group in players.groupby("match_id", sort=False):
        teams = team_lookup.get(match_id)
        if teams is None:
            valid_team_ids[match_id] = False
            continue
        expected = {teams["home_team_id"], teams["away_team_id"]}
        observed = set(group["team_id"].dropna())
        valid_team_ids[match_id] = bool(observed) and observed.issubset(expected)
    report["player_team_ids_valid"] = report["match_id"].map(valid_team_ids).fillna(False)

    report["model_ready"] = (
        report["football_data_match"]
        & report["closing_odds_available"]
        & report["score_matches_football_data"]
        & report["player_data_available"]
        & report["twenty_two_starters"]
        & report["starter_identity_complete"]
        & report["starter_minutes_complete"]
        & report["player_team_ids_valid"]
    )

    output_columns = [
        "season",
        "match_id",
        "match_date",
        "utc_date",
        "home_team_id",
        "home_team",
        "away_team_id",
        "away_team",
        "home_score",
        "away_score",
        "result_3way",
        "home_odds",
        "draw_odds",
        "away_odds",
        "football_data_match",
        "closing_odds_available",
        "score_matches_football_data",
        "raw_json_exists",
        "player_data_available",
        "player_rows",
        "unique_player_ids",
        "played_players",
        "starters",
        "home_starters",
        "away_starters",
        "starters_with_id",
        "starters_with_minutes",
        "twenty_two_starters",
        "starter_identity_complete",
        "starter_minutes_complete",
        "player_team_ids_valid",
        "model_ready",
    ]
    return report[output_columns].sort_values(["season", "utc_date", "match_id"])


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_season_coverage(match_report: pd.DataFrame, players: pd.DataFrame, duplicate_rows: pd.DataFrame) -> pd.DataFrame:
    
    rows: list[dict[str, object]] = []
    for season in SEASONS:
        matches = match_report[match_report["season"] == season]
        eligible = matches[matches["football_data_match"]]
        eligible_ids = set(eligible["match_id"])
        season_players = players[
            (players["season"] == season) & players["match_id"].isin(eligible_ids)
        ]
        starters = season_players[season_players["started"]]
        season_duplicates = duplicate_rows[duplicate_rows["season"] == season]

        rows.append(
            {
                "season": season,
                "statsapi_matches": len(matches),
                "football_data_top_division_matches": len(eligible),
                "additional_unmatched_matches": (~matches["football_data_match"]).sum(),
                "top_division_matches_with_player_data": eligible["player_data_available"].sum(),
                "player_match_coverage": safe_rate(
                    eligible["player_data_available"].sum(), len(eligible)
                ),
                "top_division_matches_with_22_starters": eligible["twenty_two_starters"].sum(),
                "twenty_two_starters_coverage": safe_rate(
                    eligible["twenty_two_starters"].sum(), len(eligible)
                ),
                "starter_rows": len(starters),
                "starters_with_id": starters["player_id"].notna().sum(),
                "starter_identity_coverage": safe_rate(
                    starters["player_id"].notna().sum(), len(starters)
                ),
                "starters_with_minutes": starters["minutes_played"].notna().sum(),
                "starter_minutes_coverage": safe_rate(
                    starters["minutes_played"].notna().sum(), len(starters)
                ),
                "matches_with_valid_player_team_ids": eligible["player_team_ids_valid"].sum(),
                "closing_odds_matches": eligible["closing_odds_available"].sum(),
                "score_agreement_matches": eligible["score_matches_football_data"].sum(),
                "duplicate_player_match_rows": len(season_duplicates),
                "model_ready_matches": eligible["model_ready"].sum(),
                "model_ready_coverage": safe_rate(eligible["model_ready"].sum(), len(eligible)),
            }
        )

    summary = pd.DataFrame(rows)
    summary["player_match_coverage_pass"] = summary["player_match_coverage"] >= PLAYER_DATA_THRESHOLD
    summary["twenty_two_starters_coverage_pass"] = (
        summary["twenty_two_starters_coverage"] >= STARTER_COVERAGE_THRESHOLD
    )
    summary["starter_identity_coverage_pass"] = (
        summary["starter_identity_coverage"] >= IDENTITY_THRESHOLD
    )
    summary["starter_minutes_coverage_pass"] = (
        summary["starter_minutes_coverage"] >= STARTER_COVERAGE_THRESHOLD
    )
    return summary


def build_field_coverage(match_report: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    
    eligible_ids = set(match_report.loc[match_report["football_data_match"], "match_id"])
    eligible_players = players[players["match_id"].isin(eligible_ids)].copy()

    # Calculate the percentage of matches with sufficiently populated npxG for both teams
    npxg_starters = eligible_players[eligible_players["started"]].copy()
    npxg_team_coverage = npxg_starters.groupby(["season", "match_id", "team_id"])[NPXG_FIELD].apply(lambda values: values.notna().mean())
    npxg_team_usable = npxg_team_coverage.ge(NPXG_MIN_PLAYER_COVERAGE)
    npxg_match_groups = npxg_team_usable.groupby(["season", "match_id"])
    npxg_match_usable = npxg_match_groups.all() & npxg_match_groups.size().eq(2)
    npxg_match_coverage = npxg_match_usable.groupby("season").mean()
    
    rows: list[dict[str, object]] = []

    for season in SEASONS:
        season_players = eligible_players[eligible_players["season"] == season]
        cohorts = {
            "played_players": season_players[season_players["played"]],
            "starters": season_players[season_players["started"]],
        }
        for cohort_name, cohort in cohorts.items():
            for field in CORE_FIELDS:
                numeric = pd.to_numeric(cohort[field], errors="coerce")
                populated = numeric.notna().sum()
                nonzero = numeric.fillna(0.0).ne(0.0).sum()
                
                rows.append(
                    {
                        "season": season,
                        "cohort": cohort_name,
                        "field": field,
                        "rows": len(cohort),
                        "populated_rows": populated,
                        "coverage": safe_rate(populated, len(cohort)),
                        "nonzero_rows": nonzero,
                        "nonzero_rate": safe_rate(nonzero, len(cohort)),
                        "mean": numeric.mean(),
                        "std": numeric.std(),
                        "usable_match_coverage": npxg_match_coverage.get(season, 0.0) if field == NPXG_FIELD else float("nan"),
                    }
                )

    return pd.DataFrame(rows)


def validate_npxg_coverage(field_coverage: pd.DataFrame) -> None:

    # Validate played-player npxG coverage before allowing the feature into modelling
    npxg = field_coverage[(field_coverage["cohort"] == "played_players") & (field_coverage["field"] == NPXG_FIELD)].set_index("season")["coverage"].reindex(SEASONS)
    low_coverage = npxg[npxg < NPXG_MIN_PLAYER_COVERAGE]

    coverage_drop = (npxg.shift(1) - npxg).dropna()
    collapsed = coverage_drop[coverage_drop > NPXG_MAX_SEASON_COVERAGE_DROP]

    if low_coverage.empty and collapsed.empty: return
    
    message = f"npxG coverage contract failed; low_coverage={low_coverage.to_dict()}, collapses={collapsed.to_dict()}"
    
    if USE_NPXG_FEATURE: raise ValueError(message)
    
    print(f"WARNING: {message}; npxG remains disabled")

def write_reports(output_dir: Path) -> dict[str, Path]:
    matches, players, football_data = load_inputs()
    duplicate_rows = find_player_duplicates(players)
    match_report = build_match_validation(matches, players, football_data)
    season_coverage = build_season_coverage(match_report, players, duplicate_rows)
    field_coverage = build_field_coverage(match_report, players)

    missing_player_stats = match_report[~match_report["player_data_available"]].copy()
    additional_matches = match_report[~match_report["football_data_match"]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        "season_coverage": output_dir / "season_coverage.csv",
        "field_coverage": output_dir / "field_coverage.csv",
        "match_validation": output_dir / "match_validation.csv",
        "missing_player_stats": output_dir / "missing_player_stats.csv",
        "additional_unmatched_matches": output_dir / "additional_unmatched_matches.csv",
        "duplicate_player_match_rows": output_dir / "duplicate_player_match_rows.csv",
    }

    season_coverage.to_csv(reports["season_coverage"], index=False)
    field_coverage.to_csv(reports["field_coverage"], index=False)
    validate_npxg_coverage(field_coverage)
    match_report.to_csv(reports["match_validation"], index=False)
    missing_player_stats.to_csv(reports["missing_player_stats"], index=False)
    additional_matches.to_csv(reports["additional_unmatched_matches"], index=False)
    duplicate_rows.to_csv(reports["duplicate_player_match_rows"], index=False)

    print(season_coverage.to_string(index=False))
    print(f"\nMissing player-stat matches: {len(missing_player_stats)}")
    print(f"Additional unmatched fixtures: {len(additional_matches)}")
    print(f"Duplicate player-match rows: {len(duplicate_rows)}")
    print(f"Reports written to: {output_dir}")
    return reports


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    write_reports(output_dir)


if __name__ == "__main__":
    main()
