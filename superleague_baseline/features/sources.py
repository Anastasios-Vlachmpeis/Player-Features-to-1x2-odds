"""Load Sofascore tables from player_stats.db."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

LINEUP_COLS = [
    "sofascore_id",
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "player_team",
    "rating",
    "minutes_played",
    "goals",
    "assists",
    "key_passes",
    "total_passes",
    "accurate_passes",
    "interceptions",
    "clearances",
    "aerial_won",
    "aerial_total",
    "is_starter",
]

XG_COLS = [
    "sofascore_id",
    "match_id",
    "match_date",
    "player_team",
    "xg",
    "xgot",
    "shots",
    "sot",
]


def load_sofascore_sources(db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return lineup stats and xG rows from player_stats.db."""
    db_path = Path(db_path)
    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as conn:
        lineup = pd.read_sql_query(
            f"SELECT {', '.join(LINEUP_COLS)} FROM sofascore_match_stats",
            conn,
        )
        xg = pd.read_sql_query(
            f"SELECT {', '.join(XG_COLS)} FROM sofascore_xg",
            conn,
        )
    lineup["match_date"] = pd.to_datetime(lineup["match_date"])
    xg["match_date"] = pd.to_datetime(xg["match_date"])
    return lineup, xg


def build_fixture_index(lineup: pd.DataFrame) -> pd.DataFrame:
    """One row per match_id with canonical home/away metadata."""
    _validate_lineup_fixture_rows(lineup)
    meta = (
        lineup.groupby("match_id", as_index=False)
        .agg(
            match_date=("match_date", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
        )
        .sort_values(["match_date", "match_id"])
        .reset_index(drop=True)
    )
    return meta


def _validate_lineup_fixture_rows(lineup: pd.DataFrame) -> None:
    """Reject incomplete or contradictory fixture metadata before aggregation."""
    required = ["match_id", "match_date", "home_team", "away_team", "player_team"]
    if lineup[required].isna().any().any():
        raise ValueError("Lineup source contains null fixture metadata")

    if lineup.duplicated(["match_id", "sofascore_id"]).any():
        raise ValueError("Duplicate player rows found within a fixture")

    grouped = lineup.groupby("match_id", sort=False)
    inconsistent = grouped[["match_date", "home_team", "away_team"]].nunique(dropna=False)
    bad_metadata = inconsistent.ne(1).any(axis=1)

    expected_sides = grouped.apply(
        lambda rows: set(rows["player_team"]) == {rows["home_team"].iloc[0], rows["away_team"].iloc[0]},
        include_groups=False,
    )
    bad_ids = inconsistent.index[bad_metadata | ~expected_sides]
    if len(bad_ids):
        preview = bad_ids[:5].tolist()
        raise ValueError(f"Inconsistent or incomplete fixture rows for match_id(s): {preview}")
