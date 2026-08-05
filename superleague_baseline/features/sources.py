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
