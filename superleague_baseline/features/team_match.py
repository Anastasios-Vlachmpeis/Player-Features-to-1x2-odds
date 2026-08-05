"""Team-level aggregates per match side."""

from __future__ import annotations

import pandas as pd


def build_side_skeleton(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Two rows per fixture: home side and away side."""
    home = fixtures.assign(
        player_team=fixtures["home_team"],
        opponent_team=fixtures["away_team"],
        venue="H",
    )
    away = fixtures.assign(
        player_team=fixtures["away_team"],
        opponent_team=fixtures["home_team"],
        venue="A",
    )
    cols = [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "player_team",
        "opponent_team",
        "venue",
    ]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def aggregate_lineup_team_match(lineup: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player lineup rows to one row per (match_id, player_team)."""
    df = lineup.copy()
    df["rating_weighted"] = df["rating"].fillna(0) * df["minutes_played"].fillna(0)
    g = df.groupby(["match_id", "match_date", "home_team", "away_team", "player_team"])
    agg = g.agg(
        gf_proxy=("goals", "sum"),
        key_passes=("key_passes", "sum"),
        passes_attempted=("total_passes", "sum"),
        passes_completed=("accurate_passes", "sum"),
        interceptions=("interceptions", "sum"),
        clearances=("clearances", "sum"),
        aerial_won=("aerial_won", "sum"),
        aerial_total=("aerial_total", "sum"),
        starters=("is_starter", "sum"),
        players_used=("minutes_played", lambda s: int((s.fillna(0) > 0).sum())),
        rating_num=("rating_weighted", "sum"),
        rating_den=("minutes_played", lambda s: float(s.fillna(0).sum())),
    ).reset_index()
    agg["lineup_complete"] = agg["starters"] == 11
    return agg


def aggregate_xg_team_match(xg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shooter xG rows by team (not player join)."""
    return (
        xg.groupby(["match_id", "player_team"], as_index=False)
        .agg(
            xg_for=("xg", "sum"),
            xgot_for=("xgot", "sum"),
            shots_for=("shots", "sum"),
            sot_for=("sot", "sum"),
        )
    )


def attach_opponent_metrics(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Add opponent gf/xg/etc. within the same match."""
    opp = team_matches[
        [
            "match_id",
            "player_team",
            "lineup_complete",
            "gf_proxy",
            "xg_for",
            "xgot_for",
            "shots_for",
            "sot_for",
        ]
    ].rename(
        columns={
            "player_team": "opponent_team",
            "lineup_complete": "opponent_lineup_complete",
            "gf_proxy": "ga_proxy",
            "xg_for": "xg_against",
            "xgot_for": "xgot_against",
            "shots_for": "shots_against",
            "sot_for": "sot_against",
        }
    )
    merged = team_matches.merge(opp, on=["match_id", "opponent_team"], how="left")
    merged["match_lineups_complete"] = (
        merged["lineup_complete"].fillna(False)
        & merged["opponent_lineup_complete"].fillna(False)
    )
    merged["points_proxy"] = merged.apply(_points_from_goals, axis=1)
    return merged


def _points_from_goals(row) -> float | None:
    if not row.get("match_lineups_complete") or pd.isna(row.get("ga_proxy")):
        return None
    gf, ga = row["gf_proxy"], row["ga_proxy"]
    if gf > ga:
        return 3.0
    if gf == ga:
        return 1.0
    return 0.0
